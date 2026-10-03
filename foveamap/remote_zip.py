"""Read members of a zip file on an HTTP server without downloading all of it.

The KITTI odometry Lidar zip is ~85 GB, more than a Colab disk holds next to
its contents, but training only needs a subsample of scans. `zipfile` works on
any seekable file object, so `HTTPRangeFile` serves its reads with HTTP range
requests: the central directory once, then each wanted member's bytes.

Trust boundary: this is an internal research data-loader helper, NOT a public
API. The ``url`` argument is trusted operator configuration (e.g. a known
dataset mirror), not externally controlled user input. To prevent accidental
SSRF/resource abuse if wired to untrusted input, only ``http``/``https`` URLs
are accepted and ``file://``, ``ftp://``, ``gopher://`` etc. are rejected.
Callers exposing URLs to untrusted users must add their own allow-list,
authentication, and size/timeout budgeting on top.
"""
from __future__ import annotations

import http.client
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

# transient failures worth another try: timeouts, dropped connections, TLS handshake stalls
RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)


def _validate_url(url: str) -> str:
    """Reject non-HTTP(S) URLs so this helper cannot become a file/SSRF boundary."""
    parsed = urllib.parse.urlparse(str(url))
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Refusing to open non-HTTP(S) URL {url!r}: only http/https dataset URLs are allowed "
            "(this helper must not be wired directly to untrusted user input)."
        )
    if not parsed.hostname:
        raise ValueError(f"Refusing to open URL with no host: {url!r}")
    return str(url)


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only view of a URL (the server must honour Range requests)."""

    def __init__(self, url, block=1 << 20, timeout=60, retries=6, backoff=2.0,
                 max_size: int | None = None):
        self.url, self.block, self.timeout = _validate_url(url), block, timeout
        self.retries, self.backoff = retries, backoff
        self.pos = 0
        size = self._retry(lambda: self._open("bytes=0-0").headers["Content-Range"])
        self.size = int(size.split("/")[1])
        if max_size is not None and self.size > max_size:
            raise OSError(
                f"{self.url}: remote size {self.size} bytes exceeds max_size {max_size} "
                "(refusing a potentially unbounded download)"
            )
        self._buf_start, self._buf = 0, b""

    def _open(self, rng):
        req = urllib.request.Request(self.url, headers={"Range": rng})
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        if resp.status != 206:
            raise OSError(f"{self.url}: server ignored the Range header (HTTP {resp.status})")
        return resp

    def _retry(self, fn):
        """fn(), retried with exponential backoff on transient network errors."""
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except RETRYABLE as e:
                if isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429:
                    raise                                   # 4xx won't fix itself
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * 2 ** attempt)

    def _get(self, start, stop):
        """Bytes [start, stop) of the URL, checked for length."""
        def once():
            with self._open(f"bytes={start}-{stop - 1}") as resp:
                data = resp.read()
            if len(data) != stop - start:
                raise http.client.IncompleteRead(data, stop - start - len(data))
            return data
        return self._retry(once)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n
        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self.pos and end <= buf_end):
            # read ahead a block: a member's local header and its data usually arrive in one request
            stop = min(self.size, max(end, self.pos + self.block))
            self._buf = self._get(self.pos, stop)
            self._buf_start = self.pos
        out = self._buf[self.pos - self._buf_start:end - self._buf_start]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_remote_zip(url, **kw):
    """zipfile.ZipFile over HTTP. Use one per thread: the file object keeps a position."""
    return zipfile.ZipFile(HTTPRangeFile(_validate_url(url), **kw))

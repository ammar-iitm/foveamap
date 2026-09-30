"""Read members of a zip file on an HTTP server without downloading all of it.

The KITTI odometry Lidar zip is ~85 GB, more than a Colab disk holds next to
its contents, but training only needs a subsample of scans. `zipfile` works on
any seekable file object, so `HTTPRangeFile` serves its reads with HTTP range
requests: the central directory once, then each wanted member's bytes.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only view of a URL (the server must honour Range requests)."""

    def __init__(self, url, block=1 << 20, timeout=60):
        self.url, self.block, self.timeout = url, block, timeout
        self.pos = 0
        self.size = int(self._open("bytes=0-0").headers["Content-Range"].split("/")[1])
        self._buf_start, self._buf = 0, b""

    def _open(self, rng):
        req = urllib.request.Request(self.url, headers={"Range": rng})
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        if resp.status != 206:
            raise OSError(f"{self.url}: server ignored the Range header (HTTP {resp.status})")
        return resp

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
            with self._open(f"bytes={self.pos}-{stop - 1}") as resp:
                self._buf = resp.read()
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
    return zipfile.ZipFile(HTTPRangeFile(url, **kw))

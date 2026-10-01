import io
import urllib.error
import zipfile

import pytest

from foveamap.remote_zip import HTTPRangeFile


class FlakyRangeFile(HTTPRangeFile):
    """Serves ranges of an in-memory blob; the first `fail` requests raise `error`."""

    def __init__(self, blob, fail=0, error=TimeoutError("handshake timed out"), **kw):
        self.blob, self.fail, self.error, self.requests = blob, fail, error, 0
        super().__init__("http://example.invalid/x.zip", backoff=0, **kw)

    def _open(self, rng):
        self.requests += 1
        if self.requests <= self.fail:
            raise self.error
        start, stop = (int(v) for v in rng.split("=")[1].split("-"))
        blob, size = self.blob[start:stop + 1], len(self.blob)

        class Resp(io.BytesIO):
            headers = {"Content-Range": f"bytes {start}-{stop}/{size}"}
        return Resp(blob)


def make_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.bin", b"\x01" * 5000)
        z.writestr("b.bin", b"\x02" * 3000)
    return buf.getvalue()


def test_reads_members_through_transient_failures():
    f = FlakyRangeFile(make_zip(), fail=3, block=1024)
    with zipfile.ZipFile(f) as z:
        assert z.read("a.bin") == b"\x01" * 5000
        assert z.read("b.bin") == b"\x02" * 3000


def test_gives_up_after_the_retry_budget():
    with pytest.raises(TimeoutError):
        FlakyRangeFile(make_zip(), fail=10, retries=2)


def test_client_errors_are_not_retried():
    err = urllib.error.HTTPError("http://example.invalid/x.zip", 404, "Not Found", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        FlakyRangeFile(make_zip(), fail=1, error=err, retries=5)

"""P2 regression: input/security boundaries (remote URL, NPY, malformed)."""
import numpy as np
import pytest

from foveamap.remote_zip import HTTPRangeFile, open_remote_zip, _validate_url
from foveamap.data.file import parse_npy
from foveamap.core.exceptions import DataAdapterError


def test_remote_url_rejects_non_http():
    for bad in ["file:///etc/passwd", "ftp://example.com/a.zip", "gopher://x/y", "not-a-url", ""]:
        with pytest.raises((ValueError, Exception)):
            _validate_url(bad)
    # http/https accepted (validation only, no network).
    assert _validate_url("https://example.com/data.zip").startswith("https://")
    assert _validate_url("http://example.com/data.zip").startswith("http://")


def test_npy_rejects_pickle_payload(tmp_path):
    # Object arrays require pickle; allow_pickle=False must reject them.
    p = tmp_path / "evil.npy"
    np.save(str(p), np.array({"a": 1}, dtype=object), allow_pickle=True)
    with pytest.raises(DataAdapterError):
        parse_npy(str(p))


def test_npy_rejects_bad_shape(tmp_path):
    p = tmp_path / "bad.npy"
    np.save(str(p), np.ones((5,), dtype=np.float32))
    with pytest.raises(DataAdapterError):
        parse_npy(str(p))

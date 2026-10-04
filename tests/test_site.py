"""The static dashboard site Vercel builds (vercel.json -> scripts/build_site.py -> site/)."""
import base64
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_site import build, data_dirs  # noqa: E402


def test_vercel_config_serves_the_built_site():
    cfg = json.load(open(os.path.join(ROOT, "vercel.json")))
    assert cfg["framework"] is None and cfg["installCommand"] == ""      # not a Python app
    assert cfg["buildCommand"] == "python3 scripts/build_site.py" and cfg["outputDirectory"] == "site"


def test_site_data_is_complete():
    dirs = data_dirs(os.path.join(ROOT, "site"))
    assert dirs
    for d in dirs:
        M = json.load(open(os.path.join(d, "metrics.json")))
        frames = sorted(f for f in os.listdir(os.path.join(d, "frames")) if f.endswith(".png"))
        assert frames == [f"f{t:03d}.png" for t in range(len(M["frames"]))]
        blob = base64.b64decode(open(os.path.join(d, "points.b64.txt")).read())
        off, n = M["summary"]["points_file_index"][-1]
        assert len(blob) == off + 5 * n              # int16 x, int16 y, uint8 class per point


def test_dataset_listing_is_well_formed():
    listing = os.path.join(ROOT, "site", "data", "datasets.json")
    L = json.load(open(listing))
    assert [set(d) >= {"id", "label"} for d in L] == [True] * len(L)
    assert len({d["id"] for d in L}) == len(L)


def test_build_checks_every_listed_dataset(tmp_path):
    shutil.copytree(os.path.join(ROOT, "site", "data"), tmp_path / "data")
    L = json.load(open(tmp_path / "data" / "datasets.json"))
    json.dump(L + [{"id": "missing", "label": "Missing"}], open(tmp_path / "data" / "datasets.json", "w"))
    with pytest.raises(FileNotFoundError, match="missing"):
        build(str(tmp_path))


def test_build_wraps_the_dashboard(tmp_path):
    shutil.copytree(os.path.join(ROOT, "site", "data"), tmp_path / "data")
    page = open(build(str(tmp_path)), encoding="utf-8").read()
    body = open(os.path.join(ROOT, "dashboard", "index.html"), encoding="utf-8").read()
    assert page.startswith("<!doctype html>") and body in page and page.endswith("</body></html>")
    assert "data/datasets.json" in page and "/metrics.json" in page


def test_build_refuses_without_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="site/data|data/metrics.json"):
        build(str(tmp_path))

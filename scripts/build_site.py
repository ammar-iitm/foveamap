"""Build the static dashboard site that Vercel serves (see vercel.json).

    python scripts/build_site.py            # writes site/index.html next to site/data/

dashboard/index.html is a page fragment; this wraps it in a full HTML page.
The replays the site shows live in site/data/<id>/ (metrics.json,
points.b64.txt, frames/*.png), listed in site/data/datasets.json as
[{"id", "label", "title"}]; the first is shown by default and ?data=<id>
picks another. They are kept apart from dashboard/data/, which local
benchmark runs overwrite. To publish a run, copy its dashboard data into
site/data/<id>/, list it in datasets.json and push. (Without datasets.json,
a single run's data in site/data/ itself works too.)
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def wrap(body):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style></head><body>'
            + body + "</body></html>")


def data_dirs(out_dir):
    """The replay folders the site serves: those listed in data/datasets.json, or data/ itself."""
    listing = os.path.join(out_dir, "data", "datasets.json")
    if not os.path.exists(listing):
        return [os.path.join(out_dir, "data")]
    with open(listing, encoding="utf-8") as fh:
        ids = [d["id"] for d in json.load(fh)]
    if not ids:
        raise ValueError(f"{listing} lists no datasets")
    return [os.path.join(out_dir, "data", i) for i in ids]


def build(out_dir=os.path.join(ROOT, "site")):
    with open(os.path.join(ROOT, "dashboard", "index.html"), encoding="utf-8") as fh:
        page = wrap(fh.read())
    for d in data_dirs(out_dir):
        for need in ("metrics.json", "points.b64.txt", "frames"):
            if not os.path.exists(os.path.join(d, need)):
                raise FileNotFoundError(f"{d}/{need} is missing: copy a run's dashboard data into {d}")
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path


if __name__ == "__main__":
    print("wrote", build())

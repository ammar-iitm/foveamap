"""Build the static dashboard site that Vercel serves (see vercel.json).

    python scripts/build_site.py            # writes site/index.html next to site/data/

dashboard/index.html is a page fragment; this wraps it in a full HTML page.
The replay data the site shows lives in site/data/ (metrics.json,
points.b64.txt, frames/*.png), kept apart from dashboard/data/, which local
benchmark runs overwrite. To publish a different run, copy that run's
dashboard data into site/data/ and push.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def wrap(body):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style></head><body>'
            + body + "</body></html>")


def build(out_dir=os.path.join(ROOT, "site")):
    with open(os.path.join(ROOT, "dashboard", "index.html"), encoding="utf-8") as fh:
        page = wrap(fh.read())
    for need in ("metrics.json", "points.b64.txt", "frames"):
        if not os.path.exists(os.path.join(out_dir, "data", need)):
            raise FileNotFoundError(f"{out_dir}/data/{need} is missing: copy a run's dashboard data into {out_dir}/data")
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path


if __name__ == "__main__":
    print("wrote", build())

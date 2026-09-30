"""Wrap dashboard/index.html in a full HTML page so it can be opened locally.

    python scripts/make_local_view.py && cd dashboard && python -m http.server 8000
    # then open http://localhost:8000/view.html
"""
import os

D = os.path.join(os.path.dirname(__file__), "..", "dashboard")
body = open(os.path.join(D, "index.html"), encoding="utf-8").read()
page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style></head><body>'
        + body + "</body></html>")
with open(os.path.join(D, "view.html"), "w", encoding="utf-8") as fh:
    fh.write(page)
print("wrote", os.path.join(D, "view.html"))

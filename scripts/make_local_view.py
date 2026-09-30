"""Wrap dashboard/index.html in a full HTML page so it can be opened locally.

    python scripts/make_local_view.py && cd dashboard && python -m http.server 8000
    # then open http://localhost:8000/view.html
"""
import os

from build_site import wrap

D = os.path.join(os.path.dirname(__file__), "..", "dashboard")
body = open(os.path.join(D, "index.html"), encoding="utf-8").read()
with open(os.path.join(D, "view.html"), "w", encoding="utf-8") as fh:
    fh.write(wrap(body))
print("wrote", os.path.join(D, "view.html"))

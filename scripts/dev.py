"""FoveaMap developer command surface.

Single, memorable entry point for common development loops:

    python scripts/dev.py test        # full suite
    python scripts/dev.py test-fast   # CPU suite minus hardware-gated tests
    python scripts/dev.py test-unit   # adversarial + hardening subset
    python scripts/dev.py demo        # canonical 3-frame demo
    python scripts/dev.py info        # system diagnostics
    python scripts/dev.py bench       # short mapping benchmark
    python scripts/dev.py lint        # ruff check (if installed) + compile check
    python scripts/dev.py clean       # remove caches and build artifacts

Prefer this over memorizing ad-hoc pytest/CLI invocations. Every command
maps to a documented workflow in DEVELOPMENT.md.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print(f"+ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def cmd_test(_: list[str]) -> int:
    return run([sys.executable, "-m", "pytest", "-q"])


def cmd_test_fast(_: list[str]) -> int:
    # Hardware-gated tests skip on their own, but deselecting them makes the
    # fast loop explicit and keeps CPU CI honest about what it covers.
    return run([sys.executable, "-m", "pytest", "-q", "-m", "not cuda and not ros2 and not slow"])


def cmd_test_unit(_: list[str]) -> int:
    return run([
        sys.executable, "-m", "pytest", "-q",
        "tests/test_hardening_p0.py",
        "tests/test_hardening_p1.py",
        "tests/test_hardening_p2.py",
        "tests/test_hardening_p3.py",
        "tests/test_hardening_p4_adversarial.py",
        "tests/test_core_foundation.py",
        "tests/test_public_api_contracts.py",
    ])


def cmd_demo(_: list[str]) -> int:
    return run([sys.executable, "-m", "foveamap.cli", "demo", "--frames", "3"])


def cmd_info(_: list[str]) -> int:
    return run([sys.executable, "-m", "foveamap.cli", "info"])


def cmd_bench(_: list[str]) -> int:
    return run([sys.executable, "-m", "foveamap.cli", "bench",
                "--engine", "numpy", "--frames", "20"])


def cmd_lint(_: list[str]) -> int:
    rc = run([sys.executable, "-m", "compileall", "-q", "foveamap", "tests",
              "scripts/dev.py"])
    if shutil.which("ruff") is None:
        print("ruff not installed; compile check done. "
              "Install dev extras for ruff: pip install -e \".[dev]\"")
        return rc
    # Scoped to the new developer surface: repo-wide ruff adoption on the
    # pre-existing codebase is tracked as remaining work (see
    # DEVELOPMENT_READINESS_REPORT.md), so `lint` must not fail on legacy debt.
    rc2 = run([sys.executable, "-m", "ruff", "check", "scripts/dev.py"])
    return rc or rc2


def cmd_clean(_: list[str]) -> int:
    for name in ("__pycache__", ".pytest_cache", ".ruff_cache"):
        for p in ROOT.rglob(name):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    for p in ROOT.glob("*.egg-info"):
        shutil.rmtree(p, ignore_errors=True)
    print("cleaned caches")
    return 0


COMMANDS = {
    "test": cmd_test,
    "test-fast": cmd_test_fast,
    "test-unit": cmd_test_unit,
    "demo": cmd_demo,
    "info": cmd_info,
    "bench": cmd_bench,
    "lint": cmd_lint,
    "clean": cmd_clean,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print("usage: python scripts/dev.py {" + ",".join(sorted(COMMANDS)) + "}")
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

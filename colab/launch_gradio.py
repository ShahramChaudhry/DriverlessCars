"""Shim — implementation is in repo-root launch_gradio.py (works after git clone)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("_driverless_launch", _ROOT / "launch_gradio.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

main = _mod.main

if __name__ == "__main__":
    main()

"""Colab launch helper — fetch latest code, purge stale imports, start Gradio."""
from __future__ import annotations

import importlib
import os
import py_compile
import subprocess
import sys

REPO = os.environ.get("DRIVERLESS_REPO", "/content/DriverlessCars")
BRANCH = os.environ.get("DRIVERLESS_BRANCH", "claude/autonomous-driving-demo-Ljmeg")
PY_FILES = ("config.py", "model_loader.py", "inference.py", "benchmark.py", "app.py")
RELOAD_ORDER = ("config", "model_loader", "pruning", "inference", "benchmark", "app")


def git_head() -> str:
    r = subprocess.run(
        ["git", "-C", REPO, "log", "-1", "--oneline"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def refresh_from_github() -> None:
    if not os.path.isdir(os.path.join(REPO, ".git")):
        print("No git repo — re-run the clone cell above.")
        return
    print("Before:", git_head() or "(unknown)")
    fetch = subprocess.run(
        ["git", "-C", REPO, "fetch", "origin", BRANCH, "--depth", "1"],
        capture_output=True,
        text=True,
    )
    if fetch.returncode != 0:
        print("git fetch failed — re-run the clone cell.\n", fetch.stderr.strip())
        return
    reset = subprocess.run(
        ["git", "-C", REPO, "reset", "--hard", f"origin/{BRANCH}"],
        capture_output=True,
        text=True,
    )
    if reset.returncode != 0:
        print("git reset failed:\n", reset.stderr.strip())
        return
    print("After:", git_head())


def compile_project() -> None:
    for name in PY_FILES:
        py_compile.compile(os.path.join(REPO, name), doraise=True)
    print("Syntax OK:", ", ".join(PY_FILES))


def close_previous_demo() -> None:
    old_app = sys.modules.get("app")
    if old_app is None:
        return
    demo = getattr(old_app, "demo", None)
    if demo is None:
        return
    try:
        demo.close()
        print("Closed previous Gradio server.")
    except Exception as exc:
        print(f"Note: could not close previous server ({exc}). Restart runtime if the link is stale.")


def reload_project_modules():
    """Drop cached project modules so git-pulled config.py is re-read."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    for name in reversed(RELOAD_ORDER):
        sys.modules.pop(name, None)
    for name in RELOAD_ORDER:
        importlib.import_module(name)
    return sys.modules["app"]


def main(*, share: bool = True) -> None:
    os.chdir(REPO)
    refresh_from_github()
    compile_project()
    close_previous_demo()
    app = reload_project_modules()
    app.launch_gradio(share=share)


if __name__ == "__main__":
    main()

"""Utility script that ensures execution inside a configured virtual
environment and runs the configured application entry file.

The script reads optional settings from ``setup-config.json`` and, when
necessary, re-executes itself using the virtual environment's Python
interpreter before running the target file as ``__main__``.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

CONFIG_FILE = "setup-config.json"


def read_cfg() -> dict:
    """Load settings from ``setup-config.json`` if present; otherwise defaults.

    Returns:
        dict: Mapping with keys:
            - ``venv_dir`` (str): Virtual environment directory name.
            - ``main_file`` (str): Entry file to run.

    Notes:
        If the config file is missing or contains invalid JSON, default values
        are used.
    """
    cfg = {"venv_dir": "venv", "main_file": "app.py"}
    p = Path(CONFIG_FILE)
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            # If the file is corrupt or invalid JSON, fall back to defaults.
            pass
    return cfg


def venv_python_path(venv_dir: str | os.PathLike) -> Path:
    """Return the path to the Python interpreter inside a virtual environment.

    Args:
        venv_dir (str | os.PathLike): The virtual environment directory.

    Returns:
        Path: Platform-specific path to the interpreter inside the venv.
    """
    v = Path(venv_dir)
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def is_inside_this_venv(venv_dir: Path) -> bool:
    """Determine whether the current process runs inside the given venv.

    Args:
        venv_dir (Path): Path to the target virtual environment directory.

    Returns:
        bool: True if the current interpreter belongs to ``venv_dir``.
    """
    try:
        return Path(sys.prefix).resolve() == venv_dir.resolve()
    except Exception:
        return False


def same_interpreter(py: Path) -> bool:
    """Check if the current interpreter matches the given interpreter path.

    Args:
        py (Path): The expected Python interpreter path.

    Returns:
        bool: True if ``sys.executable`` resolves to ``py``.
    """
    try:
        return Path(sys.executable).resolve() == py.resolve()
    except Exception:
        return False


def reexec_into_venv(venv_dir: str | os.PathLike) -> None:
    """Re-execute the process using the target venv's interpreter if needed.

    This replaces the current process with the virtual environment's Python
    using ``os.execv`` when the process is not already running inside it.

    Args:
        venv_dir (str | os.PathLike): Target virtual environment directory.

    Environment Variables:
        PRO_VENV_SKIP_REEXEC: If set to "1", skip the re-exec behavior.

    Notes:
        If the venv interpreter does not exist, this function returns without
        attempting to re-exec.

        To create the venv automatically when missing, you may uncomment the
        optional block below.
    """
    if os.environ.get("PRO_VENV_SKIP_REEXEC") == "1":
        return

    venv_dir = Path(venv_dir)
    py = venv_python_path(venv_dir)

    # Optional: create the virtual environment automatically if it does not
    # exist. Commented out to avoid side effects by default.
    # if not venv_dir.exists():
    #     import subprocess
    #     subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
    #                    check=True)

    if not py.exists():
        # No venv interpreter found; continue without re-exec.
        return

    if not is_inside_this_venv(venv_dir) or not same_interpreter(py):
        # execv replaces the current process (cleaner than subprocess + exit).
        os.execv(str(py), [str(py), *sys.argv])


def main() -> None:
    """Program entry point.

    Steps:
        1. Load configuration.
        2. Ensure execution inside the desired virtual environment.
        3. Execute the configured target file as ``__main__``.
    """
    cfg = read_cfg()

    # Enter or remain inside the correct venv interpreter.
    reexec_into_venv(cfg.get("venv_dir", "venv"))

    # After this point the process should be running inside the target venv.
    target = cfg.get("main_file", "app.py")
    target_path = Path(target)

    if not target_path.exists():
        print(f"{target} does not exist.")
        sys.exit(1)

    print(f"Running: {target}")
    # Run the target file as if it were __main__.
    runpy.run_path(str(target_path), run_name="__main__")


if __name__ == "__main__":
    main()

"""abrun: CPU/GPU A/B runner for Quantum ESPRESSO on this M2 Max.

Runs the reference OpenBLAS pw.x and the frozen apple-bottom pw.x builds (AB_MODE cpu / auto / gpu)
on the same input, under the house machine etiquette, and records energies, QE wall times and the
library's own dispatch counters.  Wall times are observations; no speedup is ever computed.

The package imports examples/common/exlib.py by path (never moved, never copied) and reuses its
constants and parsers.  Everything here is stdlib + numpy (exlib's import).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VERSION = "0.1"

PKG_DIR = Path(__file__).resolve().parent                       # tools/abrun
REPO_ROOT = PKG_DIR.parents[1]                                  # rainbow-connection checkout
EXLIB_DIR = REPO_ROOT / "examples" / "common"
if str(EXLIB_DIR) not in sys.path:
    sys.path.insert(0, str(EXLIB_DIR))
import exlib  # noqa: E402  (examples/common/exlib.py, imported by path)

HOME = Path.home()
RUNS_DIR = Path(os.environ.get("ABRUN_RUNS_DIR", str(PKG_DIR / "runs"))).expanduser()
DEFAULT_FROZEN = "33fbaa8c8944"
AB_FROZEN_ROOT = HOME / "qe-test" / "ab-frozen"
QE_BUILDS_ROOT = HOME / "qe-test" / "builds"
LIVE_LIBRARY_DIRS = (HOME / "Dev" / "Claude" / "apple-bottom" / "build", HOME / "apple-bottom" / "build")
MPIRUN = Path("/opt/homebrew/bin/mpirun")
MACHLOCK_PYTHON = Path("/opt/homebrew/bin/python3")             # the interpreter exlib uses for machlock
SHIM = PKG_DIR / "shim.py"
INTERPRETER = sys.executable

__all__ = ["VERSION", "PKG_DIR", "REPO_ROOT", "EXLIB_DIR", "exlib", "HOME", "RUNS_DIR", "DEFAULT_FROZEN",
           "AB_FROZEN_ROOT", "QE_BUILDS_ROOT", "LIVE_LIBRARY_DIRS", "MPIRUN", "MACHLOCK_PYTHON", "SHIM",
           "INTERPRETER"]

#!/usr/bin/env python3
"""abrun launch shim: runs INSIDE the machine lock, right before the solver.

    python shim.py --run-dir <cwd> --nice 10 -- <argv0 (absolute)> <args...>

0. re-checks ~/Dev/.timing-lock (ABRUN_TIMING_LOCK overrides the path, tests only): the runner waited for
   it before the (possibly 20 min) machlock wait, so it may have appeared since.  If it exists the shim
   exits 75 WITHOUT writing machine.json or starting the solver; machlock releases the lock and the runner
   waits for the timing lock again and relaunches.  It never polls inside the lock: the timing-lock holder
   may itself be waiting for machlock, which would deadlock both sessions;
1. writes <cwd>/machine.json atomically (the timing-grade snapshot at lock acquisition: time, load1,
   power, ~/Dev/.timing-lock presence, foreign solver pids, own pgid);
2. os.nice(<n>) (this process is single-threaded, so this is safe and inherited by mpirun and the ranks;
   /usr/bin/nice is never used because SIP strips DYLD_* from what it launches);
3. os.execv(argv0, argv): no extra process stays between machlock and mpirun, so signals reach them.
   A missing or non-executable argv0 is one stderr line and exit 127, not a traceback.

stdlib only, no threads, no imports from the abrun package (it must run under any python3).
"""

import json
import os
import subprocess
import sys
import tempfile
import time

SOLVERS = {"pw.x", "ph.x", "epw.x", "yambo", "p2y", "ypp", "mpirun", "prterun", "orterun"}
TIMING_LOCK = os.environ.get("ABRUN_TIMING_LOCK") or os.path.join(os.path.expanduser("~"), "Dev", ".timing-lock")
TIMING_LOCK_RC = 75          # EX_TEMPFAIL: the runner (runner.SHIM_TIMING_LOCK_RC) waits and relaunches


def power_state():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if "AC Power" in out:
        return "AC"
    if "Battery Power" in out:
        return "Battery"
    return "unknown"


def foreign_solvers(own_pgid):
    """[[pid, command]...] of solver processes outside our own process group."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,pgid=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, pgid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        cmd = parts[2]
        exe = os.path.basename(cmd.split()[0]) if cmd.split() else ""
        if exe in SOLVERS and pgid != own_pgid:
            found.append([pid, cmd[:160]])
    return found


def main(argv):
    run_dir, nice_n, rest = None, 10, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--run-dir":
            run_dir = argv[i + 1]; i += 2
        elif a == "--nice":
            nice_n = int(argv[i + 1]); i += 2
        elif a == "--":
            rest = argv[i + 1:]; break
        else:
            sys.stderr.write(f"shim: unexpected argument {a!r}\n"); return 2
    if not run_dir or not rest:
        sys.stderr.write(__doc__); return 2
    if not os.path.isabs(rest[0]):
        sys.stderr.write(f"shim: argv[0] must be absolute: {rest[0]!r}\n"); return 2
    if os.path.exists(TIMING_LOCK):
        try:
            with open(TIMING_LOCK, errors="ignore") as fh:
                text = fh.read().strip()[:80]
        except OSError:
            text = ""
        sys.stderr.write(f"shim: timing lock present inside the machine lock ({TIMING_LOCK}: {text!r}) -> "
                         f"releasing the lock without launching (exit {TIMING_LOCK_RC})\n")
        sys.stderr.flush()
        return TIMING_LOCK_RC
    pgid = os.getpgrp()
    snap = {
        "t_acquired": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "t_acquired_epoch": time.time(),
        "load1": os.getloadavg()[0],
        "power": power_state(),
        "timing_lock_present": os.path.exists(TIMING_LOCK),
        "foreign_solvers": foreign_solvers(pgid),
        "pgid": pgid,
        "pid": os.getpid(),
        "nice": nice_n,
    }
    fd, tmp = tempfile.mkstemp(prefix=".machine.", suffix=".json", dir=run_dir)
    with os.fdopen(fd, "w") as fh:
        json.dump(snap, fh, indent=1)
    os.chmod(tmp, 0o644)                      # mkstemp creates 0600; the record must be readable like the rest
    os.replace(tmp, os.path.join(run_dir, "machine.json"))
    try:
        os.nice(nice_n)
    except OSError as exc:
        sys.stderr.write(f"shim: os.nice({nice_n}) failed: {exc}\n")
    sys.stderr.flush()
    try:
        os.execv(rest[0], rest)
    except OSError as exc:
        sys.stderr.write(f"shim: cannot exec {rest[0]}: {exc} (exit 127)\n")
        return 127
    return 1  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

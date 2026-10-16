#!/usr/bin/env python3
"""A stand-in for ~/qe-test/xp/tools/machlock.py in the cancel tests: same argv shape, same job launch.

    fake_machlock.py bench|build [--max-load L] -- <cmd...>

Like machlock it runs the job with `subprocess.call(cmd)`, whose KeyboardInterrupt handling SIGKILLs the
child (the launcher) before mpirun can abort its ranks: that is the orphaning the runner's cancel must
never trigger, so this wrapper must never receive the SIGINT.  Takes no lock (the tests point
launch.MACHLOCK here; runner.machlock_bench_held() sees no machine.lock beside it).  Logs its own
acquisition/release lines to stderr in machlock's words so a test can assert on 'release'.
"""

import subprocess
import sys
import time


def main(argv):
    if not argv or argv[0] not in ("bench", "build"):
        sys.stderr.write(__doc__)
        return 2
    kind = argv[0]
    a = argv[1:]
    if a[:1] == ["--max-load"]:
        a = a[2:]
    if a[:1] != ["--"] or len(a) < 2:
        sys.stderr.write(__doc__)
        return 2
    cmd = a[1:]
    t0 = time.time()
    sys.stderr.write(f"fake_machlock: got {kind} after 0s\n")
    sys.stderr.flush()
    rc = 1
    try:
        rc = subprocess.call(cmd)
    finally:
        sys.stderr.write(f"fake_machlock: release {kind} rc={rc} after {time.time() - t0:.0f}s\n")
        sys.stderr.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

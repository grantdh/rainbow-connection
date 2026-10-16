#!/usr/bin/env python3
"""A stand-in for /opt/homebrew/bin/mpirun (prterun) in the runner tests: no MPI, no solver.

    fake_mpirun.py -np N [-x NAME=value ...] <solver argv...>

Spawns N copies of the solver argv, each in its OWN process group (PRRTE does setpgid on every rank, which
is why a killpg on the launcher's group never reaches them) with ABRUN_FAKE_RANK=i in its environment and
the -x pairs applied.  Signals behave like prterun's abort: SIGINT waits ABRUN_FAKE_LAUNCHER_ABORT_S
(default 0.5 s, prterun's measured ~0.56 s) and then SIGTERMs the ranks; SIGTERM does so at once; the ranks
get SIGKILL if they survive 2 s.  Exit status: 0 when every rank exits 0, else 1; 130 after an abort.
A SIGKILL to this launcher (what machlock's subprocess.call does on KeyboardInterrupt) leaves the ranks
running: the orphaning the runner's cancel is tested against.  ABRUN_FAKE_LAUNCHER_DEAF=1 makes the
launcher ignore SIGINT and SIGTERM and never stop its ranks (the runner's SIGKILL escalation test).
"""

import os
import signal
import subprocess
import sys
import time

_children = []
_aborting = {"sig": None}


def _abort(signum, frame):
    if _aborting["sig"] is None:
        _aborting["sig"] = signum


def _stop_ranks(grace_s):
    for p in _children:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
            except OSError:
                pass
    t0 = time.time()
    while any(p.poll() is None for p in _children) and time.time() - t0 < grace_s:
        time.sleep(0.05)
    for p in _children:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass
    for p in _children:
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def main(argv):
    n, env, i = 1, dict(os.environ), 0
    while i < len(argv):
        if argv[i] == "-np":
            n = int(argv[i + 1]); i += 2
        elif argv[i] == "-x":
            k, _, v = argv[i + 1].partition("="); env[k] = v; i += 2
        else:
            break
    solver = argv[i:]
    if not solver:
        sys.stderr.write(__doc__)
        return 2
    deaf = os.environ.get("ABRUN_FAKE_LAUNCHER_DEAF") == "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN if deaf else _abort)
    signal.signal(signal.SIGTERM, signal.SIG_IGN if deaf else _abort)
    for rank in range(n):
        e = dict(env, ABRUN_FAKE_RANK=str(rank))
        _children.append(subprocess.Popen(solver, env=e, preexec_fn=os.setpgrp))    # own pgid, like PRRTE
    sys.stderr.write(f"fake_mpirun[{os.getpid()}]: {n} rank(s) {[p.pid for p in _children]}\n")
    sys.stderr.flush()
    while any(p.poll() is None for p in _children):
        if _aborting["sig"] is not None:
            if _aborting["sig"] == signal.SIGINT:
                time.sleep(float(os.environ.get("ABRUN_FAKE_LAUNCHER_ABORT_S", "0.5")))
            sys.stderr.write(f"fake_mpirun[{os.getpid()}]: abort on signal {_aborting['sig']}\n")
            sys.stderr.flush()
            _stop_ranks(2.0)
            return 130
        time.sleep(0.05)
    return 0 if all(p.returncode == 0 for p in _children) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

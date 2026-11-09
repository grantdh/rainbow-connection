"""Paths, labelled inputs, sampling, pre-registration and solver etiquette for the Si dipole power-split example.

The device data (raw .s1p files, geometry, input values, predictions, results) stay in the private repository and are
read by path: set NM_REPRO to its checkout (default ~/Dev/Claude/nm-antenna-reproduction). Nothing device-specific is
written under this public directory: file names, geometry, stack values, run flags and record text come from device.json there,
and the deep stage's write-ups' prose from templates/.
"""
import datetime as dt
import fcntl
import functools
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
NM = Path(os.environ.get("NM_REPRO", HOME / "Dev/Claude/nm-antenna-reproduction"))
WF = NM / "workflows/nm-meander-dipole"
PS = WF / "power-split"
PREREG = PS / "prereg"
OUT = PS / "out"
STOPS = OUT / "openems_stop_steps.json"     # written by the runs, committed with their results
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exlib import MACHLOCK, TIMING_LOCK  # noqa: E402
SEED = 20261106


def require_private_repo():
    if not (PREREG / "inputs.json").exists():
        sys.exit(f"private data not found: {PREREG}/inputs.json. Set NM_REPRO to the nm-antenna-reproduction checkout.")
    OUT.mkdir(parents=True, exist_ok=True)


def inputs():
    return json.load(open(PREREG / "inputs.json"))


@functools.cache
def device():
    require_private_repo()
    return json.load(open(PS / "device.json"))


def template(name):
    """The prose of a private write-up (templates/<name>), with str.format fields for the computed values."""
    require_private_repo()
    return (PS / "templates" / name).read_text()


def _argv_key(argv):
    return hashlib.sha256(json.dumps([str(a) for a in argv]).encode()).hexdigest()[:12]


def replay(argv, tag):
    """openEMS argv for one run. openEMS tests its end criterion on a wall-clock interval, so where a run stops depends on machine load. A run that stopped on
    the criterion has its step recorded (record_stop); with the same arguments it is replayed for exactly that many steps, which reproduces it bit for bit."""
    rec = (json.loads(STOPS.read_text()) if STOPS.exists() else {}).get(tag)
    extra = ["--max-ts", str(rec["steps"]), "--end-db", "-400"] if rec and rec["argv"] == _argv_key(argv) else []
    return [*argv, *extra, "--tag", tag]


def record_stop(argv, tag, log):
    """After an openEMS run: store the step at which it stopped on its end criterion. A run that reaches --max-ts is deterministic already, and so is a replay."""
    m = re.search(r"end-criteria of \S+ reached after (\d+) timesteps", Path(log).read_text(errors="ignore"))
    if m:
        rec = json.loads(STOPS.read_text()) if STOPS.exists() else {}
        rec[tag] = {"steps": int(m[1]), "argv": _argv_key(argv)}
        STOPS.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def import_private(module, sub="materials"):
    sys.path.insert(0, str(WF / sub))
    return __import__(module)


# ---------------------------------------------------------------------------- input sampling

def _draw(rng, d, n):
    base, dist = d["value"], d["dist"]
    k = dist["kind"]
    if k == "normal":
        return rng.normal(base, dist["sd"], n)
    if k == "rel_normal":
        return rng.normal(1.0, dist["sd"], n) * (base if np.isscalar(base) and not isinstance(base, str) else 1.0)
    if k == "uniform":
        return rng.uniform(dist["lo"], dist["hi"], n)
    if k == "uniform_scale":
        return rng.uniform(dist["lo"], dist["hi"], n)
    if k == "loguniform":
        return np.exp(rng.uniform(np.log(dist["lo"]), np.log(dist["hi"]), n))
    raise ValueError(k)


def sample_inputs(n=20000, seed=SEED):
    """Named arrays of the inputs that carry a distribution. No RF value enters. The same seed gives the same samples."""
    inp, rng, s = inputs(), np.random.default_rng(seed), {}
    for sec in ("geometry", "electrical", "thermal", "light"):
        for key, d in inp[sec].items():
            if isinstance(d, dict) and "dist" in d:
                s[key] = _draw(rng, d, n)
    s["power_amps"] = rng.random(n) < 0.5                 # the two endpoint power laws
    s["tau_exponent"] = rng.uniform(0.0, 4.0, n)
    s["tau_activated"] = rng.random(n) < 0.25             # a quarter of the samples use the activated form
    s["tau_Ea_eV"] = rng.uniform(0.05, 0.40, n)
    return s


def laser_power(k, amps, ith=None):
    """Output power (W) at setting k under the two endpoint laws of inputs.json; ith overrides the amps law's threshold (A). Arrays broadcast."""
    d = device()["laser"]
    ith = d["i_th_A"] if ith is None else ith
    pa = d["p_max_W"] * np.maximum(k - ith, 0.0) / (d["i_max_A"] - ith)
    return np.where(amps, pa, d["tenth_W_per_setting"] * k)


def percentiles(a):
    return {f"q{p:g}": float(np.percentile(a, p)) for p in (2.5, 16, 50, 84, 97.5)}


# ---------------------------------------------------------------------------- raw-file metadata

def stamp(path):
    for line in open(path, errors="replace"):
        m = re.match(r"!\s*VAR\s+MeasDate=(.*)", line.strip())
        if m:
            return dt.datetime.strptime(m.group(1).strip(), "%m/%d/%Y %I:%M:%S %p")
    raise ValueError(path)


def raw_day(name):
    """A measurement day's settings from device.json, with its folder and its device's folder resolved."""
    d = dict(device()["raw"][name])
    d["root"] = NM / d["root"]
    d["dev"] = d["root"] / d["dev"]
    return d


def timeline(day):
    """[(file, setting, seconds after the dark sweep)] of a day's series, in time order. Only the header dates are used. X: dark, 1-7, the 10 V repeat of 7,
    dark after. Ka: ambient before, dark, 1-7, ambient after (no dark after was taken)."""
    d = raw_day(day)
    names = [(d["dark"], 0)] + [(d["lit"].format(k=k), k) for k in range(1, 8)]
    names += [(d["bias"], 7), (d["after"], 0)] if day == "X" else [(d["ambient"][0], 0), (d["ambient"][1], 0)]
    t0 = stamp(d["dev"] / d["dark"])
    return sorted(((n, k, (stamp(d["dev"] / n) - t0).total_seconds()) for n, k in names), key=lambda r: r[2])


def x_timeline():
    return timeline("X")


# ---------------------------------------------------------------------------- pre-registration

def git(*a, repo=NM):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()


def code_hash(*files):
    h = hashlib.sha256()
    for f in sorted(map(str, files)):
        h.update(Path(f).read_bytes())  # file contents only: paths differ between machines
    return h.hexdigest()[:12]


def _same(a, b, tol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a == b or abs(a - b) <= tol * max(abs(a), abs(b))
    return a == b


def register(name, payload, code=()):
    """Write a prediction file. It has to be committed by hand (explicit staging) before any comparison runs. A rerun that reproduces the committed
    numbers to 1 part in 10^6 leaves the file untouched, so a clean-checkout rerun passes; a rerun that differs dirties it, and validate.py then refuses."""
    payload = json.loads(json.dumps(payload, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    p = PREREG / f"{name}.json"
    if p.exists():
        old = json.load(open(p))
        stamp_keys = ("registered", "code_sha256", "inputs_sha256")
        if _same({k: v for k, v in old.items() if k not in stamp_keys}, payload):
            print(f"reproduced {p}")
            return p
    payload = {"registered": dt.datetime.now().isoformat(timespec="seconds"), "code_sha256": code_hash(*code) if code else None,
               "inputs_sha256": code_hash(PREREG / "inputs.json"), **payload}
    p.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"registered {p}")
    return p


def arrays_hash(arrays):
    """A hash of named arrays' values, independent of the file that holds them."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.asarray(arrays[k]).tobytes())
    return h.hexdigest()[:16]


def require_registered(name):
    """The commit that holds a prediction file. Refuses when it is uncommitted or dirty, or was made from other inputs: comparisons come after commits."""
    path = PREREG / f"{name}.json"
    rel = path.relative_to(NM)
    if git("status", "--porcelain", "--", str(rel)):
        sys.exit(f"{rel} is uncommitted or modified: commit the prediction before computing its comparison")
    if git("rev-parse", "--is-shallow-repository") == "true":
        return "shallow clone"          # the registering commit is outside this clone's history; git fetch --unshallow shows it
    sha = git("log", "-1", "--format=%h", "--", str(rel))
    if not sha:
        sys.exit(f"{rel} has never been committed")
    if json.load(open(path)).get("inputs_sha256") != code_hash(PREREG / "inputs.json"):
        sys.exit(f"{rel} was registered from a different inputs.json: run its stage again and commit the result")
    return sha


# ---------------------------------------------------------------------------- solver etiquette

def bench_running():
    """True while another session's machlock benchmark holds the machine exclusively."""
    lock = MACHLOCK.resolve().parent / "machine.lock"
    if not lock.exists():
        return False
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _stopper(pgid, stop, yield_to_bench):
    """SIGSTOP the solver's process group while ~/Dev/.timing-lock exists and, in yield mode, while a benchmark holds the machine."""
    held = False
    while not stop.wait(5):
        want = TIMING_LOCK.exists() or (yield_to_bench and bench_running())
        if want != held:
            os.killpg(pgid, signal.SIGSTOP if want else signal.SIGCONT)
            held = want


def run_solver(cmd, cwd, log, lock="build", env=None):
    """Run one solver: nice'd, after the timing lock clears, frozen if it reappears. lock = 'build' for epw.x (machlock's shared lock: a frozen epw.x would hold up a benchmark, which waits for any
    foreign epw.x to exit); 'yield' for the other solvers: no lock is taken, they start when no benchmark holds the machine and freeze while one does; None for neither. SIGTERM reaches the solver,
    which runs in its own session. Records load."""
    yielding = lock == "yield"
    while TIMING_LOCK.exists() or (yielding and bench_running()):
        time.sleep(15)
    argv = ["nice", "-n", "10"]
    if lock == "build" and MACHLOCK.exists():
        argv += [sys.executable, str(MACHLOCK), lock, "--"]
    argv += [str(x) for x in cmd]
    t0, load0 = time.time(), os.getloadavg()[0]
    with open(log, "w") as f:
        p = subprocess.Popen(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, start_new_session=True)
        signal.signal(signal.SIGTERM, lambda *_: (os.killpg(p.pid, signal.SIGCONT), os.killpg(p.pid, signal.SIGTERM), sys.exit(143)))
        stop = threading.Event()
        threading.Thread(target=_stopper, args=(p.pid, stop, yielding), daemon=True).start()
        rc = p.wait()
        stop.set()
    rec = {"cmd": " ".join(map(str, cmd))[:200], "rc": rc, "wall_s": round(time.time() - t0, 1), "load1_start": round(load0, 1),
           "load1_end": round(os.getloadavg()[0], 1), "when": dt.datetime.now().isoformat(timespec="seconds")}
    with open(OUT / "solver_runs.log", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rc

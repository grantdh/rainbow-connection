"""Queue of the openEMS power-split runs named in device.json: dark across the handle resistivity range, lit at nominal assumed inputs, membrane what-ifs, and one mesh step for u_num.
Each launch: nice, no machine lock taken; it starts when no benchmark holds the machine and freezes while one does (correctness runs may share, PROTOCOL section 9). Analysis follows each run, and a
run whose fields are already on disk goes straight to analysis.
  fullwave_queue.py [name ...]    default: every run
"""
import json
import sys

import common as c

PY = c.HOME / ".venvs/openems/bin/python"
RUNS = c.device()["fullwave_runs"]


def main():
    c.require_private_repo()
    logs = c.OUT / "fullwave"
    logs.mkdir(exist_ok=True)
    for name in (sys.argv[1:] or RUNS):
        done = logs / f"{name}.json"
        if done.exists():
            print("done", name, flush=True)
            continue
        res = c.WF / "results" / name
        if not ((res / "port_at_dump.npz").exists() and (res / "sim/Evol.h5").exists()):
            rc = c.run_solver([PY, c.HERE / "fullwave_openems.py", "run", *RUNS[name], "--tag", name], c.HERE, logs / f"{name}.log", lock="yield")
            print(name, "run rc", rc, flush=True)
            if rc != 0:
                continue
        rc = c.run_solver([PY, c.HERE / "fullwave_openems.py", "analyse", "--tag", name], c.HERE, logs / f"{name}.analysis", lock=None)
        print(name, "analyse rc", rc, flush=True)
        print(open(logs / f"{name}.analysis").read(), flush=True)
        if rc == 0 and done.exists() and all(abs(r["balance"]) < 0.05 for r in json.load(open(done))["freqs"].values()):
            for p in (res / "sim").glob("[EHJ]*.h5"):                 # keep the dumps of a run that fails its balance
                p.unlink()


if __name__ == "__main__":
    main()

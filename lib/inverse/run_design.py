"""Headless optimization with the live page's continuation policy; writes the same design JSON as "Download design".

    ~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/run_design.py demux --set bandwidth=0.02 --out demux.json

Then check it:  lib/inverse/meep_check.py demux.json --res 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devices  # noqa: E402
import optimize  # noqa: E402


def run(kind: str, params: dict, radius: float = 0.12, beta: float = 4.0, lr: float = 0.03,
        max_iter: int = 2000, init: str = "uniform", quiet: bool = False) -> optimize.Optimizer:
    opt = optimize.Optimizer(devices.build(kind, **params), radius=radius, beta=beta, lr=lr, init=init)
    cont = optimize.Continuation()
    t0 = time.perf_counter()
    for _ in range(max_iter):
        rec = opt.step()
        state = cont.update(opt, rec)
        if not quiet and (state != "running" or rec["it"] % 50 == 0):
            print(f"it {rec['it']:4d} beta {opt.beta:5.1f} loss {rec['loss']:.2e} gray {rec['gray']:.3f} {state}")
        if state == "converged":
            break
    if not quiet:
        print(f"{opt.t} iterations in {time.perf_counter() - t0:.1f} s")
    return opt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=sorted(devices.DEFAULTS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="device parameter (um)")
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    params = {**devices.DEFAULTS[a.kind], **{k: float(v) for k, v in (s.split("=", 1) for s in a.set)}}
    opt = run(a.kind, params, radius=a.radius, max_iter=a.max_iter)
    design = opt.export()
    for k, v in design["binary"]["T"].items():
        print(f"binary {k}: {v:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(design))


if __name__ == "__main__":
    main()

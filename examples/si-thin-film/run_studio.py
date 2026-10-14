"""Replay Example 1 through muppet-studio's graph executor (headless, no browser).

Run with the studio's own interpreter:
    ~/Dev/Claude/muppet-studio/.venv/bin/python run_studio.py

The graph is the same chain as run.py expressed as ComfyUI-style nodes. The executor caches
every node by input hash under ~/.muppet-studio/runs, so a second run is instant. Output:
results/studio/{events.json, tmm_spectrum.json, compare.json}.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIO = Path(os.environ.get("MUPPET_STUDIO", Path.home() / "Dev/Claude/muppet-studio"))
sys.path.insert(0, str(STUDIO / "server"))
os.environ.setdefault("SF_ROOT", str(HERE.parents[1]))

from muppet_studio.graph import nodes  # noqa: E402,F401
from muppet_studio.graph.executor import Executor  # noqa: E402

OUT = HERE / "results" / "studio"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = {
    "1": {"class_type": "MUPPET_Crystal", "inputs": {"material": "Si (diamond)", "lattice_bohr": 0.0}},
    "2": {"class_type": "MUPPET_QE_SCF", "inputs": {"structure": ["1", 0], "ecutwfc": 30.0, "kgrid": 4, "conv_thr": "1.0d-8",
                                                   "executable": "qe (PATH, OpenBLAS, no apple-bottom)", "ab_mode": "off", "mpi_ranks": 1}},
    "3": {"class_type": "MUPPET_QE_NSCF", "inputs": {"scf": ["2", 0], "nbnd": 20, "kgrid": 4,
                                                    "executable": "qe (PATH, OpenBLAS, no apple-bottom)", "ab_mode": "off", "mpi_ranks": 1}},
    "4": {"class_type": "MUPPET_P2Y", "inputs": {"nscf": ["3", 0]}},
    "5": {"class_type": "MUPPET_YamboOpticsIP", "inputs": {"save": ["4", 0], "bands": 20, "e_max_eV": 8.0, "steps": 400,
                                                          "broadening_eV": 0.1, "scissor_eV": 0.65}},
    "6": {"class_type": "MUPPET_LorentzFit", "inputs": {"epsilon": ["5", 0], "poles": 4, "e_min_eV": 0.0, "e_max_eV": 0.0}},
    "7": {"class_type": "MUPPET_ThinFilmTMM", "inputs": {"material": ["6", 0], "thickness_nm": 500.0, "e_min_eV": 1.0,
                                                        "e_max_eV": 4.0, "points": 300}},
    "8": {"class_type": "MUPPET_Plot", "inputs": {"spectrum": ["7", 0]}},
    "9": {"class_type": "MUPPET_Plot", "inputs": {"epsilon": ["5", 0], "material": ["6", 0]}},
}

done = threading.Event()
events: list = []
t0 = time.time()


def emit(kind, data):
    events.append({"t": round(time.time() - t0, 2), "kind": kind, "data": data})
    if kind == "executing" and data.get("node"):
        print(f"[{time.strftime('%X')}] executing node {data['node']} ({PROMPT[data['node']]['class_type']})", flush=True)
    elif kind == "execution_cached":
        print(f"[{time.strftime('%X')}] cached {data['nodes']}", flush=True)
    elif kind == "execution_error":
        print("ERROR", data.get("node_type"), data.get("exception_message"), flush=True)
    elif kind == "muppet.log":
        print("   ", str(data.get("text", ""))[:160], flush=True)
    elif kind == "muppet.done":
        print("DONE", data, flush=True)
        done.set()


ex = Executor(emit=emit)
ex.start()
prompt_id, _ = ex.submit(PROMPT)
done.wait()
ok = any(e["kind"] == "muppet.done" and e["data"].get("status") == "success" for e in events)
json.dump(events, open(OUT / "events.json", "w"), indent=1, default=str)

# Node outputs live in the executor's in-memory cache keyed by input signature; history keeps only UI outputs.
spectrum = material = None
for out in ex._cache.values():
    if isinstance(out, tuple) and out and isinstance(out[0], dict):
        d = out[0]
        if "series" in d and "T" in d.get("series", {}):
            spectrum = d
        if "poles" in d and "eps_inf" in d:
            material = d
if spectrum:
    json.dump(spectrum, open(OUT / "tmm_spectrum.json", "w"))
if material:
    json.dump(material, open(OUT / "material.json", "w"))
compare = {"status": "success" if ok else "failed", "wall_s": round(time.time() - t0, 1), "prompt_id": prompt_id}
script = HERE / "results" / "summary.json"
if spectrum and material and script.exists():
    import numpy as np
    s = json.loads(script.read_text())
    x = np.array(spectrum["x"]); T = np.array(spectrum["series"]["T"]); R = np.array(spectrum["series"]["R"])
    Ts = np.interp(x, s["airy"]["energy_eV"], s["airy"]["T"]); Rs = np.interp(x, s["airy"]["energy_eV"], s["airy"]["R"])
    compare.update({"studio_eps_inf": material["eps_inf"], "script_eps_inf": s["material"]["eps_inf"],
                    "max_dT_vs_script": float(np.abs(T - Ts).max()), "max_dR_vs_script": float(np.abs(R - Rs).max())})
    print(f"studio vs script: eps_inf {material['eps_inf']:.3f} vs {s['material']['eps_inf']:.3f}; "
          f"max|dT| {compare['max_dT_vs_script']:.4f}, max|dR| {compare['max_dR_vs_script']:.4f}")
json.dump(compare, open(OUT / "compare.json", "w"), indent=2)
print("studio replay:", compare["status"], f"in {compare['wall_s']} s")
sys.exit(0 if ok else 1)

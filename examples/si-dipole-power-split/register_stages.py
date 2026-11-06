"""Collect solver output into the registered prediction files, with no measured value in sight.

  register_stages.py fullwave     prereg/04_fullwave.json: split per run, q(x), u_num from the mesh step, identities
  register_stages.py fem-run      the Palace meshes and solves
  register_stages.py fem          prereg/05_fem.json: Y, P_rad and heat against openEMS
"""
import json
import subprocess
import sys

import numpy as np

import common as c
import fem_palace as fp

R_SQ_PAD, PAD_SQUARES = fp.STACK["pad_r_sq_ohm"], fp.STACK["pad_squares"]      # pad metal, two pads, the current spreading into the oxide capacitor


def read_run(tag):
    p = c.OUT / "fullwave" / f"{tag}.json"
    return json.load(open(p)) if p.exists() else None


def split(run):
    rows = {}
    for f, r in run["freqs"].items():
        w = 2 * np.pi * float(f) * 1e9
        pin = r["P_in"]
        y = complex(*r["Y_S"])
        i2 = abs(complex(*r["I"])) ** 2
        rows[f] = {"Y_mS": [y.real * 1e3, y.imag * 1e3], "rad": r["P_rad_A"] / pin, "rad_box_B": r["P_rad_B"] / pin, "handle": r["P_handle"] / pin, "sheet": r["P_sheet"] / pin,
                   "box": 0.5 * w * fp.EPS0 * fp.EPS_BOX * fp.TAN_BOX * r["int_E2_dV_box"] / pin, "pads": 0.5 * i2 * R_SQ_PAD * PAD_SQUARES / pin, "balance": r["balance"]}
    return rows


def q_along(run, f="10"):
    r = run["freqs"][f]
    x, q = np.array(r["q_handle_x"][0]), np.array(r["q_handle_x"][1]) / r["P_in"]
    xs, qs = np.array(r["q_sheet_x"][0]), np.array(r["q_sheet_x"][1]) / r["P_in"]
    w = np.gradient(x) * 1e3
    cum = np.cumsum(np.sort(q)[::-1]) / q.sum()
    order = np.argsort(-q)
    half = np.abs(x[order][: int(np.searchsorted(cum, 0.5)) + 1]).max() * 1e3
    ninety = np.abs(x[order][: int(np.searchsorted(cum, 0.9)) + 1]).max() * 1e3
    pick = np.linspace(0, len(x) - 1, 200).astype(int)
    return {"x_m": x[pick].tolist(), "handle_per_mm": (q / w)[pick].tolist(), "sheet_per_mm": np.interp(x[pick], xs, qs / (np.gradient(xs) * 1e3)).tolist(), "half_of_handle_heat_within_mm": float(half), "ninety_pct_within_mm": float(ninety)}


def fullwave():
    c.require_private_repo()
    tags = [p.stem for p in sorted((c.OUT / "fullwave").glob("ps_*.json"))]
    runs = {t: read_run(t) for t in tags}
    gates, table = [], {}
    for t, r in runs.items():
        table[t] = split(r)
        for f, row in table[t].items():
            gates.append({"name": f"energy balance, {t}, {f} GHz", "value": row["balance"] + row["box"] + row["pads"], "limit": 0.05, "note": "P_rad + P_heat vs P_in"})
            if row["rad"] > 1e-5:
                gates.append({"name": f"P_rad does not depend on the box, {t}, {f} GHz", "value": row["rad_box_B"] / row["rad"] - 1, "limit": 0.10, "note": "boxes 5 and 9 mm off the chip"})
    # numerics: one mesh step at constant grid phase, 25 -> 12.5 um (every drawn face is a multiple of 25 um, pads and oxide on lines in both)
    u_num = {}
    fine, coarse = runs.get("ps_dark_r10"), runs.get("ps_dark_r10_m25")
    if fine and coarse:
        for day, f in (("X", "10"), ("Ka", "31.5")):
            dy = complex(*coarse["freqs"][f]["Y_S"]) - complex(*fine["freqs"][f]["Y_S"])
            # first-order mesh error assumed for a conservative bound: the difference itself
            u_num[f"G_{day}"] = abs(dy.real)
            u_num[f"B_{day}"] = abs(dy.imag)
            u_num[f"rad_{day}"] = abs(coarse["freqs"][f]["P_rad_A"] / coarse["freqs"][f]["P_in"] - fine["freqs"][f]["P_rad_A"] / fine["freqs"][f]["P_in"])
    res = {"what": "openEMS power split; model result: one-port S11 fixes only total Re(Y)", "split": table, "u_num": u_num, "gates": gates,
           "q_along_x": q_along(fine) if fine else None, "q_along_x_lit_s7": q_along(runs["ps_lit_r10_w10_s7"]) if "ps_lit_r10_w10_s7" in runs else None,
           "q_along_x_31p5GHz": q_along(fine, "31.5") if fine else None}
    s7 = {w: runs.get(f"ps_lit_r10_w{w}_s7") for w in (5, 10, 30)}
    dark = runs.get("ps_dark_r10")
    if dark and all(s7.values()):
        dy = {w: complex(*s7[w]["freqs"]["10"]["Y_S"]) - complex(*dark["freqs"]["10"]["Y_S"]) for w in s7}
        res["u_input_lit_s7_rel"] = float(np.std([abs(v) for v in dy.values()]) / abs(dy[10]))
    rec = c.OUT / "reciprocity.json"
    if rec.exists():
        z = json.load(open(rec))
        gates.append({"name": "Lorentz reciprocity, |S21/S12 - 1|, 6-35 GHz", "value": z["max_rel_S21_S12_6_35GHz"], "limit": 0.02, "note": "antenna port and a port in the handle"})
        for p in z["kirchhoff_form"]:
            gates.append({"name": f"Kirchhoff form: transmit E predicts Z21, {p['f']} GHz", "value": p["rel"], "limit": 0.10, "note": "the heat map weights thermal currents at the port"})
    c.register("04_fullwave", res, code=[__file__, c.HERE / "fullwave_openems.py", c.HERE / "fullwave_queue.py", c.HERE / "common.py"])
    for t in table:
        print(t, json.dumps(table[t]["10"])[:240])
    print("u_num", u_num)
    print("bad gates:", [g for g in gates if abs(g["value"]) > g["limit"]])


FEM_RUNS = [("m30", 0.03, 2, 10), ("m30", 0.03, 2, 3), ("m30", 0.03, 2, 30), ("m15", 0.015, 2, 10), ("m30", 0.03, 3, 10), ("m30a", 0.03, 2, 10)]   # m30a: the dark meander arms as a sheet impedance


def fem_run():
    c.require_private_repo()
    py = sys.executable
    for name, lc, order, rho in FEM_RUNS:
        d = c.OUT / "fem" / name
        arms = ["--arms"] if name.endswith("a") else []
        if not (d / "dip.msh").exists():
            subprocess.run([py, c.HERE / "fem_palace.py", "mesh", "--name", name, "--lc-pad", str(lc), *arms], check=True)
        if not (d / f"post_r{rho}_o{order}" / "port-S.csv").exists():
            subprocess.run([py, c.HERE / "fem_palace.py", "solve", "--name", name, "--rho", str(rho), "--order", str(order), "--f", "8.25,10,11.75,31.5", *arms], check=True)


def fem():
    c.require_private_repo()
    table, gates = {}, []
    fw = json.load(open(c.PREREG / "04_fullwave.json"))["split"]
    for name, lc, order, rho in FEM_RUNS:
        r = fp.results(name, rho, order)
        table[f"{name}_o{order}_r{rho}"] = r
        for f, row in r.items():
            gates.append({"name": f"FEM energy balance, {name} order {order}, rho {rho}, {f} GHz", "value": row["balance"], "limit": 0.05, "note": ""})
    # the method-class comparison: Palace on its finest vs openEMS
    best = table["m15_o2_r10"]
    other = table["m30_o3_r10"]
    cmp_ = {}
    for f in ("10", "31.5"):
        a, b = complex(*best[f]["Y_S"]), complex(*fw["ps_dark_r10"][f]["Y_mS"]) * 1e-3
        cmp_[f] = {"Y_palace_mS": [a.real * 1e3, a.imag * 1e3], "Y_openems_mS": [b.real * 1e3, b.imag * 1e3], "rel_diff": abs(a - b) / abs(b), "rad_palace": best[f]["rad"], "rad_openems": fw["ps_dark_r10"][f]["rad"],
                   "h_vs_p_rel": abs(complex(*other[f]["Y_S"]) - a) / abs(a)}
    a, b = complex(*table["m30a_o2_r10"]["10"]["Y_S"]), complex(*table["m30_o2_r10"]["10"]["Y_S"])
    res = {"what": "Palace dark Y, P_rad, Joule heat", "table": table, "vs_openems": cmp_, "gates": gates, "dark_arms_change_Y_by": abs(a - b) / abs(b)}
    c.register("05_fem", res, code=[__file__, c.HERE / "fem_palace.py", c.HERE / "common.py"])
    print(json.dumps(cmp_, indent=0))


if __name__ == "__main__":
    {"fullwave": fullwave, "fem-run": fem_run, "fem": fem}[sys.argv[1]]()

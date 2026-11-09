"""Flip map: what heating would need to reproduce the X-day turn-on, and whether the dark sweep after the series allows it. Rules in prereg/00d_addendum.md (A);
registers prereg/07_flip_map.json, maps in OUT/flip_map.npz.

H_T  the handle's own sigma(T) moves the dark admittance: dY = Y_dark(sigma(T)) - Y_dark(sigma(T0)), Y_dark from the committed dark rho-scan (04),
     interpolated in ln sigma. Outside the scanned sigma (1-100 S/m) a point is "outside the run scan", never a pass.
H_TC the photocarrier response scaled by g = mu_sum(T) tau(T): |dY_k| / |dY_3| = (P_k / P_3) g(T_k) / g(T_3); tau activated or a power law.
T(t) a lumped chip on the recorded timeline (power steps at each file's save time), R_th x C scanned; absorbed power at the prior median and 97.5th
     percentile; laser law amps (I_th 1.5, 2, 2.5 A) or tenth. D1: the handle's shift at the dark sweep after the series, from the remaining rise.
mu(T): EPW (02) or Jacoboni's measured exponents applied to EPW's 300 K values (held out: Jacoboni 1977; reported ranges e -2.3..-2.6, h -2.2..-2.8).
"""
import json
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

import common as c
import materials_si as ms
import thermal_closed as tc

KB = ms.KB
T0 = 296.0
R_GRID = np.logspace(np.log10(0.3), np.log10(3000.0), 61)        # K/W
C_GRID = np.array([0.05, 0.10, 0.17, 0.30])                       # J/K
EA = np.round(np.arange(0.0, 1.2001, 0.01), 3)                    # eV
POW = np.round(np.arange(0.0, 12.001, 0.1), 2)
F_KA = np.logspace(-2, 2, 41)                                     # R_th(Ka) / R_th(X)
LAWS = {"amps_Ith1.5": 1.5, "amps_Ith2.0": 2.0, "amps_Ith2.5": 2.5, "tenth": None}
JACOBONI = {"h": -2.20, "e": -2.42}
FREQ = {"X": "10", "Ka": "31.5"}
K_TOL, U_REL = 2.0, 0.05
ITH_GRID = np.round(np.arange(2.0, 2.901, 0.1), 1)               # 07b
ITH_FINE = np.round(np.arange(1.95, 2.951, 0.05), 2)              # 07c, 10


def power(k, ith):
    """Laser power (W) at setting k: the amps law with threshold ith (A; arrays broadcast), or the tenth law when ith is None."""
    return c.laser_power(np.asarray(k, float), ith is not None, ith)


def tau_law(fam, x):
    """Lifetime relative to T0 at a rise T: activated by x eV, or (T/T0)^x. x may be an array."""
    return (lambda T: np.exp(x / KB * (1 / T0 - 1 / (T0 + T)))) if fam == "Ea" else (lambda T: ((T0 + T) / T0) ** x)


def tol(u, y):
    """The pass band of every comparison here: K_TOL hypot(u, U_REL y)."""
    return K_TOL * np.hypot(u, U_REL * y)


def tc_ratios(P, g, ks=range(4, 8)):
    """H_TC: |dY_k| / |dY_3| = (P_k / P_3) g_k / g_3, from powers and g = mu_sum tau at each setting's rise."""
    return {k: P[k] / P[3] * g[k] / g[3] for k in ks}


def handle_shift(ysig, mt, day, T, sig, right=np.nan):
    """The dark handle's admittance shift (S) at a rise T, through its own sigma(T). right=np.nan marks rises beyond 02's table; None clamps."""
    s = np.interp(T0 + T, mt["tt"], sig, right=right)
    return ysig[day](s) - ysig[day](np.full_like(s, np.interp(T0, mt["tt"], sig)))


def temps(tl, R, C, ith, fac):
    """Rise above the chuck at each entry of a timeline; R, C broadcast. The setting of entry i holds from entry i-1's save to its own."""
    tau = R * C
    T = [np.zeros(np.broadcast(R, C).shape)]
    for i in range(1, len(tl)):
        rise = fac * power(tl[i][1], ith) * R
        T.append(rise + (T[-1] - rise) * np.exp(-(tl[i][2] - tl[i - 1][2]) / tau))
    return np.array(T)


def materials():
    """mu_sum(T) and sigma_rho(T) for both mu sources, on 02's grid extended as a power law to 1700 K (flagged in the file)."""
    mat = json.load(open(c.PREREG / "02_materials.json"))
    tt = np.array(mat["T_K"])
    mu = {"h": np.array(mat["mu_h_ibte"]), "e": np.array(mat["mu_e_ibte"])}
    i300 = int(np.argmin(abs(tt - 300.0)))
    src = {"EPW": mu, "Jacoboni": {x: mu[x][i300] * (tt / 300.0) ** JACOBONI[x] for x in "he"}}
    ni = np.array(mat["n_i_cm3"])
    out = {}
    for name, m in src.items():
        sig = {rho: ms.sigma_table(tt, m["h"], m["e"], ni, rho)[1] for rho in (1, 3, 10, 30, 100)}
        slope = {x: np.polyfit(np.log(tt[-10:]), np.log(m[x][-10:]), 1)[0] for x in "he"}
        def mu_sum(T, m=m, slope=slope):
            T = np.asarray(T, float)
            v = sum(np.where(T <= tt[-1], np.interp(T, tt, m[x]), m[x][-1] * (T / tt[-1]) ** slope[x]) for x in "he")
            return v / sum(np.interp(T0, tt, m[x]) for x in "he")
        out[name] = {"mu_sum": mu_sum, "sigma": sig, "tt": tt, "exponents_300_500": {x: float(np.polyfit(np.log(tt[(tt >= 300) & (tt <= 500)]),
                                                                                                        np.log(m[x][(tt >= 300) & (tt <= 500)]), 1)[0]) for x in "he"}}
    return out


def y_of_sigma():
    """Complex dark admittance (S) vs handle sigma at the X and Ka centre frequencies, from the committed dark rho-scan; PCHIP in ln sigma."""
    split = json.load(open(c.PREREG / "04_fullwave.json"))["split"]
    rhos = np.array([100, 30, 10, 3, 1])
    ls = np.log(100.0 / rhos)
    out = {}
    for day, f in FREQ.items():
        y = np.array([complex(*split[f"ps_dark_r{r}"][f]["Y_mS"]) * 1e-3 for r in rhos])
        re, im = PchipInterpolator(ls, y.real), PchipInterpolator(ls, y.imag)
        def y(s, re=re, im=im, lo=ls[0], hi=ls[-1]):
            ln = np.log(s)
            return np.where((ln >= lo) & (ln <= hi), re(ln) + 1j * im(ln), np.nan)
        out[day] = y
    return out


def measured():
    m = dict(np.load(c.OUT / "measured_y.npz"))
    mag = lambda a: abs(complex(a[0], a[1]))
    out = {}
    for day in ("X", "Ka"):
        t = {k: m[f"{day}_table_{k}"] for k in range(3, 8)}
        r = {k: mag(t[k]) / mag(t[3]) for k in t}
        u = {k: r[k] * np.hypot(t[k][2] / mag(t[k]), t[3][2] / mag(t[3])) for k in t}
        out[day] = {"dY": {k: complex(t[k][0], t[k][1]) for k in t}, "u": {k: t[k][2] for k in t}, "ratio": r, "u_ratio": u}
    out["X"]["after"], out["X"]["u_after"] = complex(*m["X_table_after"][:2]), m["X_table_after"][2]
    return out


def ok_ratio(pred, meas, ks):
    return np.all([np.abs(pred[k] - meas["ratio"][k]) <= tol(meas["u_ratio"][k], meas["ratio"][k]) for k in ks], axis=0)


def ok_complex(pred, d, u):
    return np.abs(pred - d) <= tol(u, abs(d))


def setup():
    mats, ysig, meas = materials(), y_of_sigma(), measured()
    m = tc.chip_model()
    fac_all = m["s"]["train_transmission"] * m["afrac"] * m["frac"]
    facs = {"median": float(np.median(fac_all)), "q97.5": float(np.percentile(fac_all, 97.5))}
    chip_area = c.device()["stack"]["chip_m"] ** 2 * np.array([np.min(m["s"]["chip_mm"]), np.max(m["s"]["chip_mm"])]) ** 2
    tl = {d: c.timeline(d) for d in ("X", "Ka")}
    idx = {d: {name: i for i, (name, _, _) in enumerate(tl[d])} for d in tl}
    lit = {d: {k: idx[d][c.raw_day(d)["lit"].format(k=k)] for k in range(3, 8)} for d in tl}
    return dict(mats=mats, ysig=ysig, meas=meas, facs=facs, fac_all=fac_all, chip_area=chip_area, tl=tl, idx=idx, lit=lit, i_after=idx["X"][c.raw_day("X")["after"]])


def g_laws():
    return [("Ea", ea, tau_law("Ea", ea)) for ea in EA] + [("pow", a, tau_law("pow", a)) for a in POW]


def evaluate(e, mt, ith, fac):
    """Pass masks of both hypotheses for one mu source, laser threshold (None = tenth law) and absorbed-power factor."""
    tl, lit, meas, ysig = e["tl"], e["lit"], e["meas"], e["ysig"]
    Tx = temps(tl["X"], R_GRID[:, None], C_GRID[None, :], ith, fac)                                   # (t, R, C)
    Tk = temps(tl["Ka"], (R_GRID[:, None] * F_KA[None, :])[:, None, :], C_GRID[None, :, None], ith, fac)   # (t, R, C, f)
    P = {k: power(k, ith) for k in range(3, 8)}
    d1 = {}
    for rho, sig in mt["sigma"].items():
        pred = handle_shift(ysig, mt, "X", Tx[e["i_after"]], sig)
        d1[rho] = ok_complex(np.where(np.isnan(pred), np.inf, pred), meas["X"]["after"], meas["X"]["u_after"])
    laws = g_laws()
    hx = np.zeros((len(laws), R_GRID.size, C_GRID.size), bool)
    hx57, hka = np.zeros_like(hx), np.zeros_like(hx)
    mux = {k: mt["mu_sum"](T0 + Tx[lit["X"][k]]) for k in range(3, 8)}      # mobility does not depend on the lifetime law
    muk = {k: mt["mu_sum"](T0 + Tk[lit["Ka"][k]]) for k in range(3, 8)}
    for j, (_, _, tau) in enumerate(laws):
        gx = {k: mux[k] * tau(Tx[lit["X"][k]]) for k in range(3, 8)}
        gk = {k: muk[k] * tau(Tk[lit["Ka"][k]]) for k in range(3, 8)}
        rx = tc_ratios(P, gx)
        hx[j], hx57[j] = ok_ratio(rx, meas["X"], range(4, 8)), ok_ratio(rx, meas["X"], (5, 7))
        hka[j] = np.any(ok_ratio(tc_ratios(P, gk), meas["Ka"], range(4, 8)), axis=-1)
    ht = {}
    for rho, sig in mt["sigma"].items():
        dy = lambda day, T: np.nan_to_num(handle_shift(ysig, mt, day, T, sig), nan=np.inf)
        okx = {k: ok_complex(dy("X", Tx[lit["X"][k]]), meas["X"]["dY"][k], meas["X"]["u"][k]) for k in range(3, 8)}
        ht[rho] = dict(px=np.all(list(okx.values()), axis=0), p57=okx[5] & okx[7],
                       pka=np.any(np.all([ok_complex(dy("Ka", Tk[lit["Ka"][k]]), meas["Ka"]["dY"][k], meas["Ka"]["u"][k]) for k in range(3, 8)], axis=0), axis=-1),
                       outside=np.isinf(dy("X", Tx[lit["X"][7]])))
    return dict(Tx=Tx, d1=d1, d1_any=np.any(list(d1.values()), axis=0), hx=hx, hx57=hx57, hka=hka, ht=ht, laws=laws,
                no_heat_ka=bool(ok_ratio({k: P[k] / P[3] for k in range(4, 8)}, meas["Ka"], range(4, 8))))


def main():
    c.require_private_repo()
    e = setup()
    summary, maps = {}, {}
    for mu_name, mt in e["mats"].items():
        for law, ith in LAWS.items():
            for lvl, fac in e["facs"].items():
                key = f"{mu_name}|{law}|{lvl}"
                v = evaluate(e, mt, ith, fac)
                hx, d1a, dT7 = v["hx"], v["d1_any"][None], v["Tx"][e["lit"]["X"][7]]
                masks = [hx, v["hx57"], hx & d1a, hx & d1a & v["hka"]]
                maps[f"H_TC|{key}"] = np.stack(masks)
                summary[f"H_TC|{key}"] = describe(*masks, dT7, v["laws"], e["chip_area"], v["no_heat_ka"])
                for rho, h in v["ht"].items():
                    masks = [h["px"][None], h["p57"][None], (h["px"] & v["d1"][rho])[None], (h["px"] & v["d1"][rho] & h["pka"])[None]]
                    maps[f"H_T|{key}|rho{rho}"] = np.stack(masks)[:, 0]
                    summary[f"H_T|{key}|rho{rho}"] = describe(*masks, dT7, None, e["chip_area"], None,
                                                              outside_scan_from_R=float(R_GRID[np.argmax(h["outside"].any(axis=1))]) if h["outside"].any() else None)
    res = {"what": "flip map: where heating reproduces the X day, with D1 and Ka (00d A); R_th of a lumped chip, C in J/K, dT7 = rise at the 7_0V save",
           "grids": {"R_K_W": [float(R_GRID[0]), float(R_GRID[-1]), int(R_GRID.size)], "C_J_K": C_GRID.tolist(), "E_a_eV": [0.0, 1.2, 0.01], "a": [0.0, 12.0, 0.1],
                     "f_Ka": [0.01, 100.0, int(F_KA.size)]},
           "absorbed_factor": e["facs"], "chip_area_m2": e["chip_area"].tolist(), "T0_K": T0,
           "mu_exponents_300_500K": {k: v["exponents_300_500"] for k, v in e["mats"].items()},
           "notes": ["mu beyond 700 K and tau laws are extrapolations; R_th is held constant with temperature (kappa(T) and gas conductance changes not included).",
                     "Ka ratios use Ka's own R_th = f R_th(X), f 0.01-100; 'no_heating_matches_Ka' is the same test with no rise at all."],
           "summary": summary}
    np.savez_compressed(c.OUT / "flip_map.npz", R=R_GRID, C=C_GRID, Ea=EA, a=POW, **{k: v for k, v in maps.items()})
    c.register("07_flip_map", res, code=[__file__, c.HERE / "common.py", c.HERE / "thermal_closed.py", c.HERE / "materials_si.py"])
    print("hypotheses with any X pass:", sum(v["X"]["any"] for v in summary.values()), "of", len(summary),
          "| with D1:", sum(v["X_and_D1"]["any"] for v in summary.values()))


CAPS = {"Ea<=0.40": ("Ea", 0.40), "Ea<=0.55": ("Ea", 0.55), "Ea_any": ("Ea", np.inf), "pow<=4": ("pow", 4.0), "pow_any": ("pow", np.inf)}


def threshold():
    """07b (00e): the map along the laser threshold, with physical caps on the lifetime law."""
    c.require_private_repo()
    e = setup()
    table = {}
    for mu_name, mt in e["mats"].items():
        for ith in ITH_GRID:
            for lvl, fac in e["facs"].items():
                v = evaluate(e, mt, float(ith), fac)
                dT7 = v["Tx"][e["lit"]["X"][7]]
                row = {"P7_over_P3": power(7, ith) / power(3, ith), "no_heating_matches_Ka": v["no_heat_ka"]}
                for cap, (fam, lim) in CAPS.items():
                    sel = np.array([f == fam and x <= lim + 1e-9 for f, x, _ in v["laws"]])
                    mask = v["hx"] & v["d1_any"][None] & sel[:, None, None]
                    if mask.any():
                        j, i, k = np.nonzero(mask)
                        n = int(np.argmin(R_GRID[i]))
                        at = i == i[n]
                        row[cap] = {"R_th_min_K_W": float(R_GRID[i[n]]), "dT7_K_at_R_min": float(dT7[i[at], k[at]].min()), "dT7_K_range": [float(dT7[i, k].min()), float(dT7[i, k].max())],
                                    "laws": [float(min(v["laws"][jj][1] for jj in j)), float(max(v["laws"][jj][1] for jj in j))],
                                    "with_Ka": bool((mask & v["hka"]).any())}
                    else:
                        row[cap] = None
                ht = {str(rho): bool((h["px"] & v["d1"][rho]).any()) for rho, h in v["ht"].items()}
                row["H_T_reproduces_X_with_D1_by_rho"] = ht
                table[f"{mu_name}|Ith{ith:.1f}|{lvl}"] = row
    d = c.device()["laser"]
    res = {"what": f"07b: flip map along the laser threshold (amps law, {d['p_max_W']:g} W at {d['i_max_A']:g} A), H_TC with lifetime caps and H_T per rho, X with D1; Ka reported",
           "caps": list(CAPS), "absorbed_factor": e["facs"], "table": table}
    c.register("07b_flip_threshold", res, code=[__file__, c.HERE / "common.py", c.HERE / "thermal_closed.py", c.HERE / "materials_si.py"])
    for k, row in table.items():
        print(k, f"P7/P3 {row['P7_over_P3']:.1f}", {cap: (round(r["R_th_min_K_W"], 1), round(r["dT7_K_range"][0], 1)) if r else None for cap, r in row.items() if cap in CAPS})


def describe(px, p57, pd1, pall, dT7, laws, chip_area, no_heat_ka, outside_scan_from_R=None):
    """The passing region of one hypothesis: smallest R_th, its dT7, tau_th and equivalent air gap; the lifetime laws it needs."""
    def one(mask):
        if not mask.any():
            return {"any": False}
        j, i, k = np.nonzero(mask)
        r = R_GRID[i]
        out = {"any": True, "R_th_min_K_W": float(r.min()), "dT7_K_range": [float(dT7[i, k].min()), float(dT7[i, k].max())],
               "tau_th_s_range": [float((r * C_GRID[k]).min()), float((r * C_GRID[k]).max())],
               "air_gap_um_at_R_min": [float(r.min() * 0.026 * a * 1e6) for a in chip_area]}
        if laws is not None:
            fams = {f: sorted({laws[jj][1] for jj in j if laws[jj][0] == f}) for f in ("Ea", "pow")}
            out["tau_laws"] = {f: [min(v), max(v)] if v else None for f, v in fams.items()}
        return out
    res = {"X": one(px), "X_00_rule_s5_s7": one(p57), "X_and_D1": one(pd1), "X_D1_Ka": one(pall)}
    if no_heat_ka is not None:
        res["no_heating_matches_Ka"] = no_heat_ka
    if outside_scan_from_R is not None:
        res["dT7_outside_run_scan_from_R_K_W"] = outside_scan_from_R
    return res


def linearity():
    """07c (00f): is each day's |dY| proportional to P(setting) under a given threshold? One scale per day, no input fitted. Joint with 07b and 08."""
    c.require_private_repo()
    meas = measured()
    laws = {f"amps_Ith{x:.2f}": float(x) for x in ITH_FINE} | {"tenth": None}
    out = {}
    for name, ith in laws.items():
        row = {}
        for day in ("X", "Ka"):
            for ks in ((4, 5, 6, 7), (3, 4, 5, 6, 7)):
                y = np.array([abs(meas[day]["dY"][k]) for k in ks])
                u = np.array([meas[day]["u"][k] for k in ks])
                P = np.array([power(k, ith) for k in ks])
                a = np.sum(P * y / u**2) / np.sum(P**2 / u**2)
                res = a * P - y
                row[f"{day}_s{ks[0]}-7"] = {"linear": bool(np.all(np.abs(res) <= tol(u, y))), "worst_over_tol": float(np.max(np.abs(res) / tol(u, y))),
                                            "scale_uS_per_W": float(a * 1e6)}
        out[name] = row
    fb = json.load(open(c.PREREG / "07b_flip_threshold.json"))["table"]
    q99 = json.load(open(c.PREREG / "08_contact.json"))["R_th_K_W"]["q99"]
    viable = {}
    for key, row in fb.items():
        mu, ith, lvl = key.split("|")
        r = row["Ea<=0.40"]
        if r is not None and r["R_th_min_K_W"] <= q99:
            viable.setdefault(ith[3:], []).append(f"{mu}|{lvl}|R_th>={r['R_th_min_K_W']:.1f}")
    ka_linear = sorted(n[8:] for n, r in out.items() if n.startswith("amps") and r["Ka_s4-7"]["linear"])
    joint = sorted(set(f"{float(x):.1f}" for x in ka_linear) & set(viable))
    res = {"what": "07c: linearity of |dY| in P(setting) per day and threshold (one scale per day), joined with 07b and 08 (00f)",
           "R_th_q99_08": q99, "table": out, "Ka_linear_s4-7_thresholds": ka_linear, "X_heating_viable_thresholds_07b_08": viable,
           "joint_thresholds": joint, "tenth_law_Ka_linear": out["tenth"]["Ka_s4-7"]["linear"]}
    c.register("07c_one_laser", res, code=[__file__, c.HERE / "common.py"])
    for n, r in out.items():
        print(n, {k: (v["linear"], round(v["worst_over_tol"], 2)) for k, v in r.items()})
    print("Ka linear (s4-7):", ka_linear); print("X heating viable:", viable); print("joint:", joint)


if __name__ == "__main__":
    {"map": main, "threshold": threshold, "linearity": linearity}[sys.argv[1] if len(sys.argv) > 1 else "map"]()

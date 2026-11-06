"""The comparison, run only after the predictions are committed. ASME V&V 20: E = S - D against u_val^2 = u_num^2 + u_input^2 + u_D^2. Failures stay red.

Writes OUT/RESULTS.md (private: it holds measured numbers). Refuses to run if any prediction file is uncommitted or dirty.
"""
import json

import numpy as np

import common as c

K_RED = 2.0     # flagged when |E| > 2 u_val


class Table:
    def __init__(self):
        self.rows, self.gates = [], []

    def vv(self, name, s, d, u_num, u_input, u_d, blind, sha):
        e = s - d
        u_val = float(np.sqrt(u_num**2 + u_input**2 + u_d**2))
        self.rows.append(dict(name=name, S=s, D=d, u_num=u_num, u_input=u_input, u_D=u_d, E=e, u_val=u_val, ratio=abs(e) / u_val if u_val else np.inf, ok=abs(e) <= K_RED * u_val, blind=blind, sha=sha))

    def gate(self, name, value, limit, note=""):
        self.gates.append(dict(name=name, value=value, limit=limit, ok=bool(abs(value) <= limit), note=note))


def log_spread(fn, rhos):
    """Median and half the 16-84 range of fn(rho) under the prior: log-uniform handle resistivity on [1, 100], by interpolation in ln(rho) between the runs."""
    rng = np.random.default_rng(c.SEED)
    q = np.exp(rng.uniform(0, np.log(100), 20000))
    v = np.interp(np.log(q), np.log(rhos), [fn(r) for r in rhos])
    return float(np.median(v)), float(0.5 * (np.percentile(v, 84) - np.percentile(v, 16)))


def main():
    c.require_private_repo()
    sha = {n: c.require_registered(n) for n in ("01_thermal_closed_form", "02_materials", "03_fingerprints", "04_fullwave", "05_fem", "06_thermal_elmer")}
    m = dict(np.load(c.OUT / "measured_y.npz"))
    u_rel = c.device()["u_num_rel"]
    fp = json.load(open(c.PREREG / "03_fingerprints.json"))
    fw04 = json.load(open(c.PREREG / "04_fullwave.json"))
    fem = json.load(open(c.PREREG / "05_fem.json"))
    th = json.load(open(c.PREREG / "06_thermal_elmer.json"))
    T = Table()

    # ------------------------------------------------------------ dark admittance, both days
    band = {"X": (10.0, 2 * np.pi * 10e9), "Ka": (31.5, 2 * np.pi * 31.5e9)}
    split = fw04["split"]                         # S comes from the committed file only
    rhos = sorted(int(t.split("_r")[1]) for t in split if t.startswith("ps_dark_r") and not t.endswith("_m25"))
    un = fw04["u_num"]
    for day, (f, w) in band.items():
        ymeas = m[f"{day}_dark_c"]
        ucal = np.abs(np.mean(m[f"{day}_dark_c"] - m[f"{day}_dark_b"]))
        G = lambda r: split[f"ps_dark_r{r}"][f"{f:g}"]["Y_mS"][0] * 1e-3
        B = lambda r: split[f"ps_dark_r{r}"][f"{f:g}"]["Y_mS"][1] * 1e-3
        gm, gs = log_spread(G, rhos)
        bm, bs = log_spread(B, rhos)
        T.vv(f"dark G, {day}, mS", gm * 1e3, np.mean(ymeas).real * 1e3, un[f"G_{day}"] * 1e3, gs * 1e3, ucal * 1e3, False, sha["04_fullwave"])
        T.vv(f"dark C, {day}, fF", bm / w * 1e15, np.mean(ymeas).imag / w * 1e15, un[f"B_{day}"] / w * 1e15, bs / w * 1e15, ucal / w * 1e15, False, sha["04_fullwave"])

    # ------------------------------------------------------------ direct heating against the measured dY
    for day in ("X", "Ka"):
        for k in (3, 5, 7):
            d = m[f"{day}_table_{k}"]
            h = fp[f"H_T_dY_uS_{day}"][str(k)]
            for part, val, ud in (("dG", d[0], d[2]), ("dB", d[1], d[2])):
                key = "re" if part == "dG" else "im"
                s = h[key]["q50"]
                ui = 0.5 * (h[key]["q84"] - h[key]["q16"])
                T.vv(f"H_T {part}, {day} s{k}, uS", s, val * 1e6, abs(s) * u_rel["H_T"], ui, ud * 1e6, False, sha["03_fingerprints"])

    # ------------------------------------------------------------ carriers at nominal inputs: no tuned spot, lifetime or gain
    for k in (3, 5, 7):
        for day, (f, w) in band.items():
            lit, dark = split[f"ps_lit_r10_w10_s{k}"][f"{f:g}"]["Y_mS"], split["ps_dark_r10"][f"{f:g}"]["Y_mS"]
            dy = (complex(*lit) - complex(*dark)) * 1e-3
            d = m[f"{day}_table_{k}"]
            ui = fw04["u_input_lit_s7_rel"] * abs(dy)
            un_lit = abs(dy) * u_rel["lit_minus_dark"]
            T.vv(f"H_C dG nominal, {day} s{k}, uS", dy.real * 1e6, d[0] * 1e6, un_lit * 1e6, ui * 1e6, d[2] * 1e6, False, sha["04_fullwave"])
            T.vv(f"H_C dB nominal, {day} s{k}, uS", dy.imag * 1e6, d[1] * 1e6, un_lit * 1e6, ui * 1e6, d[2] * 1e6, False, sha["04_fullwave"])

    # ------------------------------------------------------------ the fingerprints
    x = {k: m[f"X_table_{k}"] for k in (3, 4, 5, 6, 7)}
    ka = {k: m[f"Ka_table_{k}"] for k in (3, 4, 5, 6, 7)}
    mag = lambda a: abs(complex(a[0], a[1]))
    r_x = {k: mag(x[k]) / mag(x[3]) for k in x}
    r_ka = {k: mag(ka[k]) / mag(ka[3]) for k in ka}
    u_x = {k: r_x[k] * np.hypot(x[k][2] / mag(x[k]), x[3][2] / mag(x[3])) for k in x}
    u_ka = {k: r_ka[k] * np.hypot(ka[k][2] / mag(ka[k]), ka[3][2] / mag(ka[3])) for k in ka}

    s = dict(np.load(c.OUT / "heating_samples.npz"))
    if c.arrays_hash(s) != fp["samples_sha256"]:
        raise SystemExit("heating_samples.npz is not the sample set registered in 03_fingerprints: run the fingerprints stage again")
    # sensitivities from the committed file, interpolated in ln(rho) across the sampled handle resistivity
    sens = fp["dark_sensitivity"]
    rr = sorted(int(r) for r in sens)
    lq, lr = np.log(s["rho"]), np.log(rr)
    dyc = np.interp(lq, lr, [sens[str(r)]["10"][0] for r in rr]) + 1j * np.interp(lq, lr, [sens[str(r)]["10"][1] for r in rr])
    # D1 and D2 bounds on each sample, from the X-day series
    after = complex(*m["X_table_after"][:2])
    u_after = m["X_table_after"][2]
    d1 = np.interp(lq, lr, [abs(complex(*sens[str(r)]["10"])) for r in rr]) * np.abs(s["dlns"] * s["dT_after"]) <= abs(after) + 3 * u_after
    frac = lambda mask: float(np.mean(mask[d1])) if d1.any() else float("nan")
    rel_tol = lambda k, u, r: K_RED * np.hypot(u[k], u_rel["ratio"] * r[k])
    ok5 = np.abs(s["ratio_5"] - r_x[5]) <= rel_tol(5, u_x, r_x)
    ok7 = np.abs(s["ratio_7"] - r_x[7]) <= rel_tol(7, u_x, r_x)
    decide = {"H_TC": {"f": frac(ok5 & ok7), "f_reaching_s7_only": frac(ok7), "excluded_by_D1": float(1 - np.mean(d1))}}
    # H_T, sample by sample: the handle's own sigma(T), complex first-order response, against the measured dY at s5 and s7
    ok_t = np.ones(lq.size, bool)
    for k in (5, 7):
        ok_t &= np.abs(dyc * s["dlns"] * s[f"dT_{k}"] - complex(x[k][0], x[k][1])) <= K_RED * np.hypot(x[k][2], u_rel["H_T"] * mag(x[k]))
    g7 = (dyc * s["dlns"] * s["dT_7"]).real * 1e6
    decide["H_T"] = {"f": frac(ok_t), "largest_dG_uS_at_s7_in_any_sample": float(g7.max()), "most_negative_dG_uS_at_s7": float(g7.min())}
    # the same samples on the Ka day
    ok_ka = np.abs(s["ratio_7"] - r_ka[7]) <= rel_tol(7, u_ka, r_ka)
    decide["H_TC_explains_X_and_Ka_s7_with_one_dT"] = frac(ok_ka & ok7)
    f_all = max(decide["H_TC"]["f"], decide["H_T"]["f"])
    verdict = "no" if f_all < 0.05 else ("yes" if f_all > 0.5 else "not decidable")

    # ------------------------------------------------------------ time and frequency fingerprints
    extra = json.load(open(c.OUT / "measured_extra.json"))
    # D1: dark after minus dark before
    q1 = fp["D1_dark_after_uS_at_273s"]
    T.vv("D1 dG, dark after - before, uS", q1["re"]["q50"], after.real * 1e6, 0.0, 0.5 * (q1["re"]["q84"] - q1["re"]["q16"]), u_after * 1e6, False, sha["03_fingerprints"])
    T.vv("D1 dB, dark after - before, uS", q1["im"]["q50"], after.imag * 1e6, 0.0, 0.5 * (q1["im"]["q84"] - q1["im"]["q16"]), u_after * 1e6, False, sha["03_fingerprints"])
    # D2: the top setting repeated later. Thermal: the remaining rise. The repeat's DC bias is in the data and cannot be separated
    u_rep = float(np.hypot(m["X_table_bias"][2], m["X_table_7"][2]))
    q2 = fp["D2_response_change_fraction_if_H_TC"]
    T.vv("D2 |dY(top, repeat)|/|dY(top)| - 1, heating-coupled", q2["q50"], abs(complex(*m["X_table_bias"][:2])) / abs(complex(*m["X_table_7"][:2])) - 1, 0.0, 0.5 * (q2["q84"] - q2["q16"]), u_rep / mag(m["X_table_7"]), False, sha["03_fingerprints"])
    # F3: the in-band shape of the response, 11.75 over 8.25 GHz, s5 and s7
    f_ = m["X_f"]
    edge = lambda y, f0: complex(np.mean(y[np.abs(f_ - f0) < 50e6]))
    f3 = {}
    for k in (3, 5, 7):
        dy = m[f"X_{k}_c"] - m["X_dark_c"]
        r = edge(dy, 11.7e9) / edge(dy, 8.3e9)
        f3[k] = {"abs": abs(r), "phase_deg": float(np.degrees(np.angle(r)))}
    # F4: the Ka day under a heating-coupled steepening, ratio to s3
    for k in (5, 7):
        h = fp["F4_Ka_ratio_to_s3_if_H_TC"][str(k)]
        T.vv(f"F4 Ka |dY(s{k})|/|dY(s3)|, heating-coupled", h["q50"], r_ka[k], u_rel["ratio"] * h["q50"], 0.5 * (h["q84"] - h["q16"]), u_ka[k], True, sha["03_fingerprints"])
        T.vv(f"H_TC X |dY(s{k})|/|dY(s3)|", fp["H_TC_ratio_to_s3"][str(k)]["q50"], r_x[k], u_rel["ratio"] * r_x[k], 0.5 * (fp["H_TC_ratio_to_s3"][str(k)]["q84"] - fp["H_TC_ratio_to_s3"][str(k)]["q16"]), u_x[k], False, sha["03_fingerprints"])

    # ------------------------------------------------------------ gates
    for g in fw04["gates"] + fem["gates"] + th["gates"]:
        T.gate(g["name"], g["value"], g["limit"], g.get("note", ""))
    mat = json.load(open(c.PREREG / "02_materials.json"))
    for x_ in "he":
        T.gate(f"Kramers-Kronig, EPW-weighted sigma(omega), {x_}", mat["kk_rms_rel"][x_], 0.01)
        for t, v in mat["control_fast_vs_unmodified_24cube"][x_].items():
            T.gate(f"fast build vs unmodified EPW, 24^3, {x_}, {t} K, IBTE", v["rel_diff_ibte"], 1e-3)

    write_results(T, fp, r_x, r_ka, u_x, u_ka, decide, verdict, m, sha, fw04, fem, th, f3, extra, mat)


def what_heating_needs(mat, r_x):
    """The temperature rise at setting 7 that the heating-coupled route needs to reach the measured s7/s3, for each lifetime law and laser law, and the s5/s3 it then gives."""
    tt, mu = np.array(mat["T_K"]), np.array(mat["mu_h_ibte"]) + np.array(mat["mu_e_ibte"])
    t0 = 296.0                                    # within the prior's chuck range; the ratios barely depend on it

    def g(dt, ea=None, a=None):
        temp = t0 + dt
        tau = np.exp(ea / 8.617333262e-5 * (1 / t0 - 1 / temp)) if ea is not None else (temp / t0) ** a
        return np.interp(temp, tt, mu) / np.interp(t0, tt, mu) * tau
    rows = []
    for law in ("amps", "tenth"):
        steps = {k: float(c.laser_power(np.array(float(k)), np.array(law == "amps"))) for k in (3, 5, 7)}      # the rise scales with the power of each setting
        steps = {k: v / steps[3] for k, v in steps.items()}
        want7, want5 = r_x[7] * steps[3] / steps[7], r_x[5] * steps[3] / steps[5]
        for name, kw in (("E_a 0.40 eV", {"ea": 0.40}), ("E_a 0.20 eV", {"ea": 0.20}), ("T^4", {"a": 4}), ("T^2", {"a": 2})):
            x = np.linspace(0.01, 400.0 / steps[7], 8000)
            ratio = g(steps[7] * x, **kw) / g(steps[3] * x, **kw)
            hit = np.where(ratio >= want7)[0]
            if len(hit):
                i = hit[0]
                rows.append((law, name, steps[7] * x[i], g(steps[5] * x[i], **kw) / g(steps[3] * x[i], **kw), want5))
            else:
                rows.append((law, name, None, None, want5))
    return rows


def write_results(T, fp, r_x, r_ka, u_x, u_ka, decide, verdict, m, sha, fw04, fem, th, f3, extra, mat):
    L = ["# Si meander dipole: power split and heating", "", f"Verdict on heating as the cause of the X-day turn-on: **{verdict}** under the decision rule fixed in prereg/00_PREREG.md.", "",
         "One-port S11 fixes only total Re(Y). The radiated/heat split below is a model result until a Wheeler cap, gain or thermal measurement exists.", "",
         "## Decision", "", "```", json.dumps(decide, indent=1, default=float), "```", "",
         "## Turn-on ratios to setting 3, measured", "", "| setting | X | Ka |", "|---|---|---|"]
    for k in r_x:
        L.append(f"| {k} | {r_x[k]:.2f} +- {u_x[k]:.2f} | {r_ka[k]:.2f} +- {u_ka[k]:.2f} |")
    s = np.load(c.OUT / "heating_samples.npz")
    L += ["", "## What heating would need", "", "The temperature rise at setting 7 that reaches the measured s7/s3 by way of tau(T) mu(T), with mu(T) from EPW:", "",
          "| laser law | lifetime law | dT at s7, K | s5/s3 steepening it gives | needed |", "|---|---|---|---|---|"]
    for law, name, dt, s5, want in what_heating_needs(mat, r_x):
        L.append(f"| {law} | {name} | {'none below 400 K' if dt is None else f'{dt:.0f}'} | {'-' if s5 is None else f'{s5:.2f}'} | {want:.2f} |")
    d7 = s["dT_7"]
    L += ["", f"Input prior for dT at s7: median {np.median(d7):.2f} K, 84 % {np.percentile(d7, 84):.2f} K, 97.5 % {np.percentile(d7, 97.5):.1f} K; P(dT > 25 K) = {(d7 > 25).mean():.4f}.", "",
          "Steepening per step, measured |dY| ratio over the amps-law power ratio, not registered in advance:", ""]
    pw = lambda k: float(c.laser_power(np.array(float(k)), np.array(True)))
    L += [f"- s{k} to s{k + 1}: {r_x[k + 1] / r_x[k] / (pw(k + 1) / pw(k)):.2f}" for k in (3, 4, 5, 6)]
    L += ["", "dT rises by equal steps here, so an activated tau(T) would give a constant or rising factor per step. The measured factor falls, and most of the steepening is at the lowest powers.", "",
          (c.PS / "reading_notes.md").read_text() if (c.PS / "reading_notes.md").exists() else ""]
    gaas = {n: {k: abs(complex(*rows[str(k)])) for k in (3, 5, 7)} for n, rows in extra["controls"].items()}
    L += ["", "Control devices on a carrier of much larger thermal resistance, same laser, same day, |dY| ratio to s3:", ""]
    L += [f"- {n}: s5/s3 {g[5] / g[3]:.2f}, s7/s3 {g[7] / g[3]:.2f}" for n, g in gaas.items()]
    L += ["", f"Ka ambient sweeps, after minus before the lit run, {extra['Ka_ambient_s_since_last_lit']:.0f} s after the last lit sweep: {extra['Ka_ambient_after_minus_before'][0] * 1e6:+.1f} {extra['Ka_ambient_after_minus_before'][1] * 1e6:+.1f}j uS", "",
          f"F3, in-band shape of dY, 11.7 over 8.3 GHz, measured: {json.dumps(f3)}; predicted: {json.dumps(fp['F3_ratio_11p75_over_8p25'])}", "",
          f"tau_th {fp['tau_th_s']}", "", "## Where the power goes (model result; not fixed by the data)", "",
          "| run | f GHz | Y mS | radiated | handle | membrane | oxide | pads | balance |", "|---|---|---|---|---|---|---|---|---|"]
    for t, fs in fw04["split"].items():
        for f in ("10", "31.5"):
            r = fs[f]
            L.append(f"| {t} | {f} | {r['Y_mS'][0]:.3f}{r['Y_mS'][1]:+.3f}j | {r['rad']:.2e} | {r['handle']:.4f} | {r['sheet']:.2e} | {r['box']:.1e} | {r['pads']:.1e} | {r['balance']:+.4f} |")
    q = fw04["q_along_x"]
    L += ["", f"Heat along the antenna, dark, 10 GHz: half of the handle's heat lies within {q['half_of_handle_heat_within_mm']:.3f} mm of the feed centre line, 90 % within {q['ninety_pct_within_mm']:.3f} mm.", "",
          "## Second method class", "", "```", json.dumps(fem["vs_openems"], indent=1), "```", f"Dark arms change Palace's Y by {fem['dark_arms_change_Y_by']:.2%}.", "",
          "## Materials", "", f"mu ~ T^n: {mat['power_law_exponents']}.", "", "| carrier | mu_IBTE(300 K) | m_c EPW | m_c Wannier | omega tau 10 GHz | omega tau 31.5 GHz |", "|---|---|---|---|---|---|"]
    for x_ in "he":
        d = mat["drude_epw_route"][x_]["300"]
        L.append(f"| {x_} | {mat['mobility_table'][x_]['300']['ibte']:.1f} | {d['m_cond']:.3f} | {mat['drude_wannier_route']['300']['m_cond_' + x_]:.3f} | {d['omega_tau_10GHz']:.4f} | {d['omega_tau_31p5GHz']:.4f} |")
    L += ["", "## Chip temperature, Elmer", ""]
    for n in c.device()["thermal_cases"]:
        L.append(f"- {n}: steady dT {th[n]['dT_mean_K']:.3g} K at setting 7 (closed form {th[n]['closed_form_dT_K']:.3g} K), 1/e cooling {th[n]['cooling_1_over_e_s']} s, P_abs {th[n]['P_abs_on_chip_W']:.3g} W")
    L += ["", "## ASME V&V 20", "", "| quantity | S | D | u_num | u_input | u_D | E | u_val | E/u_val | status | blind | prediction |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.rows:
        L.append(f"| {r['name']} | {r['S']:.4g} | {r['D']:.4g} | {r['u_num']:.2g} | {r['u_input']:.2g} | {r['u_D']:.2g} | {r['E']:+.3g} | {r['u_val']:.2g} | {r['ratio']:.2f} | {'pass' if r['ok'] else 'RED'} | {'yes' if r['blind'] else 'no'} | {r['sha']} |")
    L += ["", "## Identities and numerical gates", "", "| check | value | limit | status | note |", "|---|---|---|---|---|"]
    for g in T.gates:
        L.append(f"| {g['name']} | {g['value']:.3g} | {g['limit']:g} | {'pass' if g['ok'] else 'RED'} | {g['note']} |")
    (c.OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    n_red = sum(not r["ok"] for r in T.rows) + sum(not g["ok"] for g in T.gates)
    print(f"verdict: {verdict}; {len(T.rows)} comparisons, {len(T.gates)} gates, {n_red} red")
    print((c.OUT / "RESULTS.md").read_text()[:6000])


if __name__ == "__main__":
    main()

"""The deeper heating question, compared only after prereg/07-13 are committed (rules 00d-00g). Writes OUT/RESULTS_DEEP.md and OUT/verdict_map.png (private).

The verdict is a map over the unrecorded inputs: the laser threshold, the absorbed power, the lifetime's activation and the chip-to-chuck contact, with the part
the records exclude marked. Comparisons follow validate.py (ASME V&V 20, red when |E| > 2 u_val). The tables and numbers are computed here; the prose around
them (the records it rests on, their reading) is the private templates/RESULTS_DEEP.md.
"""
import json

import numpy as np

import common as c
import contact
import flip_map as fm
import validate as v

NAMES = ("07_flip_map", "07b_flip_threshold", "07c_one_laser", "08_contact", "09_time_fingerprints", "10_gaas_controls", "11_red_rows", "12_evidence",
         "13_protocol_predictions")
JACOBONI_RANGE = {"h": (-2.8, -2.2), "e": (-2.6, -2.3)}      # reported measured exponents (00d)


def main():
    c.require_private_repo()
    sha = {n: c.require_registered(n) for n in NAMES}
    P = {n: json.load(open(c.PREREG / f"{n}.json")) for n in NAMES}
    m = dict(np.load(c.OUT / "measured_y.npz"))
    T = v.Table()
    # dark C at Ka with the probe (11) against the measurement, as validate.py's dark rows
    w = 2 * np.pi * 31.5e9
    ymeas = m["Ka_dark_c"]
    ucal = np.abs(np.mean(m["Ka_dark_c"] - m["Ka_dark_b"]))
    spec = P["11_red_rows"]["C_Ka_fF_spec_rho_14_22"]
    T.vv("dark C, Ka, with the probe (spec handle), fF", float(np.mean(spec)), np.mean(ymeas).imag / w * 1e15, P["11_red_rows"]["u_num_probe_geometry_fF"],
         abs(spec[1] - spec[0]) / 2, ucal / w * 1e15, False, sha["11_red_rows"])
    # time fingerprints (09) against the X and Ka records
    after = complex(*m["X_table_after"][:2])
    extra = json.load(open(c.OUT / "measured_extra.json"))
    amb = complex(*extra["Ka_ambient_after_minus_before"])
    u_rep = float(np.hypot(m["X_table_bias"][2], m["X_table_7"][2]))
    d2 = abs(complex(*m["X_table_bias"][:2])) / abs(complex(*m["X_table_7"][:2])) - 1
    rho = f"rho{c.device()['deep']['handle']['rho_spec_ohm_cm'][1]:g}"     # the spec's high end, as in 09's keys
    for sc in ("nominal", "corner"):
        f = P["09_time_fingerprints"][sc]
        T.vv(f"D1 dG, dark after, {sc}, uS", f[f"D1_dark_after_uS_{rho}"][0], after.real * 1e6, 0.0, 0.0, m["X_table_after"][2] * 1e6, False, sha["09_time_fingerprints"])
        T.vv(f"D2 warming part, 7_10V vs 7_0V, {sc}", f["warming_change_7_10V_vs_7_0V"], d2, 0.0, 0.0, u_rep / abs(complex(*m["X_table_7"][:2])), False, sha["09_time_fingerprints"])
        sa = complex(*f[f"Ka_ambient_after_minus_before_uS_{rho}"])
        T.vv(f"Ka ambient after - before, dG from heating, {sc}, uS", sa.real, amb.real * 1e6, 0.0, 0.0, m["Ka_table_3"][2] * 1e6, False, sha["09_time_fingerprints"])
    # mobility exponents, EPW against Jacoboni's measured values (held out)
    ex = P["07_flip_map"]["mu_exponents_300_500K"]
    for x, name in (("h", "holes"), ("e", "electrons")):
        lo, hi = JACOBONI_RANGE[x]
        T.vv(f"mu(T) exponent, {name}, EPW vs Jacoboni", ex["EPW"][x], ex["Jacoboni"][x], 0.05, 0.0, (hi - lo) / 2, False, sha["07_flip_map"])

    # the map: what heating needs along the threshold, against the contact and the one-laser test
    fb, con, oc = P["07b_flip_threshold"]["table"], P["08_contact"], P["07c_one_laser"]
    r_contact = contact.r_th(contact.sample())["R"]                  # the 08 distribution (same seed)
    ka_lin = [float(x) for x in oc["Ka_linear_s4-7_thresholds"]]
    gaas = P["10_gaas_controls"]["post_hoc_linearity_thresholds"]
    rows, plot = [], {"median": {}, "q97.5": {}}
    for ith in fm.ITH_GRID:
        cells = []
        for lvl in ("q97.5", "median"):
            r = fb[f"EPW|Ith{ith:.1f}|{lvl}"]["Ea<=0.40"]
            if r is None:
                cells.append("none")
                continue
            plot[lvl][float(ith)] = r["R_th_min_K_W"]
            p = float(np.mean(r_contact >= r["R_th_min_K_W"]))
            cells.append(f"R_th >= {r['R_th_min_K_W']:.1f} K/W, dT7 >= {r['dT7_K_at_R_min']:.0f} K (contact allows: {p:.2f})")
        rows.append(f"| {ith:.1f} | {cells[0]} | {cells[1]} | {'yes' if ith <= max(ka_lin) + 1e-9 else 'no'} |")
    ev = P["12_evidence"]["models"]
    open_region = oc["joint_thresholds"]
    viable = [float(k) for k in oc["X_heating_viable_thresholds_07b_08"]]
    epw, jac = ev["H_TC|EPW"], ev["H_TC|Jacoboni"]
    text = c.template("RESULTS_DEEP.md").format(
        viable_min=min(viable), viable_max=max(viable), contact_q99=con["R_th_K_W"]["q99"], ka_max=max(ka_lin),
        open_region=", ".join(open_region) if open_region else "empty",
        p_d1_epw=epw["P_prior_heating_reproduces_X_with_D1"], p_d1_jac=jac["P_prior_heating_reproduces_X_with_D1"],
        p_ka_epw=epw["P_prior_and_Ka_linear_07c"], p_ka_jac=jac["P_prior_and_Ka_linear_07c"],
        threshold_rows="\n".join(rows), gaas=json.dumps(gaas), comparisons="\n".join(T.markdown()),
        evidence=json.dumps(ev, indent=1), evidence_note=P["12_evidence"]["notes"][0])
    review = c.OUT / "REVIEW_DEEP.md"
    if review.exists():
        text += "\n\n" + review.read_text()
    (c.OUT / "RESULTS_DEEP.md").write_text(text + "\n")
    figure(plot, con, ka_lin)
    n_red = sum(not r["ok"] for r in T.rows)
    print(f"{len(T.rows)} comparisons, {n_red} red; open region: {open_region}")
    print("\n".join(text.splitlines()[:20]))


def figure(plot, con, ka_lin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=surf)
    ax.set_facecolor(surf)
    q = con["R_th_K_W"]
    ax.axhspan(q["q2.5"], q["q99"], color="#e6e5e1", lw=0, zorder=0)
    ax.axhspan(q["q99"], 1000, facecolor="none", hatch="///", edgecolor="#c9c8c3", lw=0, zorder=0)
    ax.text(1.97, q["q2.5"] * 1.15, "chip resting on the chuck (08)", color=ink2, fontsize=8, va="bottom")
    ax.text(2.93, 1.3, "Ka not linear in power", color=ink2, fontsize=8, ha="right", va="bottom")
    ax.text(2.02, 300, "excluded by the records", color=ink2, fontsize=8, va="center")
    for lvl, col, lab in (("q97.5", "#2a78d6", "absorbed power at its 97.5th percentile"), ("median", "#eb6834", "median absorbed power")):
        xs = sorted(plot[lvl])
        ax.plot(xs, [plot[lvl][x] for x in xs], color=col, lw=2, marker="o", ms=6, mec=surf, mew=2, label=lab, zorder=3)
    ax.axvline(max(ka_lin), color=ink2, lw=1, ls="--")
    ax.text(max(ka_lin) - 0.01, 700, "Ka linear\nup to here", color=ink2, fontsize=8, ha="right", va="top")
    ax.axvspan(max(ka_lin) + 0.005, 2.95, color="#f3f2ef", lw=0, zorder=-1)
    ax.set_yscale("log")
    ax.set_xlim(1.95, 2.95)
    ax.set_ylim(1, 1000)
    ax.set_xlabel("laser threshold current (A); never measured", color=ink)
    ax.set_ylabel("R_th heating needs, K/W (lifetime E_a <= 0.40 eV)", color=ink)
    ax.set_title("Heating reproduces the X day only where a line enters the grey band, left of the dashed line", fontsize=9, color=ink, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(c.OUT / "verdict_map.png", dpi=160, facecolor=surf)


if __name__ == "__main__":
    main()

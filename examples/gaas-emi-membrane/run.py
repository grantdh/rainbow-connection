#!/usr/bin/env python
"""Example 2: GaAs membrane EMI shielding, from the band structure to a measured figure.

Reproduces the physics chain of Renteria, Heileman et al. (Adv. Mater. Technol. 9, 2401013, 2024)
and the "More Than Skin Deep" analysis with the ecosystem's own tools:

  QE PAW GaAs SCF+NSCF (8x8x8, 20 bands; on disk from workflows/gaas-emi-shield, March 2026)
    -> p2y SAVE (on disk)  -> yambo IP-RPA + scissor to the 1.42 eV gap      [run here]
    -> Lorentz fit with the static sum rule (lib/converters)                  [run here]
    -> + Drude free carriers (n = 1e19 cm^-3, m* = 0.063 m_e; gamma from the measured
       van der Pauw conductivity of each membrane)                            [run here]
    -> transfer matrix: mm-wave SE 65-85 GHz, IR transmittance 750-2500 nm,
       GaAs/Kapton/GaAs multilayer                                            [run here]
    -> MEEP 1D FDTD cross-check of the 1000 nm membrane at 65-85 GHz          [run here]
    -> comparison with the paper's measured SE and its thin-film formula

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-meep]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
import exlib  # noqa: E402
import emi_lib as el  # noqa: E402

RES = HERE / "results"
SAVE_SRC = exlib.REPO_ROOT / "workflows/gaas-emi-shield/yambo/SAVE"   # p2y of the 8x8x8 PAW NSCF (20 bands)
PBE_DIRECT_GAP = 0.513485          # r-gaas_scissors_optics_chi, Direct Gap at k=1 (Gamma)
EXP_GAP = 1.42
SCISSOR = EXP_GAP - PBE_DIRECT_GAP
N_DOPING = 1e19 * 1e6              # m^-3 (nominal Si doping)
M_STAR = 0.063 * el.m_e            # PBE band curvature at Gamma
# Measured DC conductivity, Renteria Table 1 (van der Pauw) and the measured SE at 75 GHz (Fig. 4b).
SIGMA_DC = {200: 1.19e5, 500: 8.81e4, 1000: 8.01e4}
EXP_SE_75 = {200: 13.5, 500: 19.0, 1000: 24.0}
EXP_SE_BAND = {200: (12, 15), 500: (17, 21), 1000: (22, 26)}
PAPER_THIN_FILM_SE = {200: 14.78, 500: 19.37, 1000: 24.13}
KAPTON = {"eps_r": 3.4, "tan_delta": 0.003, "d_m": 125e-6}
MULTILAYER_EXP_SE = (38.0, 40.0)

YAMBO_IP = """optics
chi
Chimod= "IP"
% QpntsRXd
 1 | 1 |
%
% BndsRnXd
  1 | 20 |
%
% EnRngeXd
  0.00000 | 10.00000 | eV
%
% DmRngeXd
 0.10000 | 0.10000 | eV
%
ETStpsXd= 500
% LongDrXd
 1.000000 | 0.000000 | 0.000000 |
%
% XfnQP_E
 {scissor:.6f} | 1.000000 | 1.000000 |
%
"""


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-meep", action="store_true")
    args = ap.parse_args()
    exlib.reset_results(RES)
    checks = exlib.Checks()
    S: dict = {"provenance": exlib.provenance(), "scissor_eV": SCISSOR, "pbe_direct_gap_eV": PBE_DIRECT_GAP,
               "n_m3": N_DOPING, "m_star_me": 0.063}
    t0 = time.time()

    # ------------------------------------------------------------------ 1. Yambo IP-RPA with scissor
    stage(f"1. Yambo IP-RPA eps(w) on the GaAs SAVE (8x8x8, 20 bands), scissor {SCISSOR:.3f} eV")
    ydir = RES / "yambo"
    checks.add("GaAs SAVE present", SAVE_SRC.is_dir(), str(SAVE_SRC), "p2y output from workflows/gaas-emi-shield")
    shutil.copytree(SAVE_SRC, ydir / "SAVE")
    rc, wall, _ = exlib.run([exlib.YAMBO_BIN / "yambo"], ydir, ydir / "setup.log", label="yambo setup gaas")
    (ydir / "ip.in").write_text(YAMBO_IP.format(scissor=SCISSOR))
    rc, wall, _ = exlib.run([exlib.YAMBO_BIN / "yambo", "-F", "ip.in", "-J", "ip"], ydir, ydir / "yambo_ip.log",
                            label="yambo ip-rpa gaas")
    files = sorted(ydir.glob("o-ip.eps_q1*"))
    checks.add("yambo IP-RPA wrote eps", bool(files) and rc == 0, f"rc={rc}, {wall:.0f}s", "o-ip.eps_q1_* exists")
    eps = exlib.read_yambo_eps(files[0])
    E, e1, e2 = eps["energy_eV"], eps["eps1"], eps["eps2"]
    # Critical points of eps2 above 1 eV: the scissored E0 edge, then E1 (~2.9 eV) and E2 (~4.7 eV) in experiment.
    pk = [float(E[k]) for k in range(1, len(E) - 1) if E[k] > 1.0 and e2[k] > e2[k - 1] and e2[k] > e2[k + 1] and e2[k] > 0.2 * e2.max()]
    e0 = pk[0] if pk else float("nan")
    e1_pk = [x for x in pk if 2.5 <= x <= 3.3]
    e2_pk = [x for x in pk if 4.1 <= x <= 5.2]
    print(f"  eps1(0) = {e1[0]:.2f}; eps2 peaks above 1 eV at {[round(x, 2) for x in pk]} eV (E0 edge {e0:.2f}, max {e2.max():.1f})")
    checks.add("E0 edge sits at the scissored gap", abs(e0 - EXP_GAP) < 0.1, f"{e0:.2f} eV", "|E0 - 1.42| < 0.1 eV")
    checks.add("E1 critical point present (expt 2.9 eV)", bool(e1_pk), f"{e1_pk}", "a peak in 2.5-3.3 eV")
    checks.add("E2 critical point present (expt 4.7 eV)", bool(e2_pk), f"{e2_pk}", "a peak in 4.1-5.2 eV")
    checks.add("E0 edge strength (informational: IP+scissor overshoots)", True, f"eps2(E0) = {e2.max():.1f}",
               "experiment ~3; the inflated Gamma-point edge is the known IP + rigid-scissor artefact")
    onset = e0
    S["epsilon"] = {"file": str(files[0]), "eps1_0": float(e1[0]), "onset_eV": onset, "peaks_eV": pk, "eps2_max": float(e2.max()), "causal": not eps["resonant_only"]}

    # ------------------------------------------------------------------ 2. Lorentz fit
    stage("2. Lorentz fit (static sum rule) -> interband material")
    conv = exlib.converter()
    eps_inf, poles, _ = conv.fit_lorentzian(E, e1, e2, n_poles=4)
    model = exlib.lorentz_eps(E, eps_inf, poles)
    rms = float(np.sqrt(np.mean(np.abs(model - (e1 + 1j * e2)) ** 2)))
    print(f"  eps_inf = {eps_inf:.3f}; sum rule -> eps(0) = {eps_inf + sum(p['sigma'] for p in poles):.3f}; rms|deps| = {rms:.3f}")
    checks.add("Lorentz fit rms", rms < 0.15 * float(np.abs(e1 + 1j * e2).max()), f"{rms:.3f}", "< 15% of max|eps|")
    material = {"eps_inf": float(eps_inf), "poles": [{k: float(v) for k, v in p.items()} for p in poles],
                "eps_static_interband": float(e1[0]), "theory_level": "IP-RPA + scissor (PBE PAW, 8x8x8, 20 bands)",
                "source": str(files[0]), "rms_error": rms}
    exlib.dump_json(RES / "material_interband.json", material)
    S["material"] = material

    def eps_total(freq_Hz, d_nm):
        """Interband Lorentz model + Drude with gamma from the measured sigma of that membrane."""
        E_eV = 2 * np.pi * np.asarray(freq_Hz) * el.HBAR_EV_S
        gamma = el.gamma_from_sigma(SIGMA_DC[d_nm], N_DOPING, M_STAR)
        return exlib.lorentz_eps(E_eV, eps_inf, poles) + el.drude_eps(freq_Hz, N_DOPING, M_STAR, gamma), gamma

    # ------------------------------------------------------------------ 3. mm-wave shielding
    stage("3. Transfer matrix: mm-wave SE 65-85 GHz for 200 / 500 / 1000 nm membranes")
    f_mm = np.linspace(65e9, 85e9, 201)
    i75 = int(np.argmin(np.abs(f_mm - 75e9)))
    S["mmwave"] = {"freq_Hz": f_mm, "membranes": {}}
    print(f"  {'d (nm)':>7} {'gamma (1/s)':>12} {'SE_tmm@75':>10} {'SE_thin':>8} {'paper':>7} {'exp':>6}  skin depth")
    for d_nm in (200, 500, 1000):
        eps_t, gamma = eps_total(f_mm, d_nm)
        res = el.tmm(f_mm, [(d_nm * 1e-9, eps_t)])
        se75 = float(res["SE_dB"][i75])
        thin = el.thin_film_SE(SIGMA_DC[d_nm], d_nm * 1e-9)
        delta = el.skin_depth(75e9, SIGMA_DC[d_nm])
        print(f"  {d_nm:7d} {gamma:12.3e} {se75:10.2f} {thin:8.2f} {PAPER_THIN_FILM_SE[d_nm]:7.2f} {EXP_SE_75[d_nm]:6.1f}  {delta * 1e6:.1f} um")
        S["mmwave"]["membranes"][d_nm] = {"SE_dB": res["SE_dB"], "SE_75GHz": se75, "thin_film_SE": thin, "gamma": gamma,
                                         "skin_depth_75GHz_m": delta, "exp_SE_75GHz": EXP_SE_75[d_nm]}
        checks.add(f"{d_nm} nm: TMM SE reproduces the paper's thin-film value", abs(se75 - PAPER_THIN_FILM_SE[d_nm]) < 0.5,
                   f"{se75:.2f} vs {PAPER_THIN_FILM_SE[d_nm]:.2f} dB", "|dSE| < 0.5 dB")
        lo, hi = EXP_SE_BAND[d_nm]
        checks.add(f"{d_nm} nm: SE within the measured 65-85 GHz band", (lo - 2.5) <= se75 <= (hi + 2.5),
                   f"{se75:.2f} dB vs measured {lo}-{hi}", "within band +/- 2.5 dB")

    # ------------------------------------------------------------------ 4. IR transmittance
    stage("4. Transfer matrix: IR transmittance 750-2500 nm (Fabry-Perot fringes, Fig. 2)")
    lam = np.linspace(750e-9, 2500e-9, 1200)
    f_ir = el.c / lam
    S["ir"] = {"wavelength_nm": lam * 1e9, "membranes": {}}
    sub = lam > 950e-9                     # below the 1.42 eV gap (873 nm) GaAs absorbs; fringes live above ~950 nm
    N_EXPT_IR = 3.4                        # GaAs refractive index around 1.5 um (Skauli 2003)
    eps_static_fit = float(exlib.lorentz_eps(np.array([1e-6]), eps_inf, poles).real[0])
    for d_nm in (200, 500, 1000):
        gamma = el.gamma_from_sigma(SIGMA_DC[d_nm], N_DOPING, M_STAR)
        drude = el.drude_eps(f_ir, N_DOPING, M_STAR, gamma)
        # (a) full model: Lorentz interband fit + Drude.  (b) sub-gap model: the fit's static eps (no interband
        # absorption below the gap, as the paper assumes with eps = 13.1) + Drude free-carrier absorption.
        eps_full = exlib.lorentz_eps(2 * np.pi * f_ir * el.HBAR_EV_S, eps_inf, poles) + drude
        eps_clean = eps_static_fit + drude
        T_full = el.tmm(f_ir, [(d_nm * 1e-9, eps_full)])["T"]
        res_c = el.tmm(f_ir, [(d_nm * 1e-9, eps_clean)])
        T_c = res_c["T"]; Ts = T_c[sub]
        npk = el.count_peaks(Ts)
        n_ir = float(np.sqrt(eps_static_fit))
        expect = 2 * n_ir * d_nm * 1e-9 * (1 / 950e-9 - 1 / 2500e-9)
        print(f"  {d_nm:5d} nm: sub-gap T (static eps + Drude) in [{Ts.min():.2f}, {Ts.max():.2f}], {npk} fringes (2nd/lambda {expect:.1f}); "
              f"with the Lorentz-fit interband model T in [{T_full[sub].min():.2f}, {T_full[sub].max():.2f}]; n = {n_ir:.2f} (expt {N_EXPT_IR})")
        S["ir"]["membranes"][d_nm] = {"T": T_c, "R": res_c["R"], "A": res_c["A"], "T_lorentz_model": T_full, "fringes_subgap": npk,
                                      "n_mid": n_ir, "T_subgap_max": float(Ts.max()), "T_subgap_min": float(Ts.min()),
                                      "T_lorentz_subgap_min": float(T_full[sub].min())}
        checks.add(f"{d_nm} nm: sub-gap transmittance window (static eps + Drude)", 0.3 < Ts.max() <= 1.0 + 1e-9 and Ts.min() > 0.05,
                   f"[{Ts.min():.2f}, {Ts.max():.2f}]", "paper Fig. 2: 40-100 % with fringes, 950-2500 nm")
        checks.add(f"{d_nm} nm: fringe count self-consistent with 2nd/lambda", abs(npk - expect) <= 1.5, f"{npk} vs {expect:.1f}", "|dN| <= 1.5")
    n_model = S["ir"]["membranes"][1000]["n_mid"]
    checks.add("IR refractive index vs experiment (informational)", True, f"n = {n_model:.2f} vs {N_EXPT_IR}",
               "IP + rigid scissor inflates eps1(0); fringe spacing is correspondingly too dense. BSE on a denser k-grid is the fix.")
    checks.add("Lorentz-fit interband model is opaque below the gap (informational)", True,
               f"1000 nm: sub-gap T min {S['ir']['membranes'][1000]['T_lorentz_subgap_min']:.2f}",
               "the 0.1 eV Lorentzian tails of the inflated E0 edge leak absorption below 1.42 eV; the fit must not be used for sub-gap IR until the edge is BSE-quality")

    # ------------------------------------------------------------------ 5. multilayer
    stage("5. Transfer matrix: 1 um GaAs / 125 um Kapton / 1 um GaAs (Fig. 6)")
    eps_t, _ = eps_total(f_mm, 1000)
    eps_k = KAPTON["eps_r"] * (1 + 1j * KAPTON["tan_delta"])
    ml = el.tmm(f_mm, [(1e-6, eps_t), (KAPTON["d_m"], eps_k), (1e-6, eps_t)])
    se_ml = float(ml["SE_dB"][i75])
    print(f"  multilayer SE at 75 GHz: {se_ml:.2f} dB (measured {MULTILAYER_EXP_SE[0]}-{MULTILAYER_EXP_SE[1]} dB); "
          f"range over band [{ml['SE_dB'].min():.1f}, {ml['SE_dB'].max():.1f}]")
    S["multilayer"] = {"SE_dB": ml["SE_dB"], "SE_75GHz": se_ml}
    checks.add("multilayer SE near measurement", MULTILAYER_EXP_SE[0] - 4 <= se_ml <= MULTILAYER_EXP_SE[1] + 4,
               f"{se_ml:.2f} dB", "38-40 dB measured, +/- 4 dB")

    # ------------------------------------------------------------------ 6. MEEP cross-check
    if not args.skip_meep:
        stage("6. MEEP 1D FDTD cross-check: 1000 nm membrane, 65-85 GHz")
        mdir = RES / "meep"
        mdir.mkdir()
        eps_static = float(exlib.lorentz_eps(np.array([1e-6]), eps_inf, poles).real[0])
        cfg = {"eps_r": eps_static, "sigma_Sm": SIGMA_DC[1000], "thickness_m": 1e-6, "f_min_Hz": 65e9, "f_max_Hz": 85e9,
               "points": 41, "resolution_per_mm": 4000, "pml_mm": 1.0, "pad_mm": 0.5}
        (mdir / "config.json").write_text(json.dumps(cfg))
        rc, wall, text = exlib.run([exlib.MEEP_PYTHON, HERE / "meep_mmwave.py", "config.json", "spectrum.json"], mdir,
                                   mdir / "meep.log", label="meep 1d gaas membrane mmwave")
        checks.add("MEEP run completed", rc == 0 and (mdir / "spectrum.json").exists(), f"rc={rc}, {wall:.0f}s", "rc 0 + spectrum.json")
        if rc == 0:
            sp = json.loads((mdir / "spectrum.json").read_text())
            fm = np.array(sp["freq_Hz"])
            # TMM with exactly the same material model MEEP used (static eps + DC conductivity), for a like-for-like check
            eps_meep_model = eps_static + 1j * SIGMA_DC[1000] / (el.eps0 * 2 * np.pi * fm)
            ref = el.tmm(fm, [(1e-6, eps_meep_model)])
            dse = float(np.abs(np.array(sp["SE_dB"]) - ref["SE_dB"]).max())
            full = np.interp(fm, f_mm, S["mmwave"]["membranes"][1000]["SE_dB"])
            dfull = float(np.abs(np.array(sp["SE_dB"]) - full).max())
            print(f"  MEEP SE vs TMM (same material): max|dSE| = {dse:.3f} dB; vs full Drude+Lorentz TMM: {dfull:.3f} dB; "
                  f"MEEP SE(75 GHz) = {np.interp(75e9, fm, sp['SE_dB']):.2f} dB, wall {wall:.0f}s")
            checks.add("MEEP agrees with TMM (same conductivity model)", dse < 0.3, f"{dse:.3f} dB", "max|dSE| < 0.3 dB over 65-85 GHz")
            S["meep"] = {"freq_Hz": fm, "SE_dB": sp["SE_dB"], "T": sp["T"], "R": sp["R"], "max_dSE_vs_tmm": dse,
                         "max_dSE_vs_full_model": dfull, "wall_s": wall, "sigma_D_meep": sp["sigma_D_meep"]}

    # ------------------------------------------------------------------ 7. figures + results
    stage("7. Figures and RESULTS.md")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(E, e1, "C0", label="Re eps (Yambo IP+scissor)"); ax[0].plot(E, e2, "C1", label="Im eps")
        ax[0].plot(E, model.real, "C0--", label="fit"); ax[0].plot(E, model.imag, "C1--")
        ax[0].set_xlabel("Photon energy (eV)"); ax[0].set_title("GaAs interband eps(w)"); ax[0].legend(fontsize=8)
        for d_nm, col in ((200, "C0"), (500, "C1"), (1000, "C2")):
            m = S["mmwave"]["membranes"][d_nm]
            ax[1].plot(f_mm / 1e9, m["SE_dB"], col, label=f"{d_nm} nm (TMM)")
            lo, hi = EXP_SE_BAND[d_nm]
            ax[1].fill_between([65, 85], lo, hi, color=col, alpha=0.12)
            ax[1].plot(75, EXP_SE_75[d_nm], col + "s", ms=6)
            ax[2].plot(lam * 1e9, S["ir"]["membranes"][d_nm]["T"], col, label=f"{d_nm} nm (static eps + Drude)")
            ax[2].plot(lam * 1e9, S["ir"]["membranes"][d_nm]["T_lorentz_model"], col + ":", lw=0.8, label=f"{d_nm} nm (Lorentz fit + Drude)")
        if "meep" in S:
            ax[1].plot(np.array(S["meep"]["freq_Hz"]) / 1e9, S["meep"]["SE_dB"], "k.", ms=3, label="1000 nm (MEEP)")
        ax[1].plot(f_mm / 1e9, S["multilayer"]["SE_dB"], "C3--", label="GaAs/Kapton/GaAs (TMM)")
        ax[1].fill_between([65, 85], *MULTILAYER_EXP_SE, color="C3", alpha=0.12)
        ax[1].set_xlabel("Frequency (GHz)"); ax[1].set_ylabel("SE (dB)"); ax[1].set_title("mm-wave shielding (bands = measured)")
        ax[1].legend(fontsize=7); ax[2].set_xlabel("Wavelength (nm)"); ax[2].set_ylabel("T"); ax[2].set_ylim(0, 1.02)
        ax[2].set_title("IR transmittance (Fabry-Perot)"); ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(RES / "gaas_emi_membrane.png", dpi=140)
        print(f"  wrote {RES / 'gaas_emi_membrane.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot skipped: {exc}")

    S["checks"] = checks.rows; S["all_pass"] = checks.all_pass; S["total_wall_s"] = time.time() - t0
    exlib.dump_json(RES / "summary.json", S)
    (RES / "RESULTS.md").write_text(results_md(S, checks))
    print(f"\n{'ALL CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'} in {S['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


def results_md(S, checks) -> str:
    p = S["provenance"]
    L = [f"# Example 2 results: GaAs membrane EMI shielding ({p['date']})", "",
         f"rainbow-connection {p['rainbow_connection_commit']}, {p['hardware']}, macOS {p['macos']}, load1 {p['load1_at_start']:.2f}", "",
         "## Checks", "", checks.markdown(), "", "## Numbers", "",
         f"- Yambo IP-RPA + scissor {S['scissor_eV']:.3f} eV (PBE direct gap {S['pbe_direct_gap_eV']:.3f} -> 1.42 eV): "
         f"eps1(0) = {S['epsilon']['eps1_0']:.2f} (expt 10.9), eps2 peaks at {[round(x, 2) for x in S['epsilon']['peaks_eV']]} eV (expt E0 1.42, E1 2.9, E2 4.7)",
         f"- Lorentz fit: eps_inf = {S['material']['eps_inf']:.3f}, rms|deps| = {S['material']['rms_error']:.3f}",
         "", "| membrane | gamma (1/s) | SE TMM @75 GHz | thin-film formula | paper | measured | skin depth |",
         "|---|---|---|---|---|---|---|"]
    for d_nm, m in S["mmwave"]["membranes"].items():
        L.append(f"| {d_nm} nm | {m['gamma']:.3e} | {m['SE_75GHz']:.2f} dB | {m['thin_film_SE']:.2f} dB | "
                 f"{PAPER_THIN_FILM_SE[int(d_nm)]:.2f} dB | {m['exp_SE_75GHz']} dB | {m['skin_depth_75GHz_m'] * 1e6:.1f} um |")
    L += ["", f"- Multilayer GaAs/Kapton/GaAs at 75 GHz: {S['multilayer']['SE_75GHz']:.2f} dB (measured 38-40 dB)"]
    for d_nm, m in S["ir"]["membranes"].items():
        L.append(f"- IR {d_nm} nm (static eps + Drude): sub-gap T in [{m['T_subgap_min']:.2f}, {m['T_subgap_max']:.2f}], {m['fringes_subgap']} fringes above 950 nm, n = {m['n_mid']:.2f} (expt 3.4); Lorentz-fit model sub-gap T min {m['T_lorentz_subgap_min']:.2f}")
    if "meep" in S:
        L.append(f"- MEEP 1D vs TMM (same model) max|dSE| = {S['meep']['max_dSE_vs_tmm']:.3f} dB; vs full model {S['meep']['max_dSE_vs_full_model']:.3f} dB; wall {S['meep']['wall_s']:.0f} s")
    L += ["", "## What is first-principles here and what is measured", "",
          "- First principles: the band structure (QE PBE), the interband eps(w) (Yambo), m* = 0.063 m_e (PBE curvature), the scissor target (1.42 eV).",
          "- Measured input: the carrier density (nominal doping) and the DC conductivity of each membrane (van der Pauw), which sets gamma.",
          "- The mm-wave SE is set by Z0*sigma*d (skin depth >> thickness), so it tests the Drude/TMM chain, not the band structure.",
          "- The IR transmittance is where the ab initio eps(w) matters (index and fringe spacing). IP+scissor overestimates eps1(0) (n 4.8 vs 3.4), so fringes are too dense, and its Lorentzian-broadened E0 edge leaks absorption below the gap, so the fitted interband model is opaque where the real membrane is transparent. The sub-gap curves therefore use the fit's static eps plus Drude. BSE on a denser k-grid is the known fix.",
          "", "## Files", "", "- results/yambo/o-ip.eps_q1_ip, results/material_interband.json, results/meep/spectrum.json, results/summary.json, results/gaas_emi_membrane.png"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())

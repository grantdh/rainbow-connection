#!/usr/bin/env python
"""Example 1: ab initio optical response of a 500 nm silicon film.

Chain (every arrow is a file handoff; see README.md):

  QE pw.x SCF ─┬─> apple-bottom parity leg (frozen pw.x, AB_MODE=cpu vs gpu, AB_PROFILE)
               └─> QE NSCF (20 bands) -> p2y -> yambo setup -> yambo IP-RPA + scissor
                   -> Lorentz fit (lib/converters/yambo_eps_to_meep.py, static sum rule)
                   -> analytic Airy T/R/A  and  MEEP 1D FDTD T/R/A  -> checks, plots, RESULTS.md

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-ab] [--skip-meep]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import exlib  # noqa: E402

RES = HERE / "results"
SCISSOR_EV = 0.65        # rigid shift of the PBE conduction bands (GW for Si gives ~0.6-0.7 eV)
NBND = 20
THICK_NM = 500.0
E_RANGE = (1.0, 4.0)
REF_ENERGY_RY = -22.83836927   # workflows/si-photovoltaic/inputs/si_scf.out, same input, QE 7.4.1, 2026-03-29

SI_SCF = """&CONTROL
  calculation = '{calc}'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.20
  nat = 2
  ntyp = 1
  ecutwfc = 30.0
  force_symmorphic = .true.{nbnd}
/
&ELECTRONS
  conv_thr = 1.0d-8{diago}
/
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4 {shift}
"""

YAMBO_IP = """optics
chi
Chimod= "IP"
% QpntsRXd
 1 | 1 |
%
% BndsRnXd
  1 | {nb} |
%
% EnRngeXd
  0.00000 | 8.00000 | eV
%
% DmRngeXd
 0.10000 | 0.10000 | eV
%
ETStpsXd= 400
% LongDrXd
 1.000000 | 0.000000 | 0.000000 |
%
% XfnQP_E
 {scissor:.6f} | 1.000000 | 1.000000 |
%
"""


def qe_input(calc: str) -> str:
    # SCF on the shifted mesh (the March reference); NSCF for Yambo on the Gamma-centred mesh, as muppet-studio does.
    return SI_SCF.format(calc=calc, nbnd=f"\n  nbnd = {NBND}" if calc == "nscf" else "",
                         diago="\n  diago_full_acc = .true." if calc == "nscf" else "",
                         shift="0 0 0" if calc == "nscf" else "1 1 1")


def stage(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ab", action="store_true", help="skip the apple-bottom parity leg")
    ap.add_argument("--skip-meep", action="store_true")
    args = ap.parse_args()

    exlib.reset_results(RES, keep=("run.log", "studio"))
    (RES / "qe").mkdir()
    checks = exlib.Checks()
    summary: dict = {"provenance": exlib.provenance(), "scissor_eV": SCISSOR_EV, "thickness_nm": THICK_NM}
    t_start = time.time()

    # ------------------------------------------------------------------ 1. QE SCF + NSCF (vanilla 7.4.1)
    stage("1. Quantum ESPRESSO SCF + NSCF (vanilla pw.x)")
    qe = RES / "qe"
    (qe / "pseudo").mkdir()
    os.symlink(exlib.PSEUDO_SI, qe / "pseudo" / exlib.PSEUDO_SI.name)
    (qe / "scf.in").write_text(qe_input("scf"))
    (qe / "nscf.in").write_text(qe_input("nscf"))
    rc, wall, text = exlib.run([exlib.QE_BIN / "pw.x", "-in", "scf.in"], qe, qe / "scf.out", label="si scf vanilla")
    scf = exlib.parse_pw(text)
    print(f"  SCF rc={rc} wall={wall:.1f}s E_tot={scf.get('total_energy_Ry')} Ry gap={scf.get('gap_eV')} eV")
    checks.add("QE SCF converged", scf.get("job_done") and "total_energy_Ry" in scf, scf.get("scf_iterations"), "JOB DONE + total energy")
    de = abs(scf.get("total_energy_Ry", 0) - REF_ENERGY_RY)
    checks.add("SCF energy reproduces March reference", de < 1e-6, f"{de:.1e} Ry", "|dE| < 1e-6 Ry vs -22.83836927")
    rc, wall, text = exlib.run([exlib.QE_BIN / "pw.x", "-in", "nscf.in"], qe, qe / "nscf.out", label="si nscf vanilla")
    nscf = exlib.parse_pw(text)
    print(f"  NSCF rc={rc} wall={wall:.1f}s gap={nscf.get('gap_eV')} eV (PBE, Gamma-centred 4x4x4 mesh)")
    checks.add("QE NSCF produced 20 bands", nscf.get("job_done"), nscf.get("gap_eV"), "JOB DONE")
    summary["qe"] = {"scf": scf, "nscf": nscf, "scf_wall_s": wall}

    # ------------------------------------------------------------------ 1b. apple-bottom parity leg
    if not args.skip_ab and (exlib.AB_FROZEN / "pw.x").exists():
        stage("1b. apple-bottom parity: frozen pw.x, AB_MODE=cpu vs AB_MODE=gpu")
        ab_res = {}
        for mode, extra in (("cpu", {}), ("gpu", {"AB_MIN_GPU_DIM": "0"})):
            d = RES / "ab" / mode
            (d / "pseudo").mkdir(parents=True)
            os.symlink(exlib.PSEUDO_SI, d / "pseudo" / exlib.PSEUDO_SI.name)
            (d / "scf.in").write_text(qe_input("scf"))
            env = {"AB_MODE": mode, "AB_PROFILE": "1", "AB_PROFILE_JSON": str(d / "ab_profile.json"),
                   "OMP_NUM_THREADS": "1", **extra}
            rc, wall, text = exlib.run(["mpirun", "-np", "1", exlib.AB_FROZEN / "pw.x", "-in", "scf.in"], d,
                                       d / "scf.out", env=env, label=f"si scf apple-bottom AB_MODE={mode}")
            info = exlib.parse_pw(text)
            prof = {}
            if (d / "ab_profile.json").exists():
                try:
                    prof = json.loads((d / "ab_profile.json").read_text())
                except Exception as e:  # noqa: BLE001
                    prof = {"error": str(e)}
            ab_res[mode] = {"rc": rc, "wall_s": wall, **info, "profile": prof,
                            "stderr_tail": text[-1500:] if rc else ""}
            print(f"  AB_MODE={mode}: rc={rc} wall={wall:.1f}s E_tot={info.get('total_energy_Ry')} Ry "
                  f"iters={info.get('scf_iterations')}")
        summary["apple_bottom"] = ab_res
        e_cpu, e_gpu = ab_res["cpu"].get("total_energy_Ry"), ab_res["gpu"].get("total_energy_Ry")
        both = e_cpu is not None and e_gpu is not None
        checks.add("apple-bottom cpu/gpu energies identical", both and abs(e_cpu - e_gpu) < 1e-8,
                   f"cpu={e_cpu} gpu={e_gpu}", "|dE| < 1e-8 Ry on the same frozen binary")
        if both:
            checks.add("apple-bottom build agrees with vanilla 7.4.1", abs(e_cpu - scf["total_energy_Ry"]) < 1e-5,
                       f"{abs(e_cpu - scf['total_energy_Ry']):.1e} Ry", "|dE| < 1e-5 Ry (QE develop vs 7.4.1)")
            print(f"  wall: cpu {ab_res['cpu']['wall_s']:.1f}s, gpu {ab_res['gpu']['wall_s']:.1f}s "
                  f"(2-atom cell: every GEMM is far below the DD crossover; a slower GPU leg is expected)")
    else:
        stage("1b. apple-bottom parity leg skipped")

    # ------------------------------------------------------------------ 2. p2y + yambo setup
    stage("2. p2y (QE -> Yambo SAVE) and yambo setup")
    ydir = RES / "yambo"
    ydir.mkdir()
    save_src = qe / "tmp" / "si.save"
    rc, wall, text = exlib.run([exlib.YAMBO_BIN / "p2y"], save_src, ydir / "p2y.log", label="p2y si")
    checks.add("p2y wrote SAVE", (save_src / "SAVE").is_dir(), rc, "SAVE directory exists")
    shutil.move(str(save_src / "SAVE"), ydir / "SAVE")
    rc, wall, text = exlib.run([exlib.YAMBO_BIN / "yambo"], ydir, ydir / "setup.log", label="yambo setup si")
    checks.add("yambo setup", rc == 0 and any(ydir.glob("r_setup*")), rc, "r_setup written")

    # ------------------------------------------------------------------ 3. IP-RPA optics with scissor
    stage(f"3. Yambo IP-RPA eps(w), scissor {SCISSOR_EV} eV, bands 1-{NBND}")
    (ydir / "ip.in").write_text(YAMBO_IP.format(nb=NBND, scissor=SCISSOR_EV))
    rc, wall, text = exlib.run([exlib.YAMBO_BIN / "yambo", "-F", "ip.in", "-J", "ip"], ydir, ydir / "yambo_ip.log",
                               label="yambo ip-rpa si")
    files = sorted(ydir.glob("o-ip.eps_q1*"))
    checks.add("yambo IP-RPA wrote eps", bool(files), rc, "o-ip.eps_q1_* exists")
    eps = exlib.read_yambo_eps(files[0])
    e, e1, e2 = eps["energy_eV"], eps["eps1"], eps["eps2"]
    i_peak = int(np.argmax(e2))
    print(f"  eps1(0) = {e1[0]:.2f}; eps2 peak {e2[i_peak]:.1f} at {e[i_peak]:.2f} eV; "
          f"{len(e)} points, T-ordered (causal) = {not eps['resonant_only']}")
    checks.add("eps1(0) in the IP+scissor range for Si", 8.0 < e1[0] < 25.0, f"{e1[0]:.2f}",
               "8 < eps1(0) < 25 (expt 11.7; IP+scissor on a coarse Gamma-centred mesh overshoots: 21.9 here, 13.4 on the shifted mesh)")
    report = (ydir / "r-ip_optics_chi").read_text() if (ydir / "r-ip_optics_chi").exists() else ""
    gaps = re.findall(r"\[X(\+QP)?\] Direct Gap\s+:\s+([\d.]+)", report)
    gap_pbe = float(gaps[0][1]) if gaps else float("nan")
    gap_qp = float(gaps[-1][1]) if len(gaps) > 1 else gap_pbe + SCISSOR_EV
    onset = float(e[np.argmax(e2 > 0.05 * e2.max())])
    print(f"  Yambo direct gap on this k-mesh: {gap_pbe:.3f} eV (PBE) -> {gap_qp:.3f} eV with scissor; eps2 onset (5% of peak) {onset:.2f} eV")
    checks.add("scissor applied: direct gap shifted by SCISSOR_EV", abs(gap_qp - gap_pbe - SCISSOR_EV) < 1e-3,
               f"{gap_pbe:.3f} -> {gap_qp:.3f} eV", f"X+QP gap = X gap + {SCISSOR_EV}")
    checks.add("absorption onset at the scissored direct gap", abs(onset - gap_qp) < 0.4, f"{onset:.2f} vs {gap_qp:.2f} eV",
               "|onset - gap| < 0.4 eV (0.1 eV Lorentzian broadening)")
    summary["epsilon"] = {"file": str(files[0]), "eps1_0": float(e1[0]), "eps2_peak": float(e2[i_peak]),
                          "eps2_peak_eV": float(e[i_peak]), "onset_eV": onset, "direct_gap_pbe_eV": gap_pbe,
                          "direct_gap_scissored_eV": gap_qp, "causal": not eps["resonant_only"]}

    # ------------------------------------------------------------------ 4. Lorentz fit (sum rule)
    stage("4. Lorentz fit with the static sum rule (lib/converters/yambo_eps_to_meep.py)")
    conv = exlib.converter()
    eps_inf, poles, fit = conv.fit_lorentzian(e, e1, e2, n_poles=4)
    model = exlib.lorentz_eps(e, eps_inf, poles)
    rms = float(np.sqrt(np.mean(np.abs(model - (e1 + 1j * e2)) ** 2)))
    sum_rule = eps_inf + sum(p["sigma"] for p in poles)
    print(f"  eps_inf = {eps_inf:.3f}; poles: " + "; ".join(f"{p['omega_eV']:.2f} eV (g {p['gamma_eV']:.2f}, s {p['sigma']:.2f})" for p in poles))
    print(f"  eps_inf + sum(sigma) = {sum_rule:.3f} vs eps1(0) = {e1[0]:.3f}; rms|deps| = {rms:.3f}")
    checks.add("sum rule eps_inf+sum(sigma)=eps1(0)", abs(sum_rule - e1[0]) < 1e-6, f"{sum_rule:.4f}", "exact by construction")
    checks.add("Lorentz fit rms", rms < 0.15 * float(np.abs(e1 + 1j * e2).max()), f"{rms:.3f}", "< 15% of max|eps|")
    material = {"eps_inf": float(eps_inf), "poles": [{k: float(v) for k, v in p.items()} for p in poles],
                "source": str(files[0]), "scissor_eV": SCISSOR_EV, "rms_error": rms,
                "valid_range_eV": [float(e.min()), float(e.max())], "theory_level": "IP-RPA + scissor (PBE, 4x4x4, 20 bands)"}
    exlib.dump_json(RES / "material.json", material)
    summary["material"] = material

    # ------------------------------------------------------------------ 5. Airy and MEEP
    stage(f"5. Thin film T/R/A, {THICK_NM:g} nm: analytic Airy vs MEEP 1D FDTD")
    eg = np.linspace(E_RANGE[0], E_RANGE[1], 300)
    airy = exlib.airy_tra(exlib.lorentz_eps(eg, eps_inf, poles), THICK_NM, eg)
    checks.add("Airy energy conservation", bool(np.all(airy["A"] > -1e-9) and np.all(airy["A"] < 1 + 1e-9)),
               f"A in [{airy['A'].min():.3f}, {airy['A'].max():.3f}]", "0 <= A <= 1")
    summary["airy"] = {"energy_eV": eg, "T": airy["T"], "R": airy["R"], "A": airy["A"]}
    meep_out = None
    if not args.skip_meep:
        mdir = RES / "meep"
        mdir.mkdir()
        conv_rows = []
        for res_px in (200, 800):
            cfg = {"material": material, "thickness_nm": THICK_NM, "e_min_eV": E_RANGE[0], "e_max_eV": E_RANGE[1],
                   "points": 150, "resolution": res_px}
            (mdir / f"config_{res_px}.json").write_text(json.dumps(cfg))
            rc, wall, text = exlib.run([exlib.MEEP_PYTHON, HERE / "meep_thin_film.py", f"config_{res_px}.json", f"spectrum_{res_px}.json"],
                                       mdir, mdir / f"meep_{res_px}.log", label=f"meep 1d si film res {res_px}")
            if rc != 0 or not (mdir / f"spectrum_{res_px}.json").exists():
                checks.add(f"MEEP run completed (res {res_px})", False, f"rc={rc}", "rc 0 + spectrum.json")
                continue
            meep_out = json.loads((mdir / f"spectrum_{res_px}.json").read_text())
            em = np.array(meep_out["energy_eV"])
            Tm, Rm = np.array(meep_out["T"]), np.array(meep_out["R"])
            Ta, Ra = np.interp(em, eg, airy["T"]), np.interp(em, eg, airy["R"])
            row = {"resolution": res_px, "wall_s": wall, "max_dT": float(np.abs(Tm - Ta).max()), "rms_dT": float(np.sqrt(np.mean((Tm - Ta) ** 2))),
                   "max_dR": float(np.abs(Rm - Ra).max()), "rms_dR": float(np.sqrt(np.mean((Rm - Ra) ** 2)))}
            conv_rows.append(row)
            print(f"  MEEP res {res_px} px/um ({wall:.0f}s): max|dT| {row['max_dT']:.4f} rms {row['rms_dT']:.4f}; max|dR| {row['max_dR']:.4f} rms {row['rms_dR']:.4f}")
            summary["meep"] = {"energy_eV": em, "T": Tm, "R": Rm, "A": 1 - Tm - Rm, "max_dT": row["max_dT"], "max_dR": row["max_dR"],
                               "rms_dT": row["rms_dT"], "rms_dR": row["rms_dR"], "wall_s": wall, "resolution": res_px}
        summary["meep_convergence"] = conv_rows
        if conv_rows:
            last = conv_rows[-1]
            checks.add("MEEP run completed", True, f"{len(conv_rows)} resolutions", "rc 0 + spectrum.json")
            checks.add("MEEP converges to Airy (T)", last["rms_dT"] < 0.01 and last["max_dT"] < 0.03, f"rms {last['rms_dT']:.4f}, max {last['max_dT']:.4f} at {last['resolution']} px/um",
                       "rms|dT| < 0.01 and max|dT| < 0.03 at the finest resolution")
            checks.add("MEEP converges to Airy (R)", last["rms_dR"] < 0.015 and last["max_dR"] < 0.04, f"rms {last['rms_dR']:.4f}, max {last['max_dR']:.4f}",
                       "rms|dR| < 0.015 and max|dR| < 0.04 at the finest resolution")
            if len(conv_rows) > 1:
                ratio = conv_rows[0]["rms_dT"] / max(last["rms_dT"], 1e-12)
                checks.add("MEEP error shrinks with resolution", ratio > 2.0, f"rms|dT| ratio {ratio:.1f} for {conv_rows[0]['resolution']} -> {last['resolution']}",
                           "> 2x reduction (first-order interface error)")
            Am = summary["meep"]["A"]
            checks.add("MEEP energy conservation", bool(np.all(Am > -0.02) and np.all(Am < 1.02)),
                       f"A in [{Am.min():.3f}, {Am.max():.3f}]", "-0.02 <= A <= 1.02")

    # ------------------------------------------------------------------ 6. plots + results
    stage("6. Plots and RESULTS.md")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(e, e1, "C0", label="Re eps (Yambo IP+scissor)")
        ax[0].plot(e, e2, "C1", label="Im eps")
        ax[0].plot(e, model.real, "C0--", label="Re eps (Lorentz fit)")
        ax[0].plot(e, model.imag, "C1--", label="Im eps (fit)")
        ax[0].set_xlabel("Photon energy (eV)"); ax[0].set_ylabel("eps"); ax[0].legend(fontsize=8)
        ax[0].set_title("Si dielectric function: QE -> Yambo -> Lorentz fit")
        for k, c in (("T", "C2"), ("R", "C3"), ("A", "C4")):
            ax[1].plot(eg, airy[k], c, label=f"{k} (Airy)")
            if meep_out is not None:
                ax[1].plot(summary["meep"]["energy_eV"], summary["meep"][k], c + "o", ms=2.5, label=f"{k} (MEEP)")
        ax[1].set_xlabel("Photon energy (eV)"); ax[1].set_ylabel("fraction"); ax[1].set_ylim(-0.02, 1.02)
        ax[1].set_title(f"{THICK_NM:g} nm Si film, normal incidence"); ax[1].legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(RES / "si_thin_film.png", dpi=140)
        print(f"  wrote {RES / 'si_thin_film.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot skipped: {exc}")

    summary["checks"] = checks.rows
    summary["all_pass"] = checks.all_pass
    summary["total_wall_s"] = time.time() - t_start
    exlib.dump_json(RES / "summary.json", summary)
    (RES / "RESULTS.md").write_text(results_md(summary, checks))
    print(f"\n{'ALL CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'} in {summary['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


def results_md(s: dict, checks: exlib.Checks) -> str:
    p = s["provenance"]
    lines = [f"# Example 1 results: Si thin film ({p['date']})", "",
             f"rainbow-connection {p['rainbow_connection_commit']}, {p['hardware']}, macOS {p['macos']}, "
             f"QE {s['qe']['scf'].get('qe_version')}, MEEP {s.get('meep', {}).get('resolution', '-')} px/um, load1 {p['load1_at_start']:.2f}", "",
             "## Checks", "", checks.markdown(), "",
             "## Numbers", "",
             f"- QE SCF total energy: {s['qe']['scf'].get('total_energy_Ry')} Ry in {s['qe']['scf'].get('scf_iterations')} iterations (reference {REF_ENERGY_RY})",
             f"- PBE gap from NSCF (HOMO/LUMO on the 4x4x4 grid): {s['qe']['nscf'].get('gap_eV')} eV; scissor applied in Yambo: {s['scissor_eV']} eV",
             f"- Yambo direct gap on the 4x4x4 mesh: {s['epsilon']['direct_gap_pbe_eV']:.3f} eV (PBE) -> {s['epsilon']['direct_gap_scissored_eV']:.3f} eV scissored; eps1(0) = {s['epsilon']['eps1_0']:.2f}; eps2 peak {s['epsilon']['eps2_peak']:.1f} at {s['epsilon']['eps2_peak_eV']:.2f} eV; onset {s['epsilon']['onset_eV']:.2f} eV",
             f"- Lorentz fit: eps_inf = {s['material']['eps_inf']:.3f}, rms|deps| = {s['material']['rms_error']:.3f}, poles at "
             + ", ".join(f"{q['omega_eV']:.2f} eV" for q in s['material']['poles'])]
    if s.get("meep_convergence"):
        lines += ["", "| MEEP resolution (px/um) | wall (s) | rms dT | max dT | rms dR | max dR |", "|---|---|---|---|---|---|"]
        for r in s["meep_convergence"]:
            lines.append(f"| {r['resolution']} | {r['wall_s']:.1f} | {r['rms_dT']:.4f} | {r['max_dT']:.4f} | {r['rms_dR']:.4f} | {r['max_dR']:.4f} |")
        lines.append("")
        lines.append("MEEP's remaining difference from the analytic Airy result halves with each doubling of resolution (first-order interface error of the dispersive slab), which is the expected behaviour.")
    if "apple_bottom" in s:
        ab = s["apple_bottom"]
        lines += ["", "## apple-bottom parity leg (frozen pw.x + libapplebottom, 1 MPI rank)", "",
                  "| AB_MODE | rc | E_tot (Ry) | SCF iters | wall (s) |", "|---|---|---|---|---|"]
        for m in ("cpu", "gpu"):
            r = ab[m]
            lines.append(f"| {m} | {r['rc']} | {r.get('total_energy_Ry')} | {r.get('scf_iterations')} | {r['wall_s']:.1f} |")
        lines += ["", "The 2-atom cell keeps every GEMM far below the DD crossover, so this leg tests correctness "
                  "(identical energies, honest dispatch report), not speed. Profile JSON: results/ab/*/ab_profile.json."]
    lines += ["", "## Files", "", "- results/qe/{scf,nscf}.out, results/yambo/o-ip.eps_q1_ip, results/material.json, "
              "results/meep/spectrum.json, results/summary.json, results/si_thin_film.png"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())

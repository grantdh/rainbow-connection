#!/usr/bin/env python
"""Example 4: silicon equation of state and band structure against published PBE results,
with apple-bottom parity along the whole energy-volume curve.

  QE pw.x SCF at 9 lattice constants (PBE, ultrasoft Si, 6x6x6 shifted mesh)
    -> Birch-Murnaghan fit: a0, B0, B0'            vs literature PBE values (and experiment)
    -> the same 9 SCFs on the frozen apple-bottom pw.x, AB_MODE=gpu, AB_MIN_GPU_DIM=0
       (every point must agree with the reference binary; the fitted a0 must agree)
    -> SCF at the experimental lattice constant + "bands" run on Gamma, X, L and the Delta line
       -> indirect gap, its k position, direct Gamma gap, L1c   vs well-known PBE numbers

References (literature, tagged where used):
  PBE Si a0 = 5.469 A   [Haas, Tran, Blaha, PRB 79, 085104 (2009), Table I]; all-electron WIEN2k Delta-test
                        reference V0 = 20.4530 A^3/atom -> 5.4695 A, B0 = 88.545 GPa, B0' = 4.31
                        [Lejaeghere et al., Science 351, aad3000 (2016)]
  PBE B0 = 89.2 GPa, experiment B0 = 99.2 GPa [Csonka et al., PRB 79, 155107 (2009), Table V]
  experiment a0 = 5.431 A (room temperature; CODATA 2018 5.431020 A at 22.5 C)
  PBE band gaps at the experimental lattice constant: indirect ~0.61 eV with the CBM at ~0.85 of Gamma-X,
  direct gap at Gamma ~2.56 eV, L1c ~1.5 eV above the VBM (standard textbook PBE values; experiment 1.17, 3.4, 2.0)

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-ab]   (~3 min)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import exlib  # noqa: E402

RES = HERE / "results"
PSEUDO = exlib.PSEUDO_SI                      # Si.pbe-n-rrkjus_psl.1.0.0.UPF (PBE, ultrasoft; fine for QE)
ECUT, ECUTRHO = 45.0, 360.0                   # the UPF header suggests >= 44 Ry for the wavefunctions
BOHR_A = 0.529177210903
RY_BOHR3_GPA = 14710.507                      # 1 Ry/bohr^3 in GPa
A_EXP_BOHR = 5.431 / BOHR_A
LIT = {"a0_pbe_A": 5.469, "B0_pbe_GPa": 89.2, "a0_exp_A": 5.431, "B0_exp_GPa": 99.2, "a0_ae_A": 5.4695, "B0_ae_GPa": 88.545, "Bp_ae": 4.31}
A_SCAN = np.linspace(10.05, 10.61, 9)         # bohr, centred on the PBE a0 (10.33 bohr)

QE_IN = """&CONTROL
  calculation = '{calc}'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 2
  celldm(1) = {a:.5f}
  nat = 2
  ntyp = 1
  ecutwfc = {ecut}
  ecutrho = {ecutrho}{nbnd}
/
&ELECTRONS
  conv_thr = 1.0d-9
/
ATOMIC_SPECIES
  Si 28.086 {pseudo}
ATOMIC_POSITIONS crystal
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
{kpoints}
"""
K_SCF = "K_POINTS automatic\n  6 6 6 1 1 1\n"
# Gamma, L, X, then the Delta line Gamma->X (x,0,0) in 2pi/a units
DELTA = np.round(np.arange(0.70, 1.0001, 0.025), 3)
K_BANDS = "K_POINTS tpiba\n  {n}\n  0.0 0.0 0.0 1\n  0.5 0.5 0.5 1\n" + "".join(f"  {x:.3f} 0.0 0.0 1\n" for x in DELTA)
K_BANDS = K_BANDS.format(n=2 + len(DELTA))


def qe_input(calc, a, kpoints, nbnd=None):
    return QE_IN.format(calc=calc, a=a, ecut=ECUT, ecutrho=ECUTRHO, pseudo=PSEUDO.name, kpoints=kpoints,
                        nbnd=f"\n  nbnd = {nbnd}" if nbnd else "")


def prep(d: Path):
    (d / "pseudo").mkdir(parents=True, exist_ok=True)
    link = d / "pseudo" / PSEUDO.name
    if not link.exists():
        os.symlink(PSEUDO, link)


def birch_murnaghan(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))


def fit_eos(V, E):
    from scipy.optimize import least_squares
    i = int(np.argmin(E))
    x0 = [E[i], V[i], 0.006, 4.0]
    r = least_squares(lambda x: birch_murnaghan(V, *x) - E, x0, max_nfev=20000)
    E0, V0, B0, Bp = r.x
    resid = float(np.sqrt(np.mean((birch_murnaghan(V, *r.x) - E) ** 2)))
    return {"E0_Ry": E0, "V0_bohr3": V0, "a0_bohr": (4 * V0) ** (1 / 3), "a0_A": (4 * V0) ** (1 / 3) * BOHR_A,
            "B0_GPa": B0 * RY_BOHR3_GPA, "Bp": Bp, "rms_Ry": resid}


def parse_bands(text):
    """Eigenvalues per k-point from a verbosity='high' pw.x output (tpiba coordinates)."""
    blocks = re.findall(r"k =\s*(-?\d+\.\d{4})\s*(-?\d+\.\d{4})\s*(-?\d+\.\d{4})\s*\(\s*\d+ PWs\)\s+bands \(ev\):\s+((?:[-\d.\s]+\n)+)", text)
    out = []
    for kx, ky, kz, body in blocks:
        vals = [float(v) for v in re.findall(r"-?\d+\.\d+", body)]
        out.append(((float(kx), float(ky), float(kz)), vals))
    return out


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-ab", action="store_true"); args = ap.parse_args()
    exlib.reset_results(RES)
    checks = exlib.Checks()
    S: dict = {"provenance": exlib.provenance(), "literature": LIT, "ecutwfc": ECUT, "ecutrho": ECUTRHO, "kmesh": "6x6x6 shifted"}
    t0 = time.time()
    exlib.check_upf(PSEUDO)
    n_ab = exlib.apple_bottom_symbols(exlib.QE_REF)
    checks.add("reference pw.x is apple-bottom-free (nm)", n_ab == 0, f"{n_ab} ab symbols in {exlib.QE_REF}", "no _ab_* exports")

    # ---------------------------------------------------------------- 1. E(V) on the reference binary
    stage("1. Equation of state: 9 SCF points on the reference pw.x")
    E_ref, V = [], []
    for a in A_SCAN:
        d = RES / "eos" / f"a_{a:.3f}"; prep(d)
        (d / "scf.in").write_text(qe_input("scf", a, K_SCF))
        rc, wall, text = exlib.run([exlib.QE_REF, "-in", "scf.in"], d, d / "scf.out", env={"OMP_NUM_THREADS": "1"}, label=f"eos a={a:.3f} reference")
        info = exlib.parse_pw(text)
        E_ref.append(info.get("total_energy_Ry", np.nan)); V.append(a ** 3 / 4)
        print(f"  a = {a:.3f} bohr  V = {a**3/4:8.3f} bohr^3  E = {info.get('total_energy_Ry')} Ry  ({wall:.1f} s, {info.get('scf_iterations')} it)")
    E_ref, V = np.array(E_ref), np.array(V)
    checks.add("all EOS points converged", bool(np.all(np.isfinite(E_ref))), f"{np.isfinite(E_ref).sum()}/9", "9 energies")
    i_min = int(np.argmin(E_ref))
    checks.add("energy minimum inside the scan", 0 < i_min < len(E_ref) - 1, f"index {i_min} (a = {A_SCAN[i_min]:.3f})", "not at an edge")
    eos = fit_eos(V, E_ref)
    print(f"  Birch-Murnaghan: a0 = {eos['a0_A']:.4f} A, B0 = {eos['B0_GPa']:.1f} GPa, B0' = {eos['Bp']:.2f}, fit rms {eos['rms_Ry']:.2e} Ry")
    checks.add("fit residual", eos["rms_Ry"] < 2e-5, f"{eos['rms_Ry']:.1e} Ry", "rms < 2e-5 Ry (smooth E(V))")
    checks.add("PBE lattice constant vs literature", abs(eos["a0_A"] - LIT["a0_pbe_A"]) < 0.02, f"{eos['a0_A']:.4f} A vs 5.469",
               "|da0| < 0.02 A (HTB 2009 PBE; pseudopotential spread ~0.01 A)")
    checks.add("PBE bulk modulus vs literature", abs(eos["B0_GPa"] - LIT["B0_pbe_GPa"]) < 7, f"{eos['B0_GPa']:.1f} GPa vs 89.2 (Csonka 2009) / 88.5 (AE Delta-test)",
               "|dB0| < 7 GPa")
    checks.add("B0' in the usual range", 3.5 < eos["Bp"] < 5.0, f"{eos['Bp']:.2f}", "3.5 < B0' < 5.0")
    checks.add("PBE overestimates the lattice constant (known)", eos["a0_A"] > LIT["a0_exp_A"], f"{eos['a0_A']:.4f} > 5.431", "a0_PBE > a0_exp")
    S["eos"] = {"a_bohr": A_SCAN, "V_bohr3": V, "E_ref_Ry": E_ref, "fit_ref": eos}

    # ---------------------------------------------------------------- 2. apple-bottom parity along the curve
    if not args.skip_ab and (exlib.AB_FROZEN / "pw.x").exists():
        stage("2. The same 9 SCFs on the frozen apple-bottom pw.x, AB_MODE=gpu (all GEMMs forced to the GPU)")
        E_ab, gpu_calls, gpu_disp = [], 0, 0
        for a in A_SCAN:
            d = RES / "eos_ab" / f"a_{a:.3f}"; prep(d)
            (d / "scf.in").write_text(qe_input("scf", a, K_SCF))
            env = {"AB_MODE": "gpu", "AB_MIN_GPU_DIM": "0", "AB_PROFILE": "1", "AB_PROFILE_JSON": str(d / "ab_profile.json"), "OMP_NUM_THREADS": "1"}
            rc, wall, text = exlib.run(["mpirun", "-np", "1", exlib.AB_FROZEN / "pw.x", "-in", "scf.in"], d, d / "scf.out", env=env,
                                       label=f"eos a={a:.3f} apple-bottom gpu")
            info = exlib.parse_pw(text); E_ab.append(info.get("total_energy_Ry", np.nan))
            try:
                import json
                for r in json.loads((d / "ab_profile.json").read_text()).get("routines", []):
                    gpu_calls += r.get("calls", 0); gpu_disp += r.get("gpu_dispatch_count", 0)
            except Exception:  # noqa: BLE001
                pass
            print(f"  a = {a:.3f}: E_ab = {E_ab[-1]} Ry  dE vs reference = {E_ab[-1] - E_ref[len(E_ab)-1]:+.2e} Ry ({wall:.1f} s)")
        E_ab = np.array(E_ab)
        dE = np.abs(E_ab - E_ref)
        checks.add("apple-bottom gpu energies match the reference binary at every volume", bool(np.all(dE < 1e-7)),
                   f"max |dE| = {np.nanmax(dE):.1e} Ry", "< 1e-7 Ry at all 9 points (printed precision 1e-8; QE 7.5 + apple-bottom vs 7.4.1 OpenBLAS)")
        eos_ab = fit_eos(V, E_ab)
        checks.add("apple-bottom E(V) gives the same a0 and B0", abs(eos_ab["a0_A"] - eos["a0_A"]) < 1e-4 and abs(eos_ab["B0_GPa"] - eos["B0_GPa"]) < 0.1,
                   f"a0 {eos_ab['a0_A']:.5f} vs {eos['a0_A']:.5f} A; B0 {eos_ab['B0_GPa']:.2f} vs {eos['B0_GPa']:.2f} GPa", "|da0| < 1e-4 A, |dB0| < 0.1 GPa")
        checks.add("profiler: every BLAS call ran on the GPU", gpu_calls > 0 and gpu_disp == gpu_calls, f"{gpu_disp}/{gpu_calls}", "gpu_dispatch_count == calls (frozen library's report)")
        S["eos"]["E_ab_Ry"] = E_ab; S["eos"]["fit_ab"] = eos_ab; S["eos"]["ab_gpu_dispatch"] = [gpu_disp, gpu_calls]

    # ---------------------------------------------------------------- 3. bands at the experimental lattice constant
    stage("3. Band energies at a_exp: SCF then a bands run on Gamma, L and the Delta line")
    d = RES / "bands"; prep(d)
    (d / "scf.in").write_text(qe_input("scf", A_EXP_BOHR, K_SCF))
    rc, wall, text = exlib.run([exlib.QE_REF, "-in", "scf.in"], d, d / "scf.out", label="bands scf a_exp")
    (d / "bands.in").write_text(qe_input("bands", A_EXP_BOHR, K_BANDS, nbnd=8))
    rc, wall, text = exlib.run([exlib.QE_REF, "-in", "bands.in"], d, d / "bands.out", label="bands run a_exp")
    kb = parse_bands(text)
    checks.add("bands run produced eigenvalues at all k", len(kb) == 2 + len(DELTA) and all(len(v) >= 8 for _, v in kb), f"{len(kb)} k-points", f"{2 + len(DELTA)} k-points x 8 bands")
    checks.add("all eigenvalues converged", "eigenvalues not converged" not in text, "", "no c_bands warning in bands.out")
    ev = {tuple(np.round(k, 3)): v for k, v in kb}
    needed = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)] + [(float(x), 0.0, 0.0) for x in DELTA]
    missing = [k for k in needed if k not in ev]
    checks.add("every requested k-point is in the output", not missing, f"missing {missing[:3]}", "Gamma, L and 13 Delta points")
    if missing:
        raise SystemExit("bands output incomplete; see results/bands/bands.out")
    G, Lp = ev[(0.0, 0.0, 0.0)], ev[(0.5, 0.5, 0.5)]
    X = ev[(1.0, 0.0, 0.0)]
    checks.add("X-point pairs are degenerate (bands 5 and 6)", abs(X[4] - X[5]) < 2e-3, f"split {abs(X[4] - X[5])*1e3:.2f} meV",
               "< 2 meV (X1 is doubly degenerate; a wrong k convention breaks this)")
    vbm = G[3]                                   # Gamma25' (threefold at Gamma: bands 2-4)
    delta_cb = [(x, ev[(x, 0.0, 0.0)][4]) for x in DELTA]
    x_min, e_min = min(delta_cb, key=lambda t: t[1])
    gap_ind, gap_dir, l1c = e_min - vbm, G[4] - vbm, Lp[4] - vbm
    print(f"  VBM (Gamma25') = {vbm:.3f} eV; CBM on Delta at x = {x_min:.3f} of Gamma-X: {e_min:.3f} eV")
    print(f"  indirect gap {gap_ind:.3f} eV; direct Gamma gap {gap_dir:.3f} eV; L1c - VBM {l1c:.3f} eV; X1c - VBM {ev[(1.0, 0.0, 0.0)][4] - vbm:.3f} eV")
    checks.add("Gamma25' is threefold degenerate", abs(G[1] - G[3]) < 2e-3, f"spread {G[3] - G[1]:.4f} eV", "bands 2-4 within 2 meV")
    checks.add("indirect PBE gap vs known value", 0.50 < gap_ind < 0.72, f"{gap_ind:.3f} eV", "0.50-0.72 eV (PBE ~0.61 at a_exp; expt 1.17)")
    checks.add("CBM sits on the Delta line near 0.85 X", 0.78 <= x_min <= 0.92, f"x = {x_min:.3f}", "0.78-0.92 of Gamma-X")
    checks.add("direct gap at Gamma vs known value", 2.45 < gap_dir < 2.67, f"{gap_dir:.3f} eV", "2.45-2.67 eV (PBE ~2.56; expt 3.4)")
    checks.report("L1c above the VBM (informational)", f"{l1c:.3f} eV", "PBE ~1.5 eV; expt ~2.0 eV")
    S["bands"] = {"a_bohr": A_EXP_BOHR, "vbm_eV": vbm, "cbm_x": x_min, "gap_indirect_eV": gap_ind, "gap_direct_gamma_eV": gap_dir,
                  "L1c_eV": l1c, "X1c_eV": ev[(1.0, 0.0, 0.0)][4] - vbm, "delta_line": delta_cb, "gamma": G, "L": Lp}

    # ---------------------------------------------------------------- 4. plot + results
    stage("4. Plot and RESULTS.md")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        Vf = np.linspace(V.min(), V.max(), 200)
        ax[0].plot(V, E_ref - eos["E0_Ry"], "ko", label="reference pw.x")
        if "E_ab_Ry" in S["eos"]:
            ax[0].plot(V, S["eos"]["E_ab_Ry"] - eos["E0_Ry"], "r+", ms=10, label="apple-bottom gpu")
        ax[0].plot(Vf, birch_murnaghan(Vf, eos["E0_Ry"], eos["V0_bohr3"], eos["B0_GPa"] / RY_BOHR3_GPA, eos["Bp"]) - eos["E0_Ry"], "C0", label="Birch-Murnaghan")
        ax[0].axvline((LIT["a0_pbe_A"] / BOHR_A) ** 3 / 4, color="C2", ls="--", label="literature PBE a0")
        ax[0].axvline((LIT["a0_exp_A"] / BOHR_A) ** 3 / 4, color="C3", ls=":", label="experiment a0")
        ax[0].set_xlabel("V (bohr^3)"); ax[0].set_ylabel("E - E0 (Ry)"); ax[0].legend(fontsize=8); ax[0].set_title("Si PBE equation of state")
        xs = [t[0] for t in delta_cb]; ax[1].plot(xs, [t[1] - vbm for t in delta_cb], "C0o-", label="lowest CB on Gamma-X")
        ax[1].axhline(0, color="k", lw=0.8); ax[1].axhline(gap_ind, color="C3", ls="--", label=f"indirect gap {gap_ind:.2f} eV")
        ax[1].set_xlabel("k (2pi/a) along Gamma-X"); ax[1].set_ylabel("E - VBM (eV)"); ax[1].legend(fontsize=8); ax[1].set_title("PBE conduction band along Delta")
        fig.tight_layout(); fig.savefig(RES / "si_eos_bands.png", dpi=140); print(f"  wrote {RES / 'si_eos_bands.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot skipped: {exc}")
    S["checks"] = checks.rows; S["all_pass"] = checks.all_pass; S["total_wall_s"] = time.time() - t0
    exlib.dump_json(RES / "summary.json", S)
    (RES / "RESULTS.md").write_text(results_md(S, checks))
    print(f"\n{'ALL CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'} ({checks.summary()}) in {S['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


def results_md(S, checks):
    p = S["provenance"]; e = S["eos"]["fit_ref"]; b = S["bands"]
    L = [f"# Example 4 results: Si equation of state and bands ({p['date']})", "",
         f"rainbow-connection {p['rainbow_connection_commit']}, {p['hardware']}, reference pw.x {p['pw_x_reference']} ({p['pw_x_reference_ab_symbols']} apple-bottom symbols), "
         f"ecutwfc {S['ecutwfc']} / ecutrho {S['ecutrho']} Ry, {S['kmesh']}, load1 {p['load1_at_start']:.2f}", "", checks.summary(), "", "## Checks", "", checks.markdown(), "",
         "## Numbers", "", "| quantity | this run | literature PBE | experiment |", "|---|---|---|---|",
         f"| a0 (A) | {e['a0_A']:.4f} | 5.469 (HTB 2009); 5.4695 (AE Delta-test) | 5.431 |", f"| B0 (GPa) | {e['B0_GPa']:.1f} | 89.2 (Csonka 2009); 88.5 (AE Delta-test) | 99.2 |", f"| B0' | {e['Bp']:.2f} | 4.31 (AE Delta-test) | ~4.1 |",
         f"| indirect gap at a_exp (eV) | {b['gap_indirect_eV']:.3f} | ~0.61 | 1.17 |", f"| CBM position on Gamma-X | {b['cbm_x']:.3f} | ~0.85 | ~0.85 |",
         f"| direct gap at Gamma (eV) | {b['gap_direct_gamma_eV']:.3f} | ~2.56 | 3.4 |", f"| L1c - VBM (eV) | {b['L1c_eV']:.3f} | ~1.5 | ~2.0 |", ""]
    if "fit_ab" in S["eos"]:
        a = S["eos"]["fit_ab"]
        L += [f"apple-bottom (frozen pw.x, AB_MODE=gpu, every GEMM on the GPU: {S['eos']['ab_gpu_dispatch'][0]}/{S['eos']['ab_gpu_dispatch'][1]} dispatches): "
              f"a0 = {a['a0_A']:.4f} A, B0 = {a['B0_GPa']:.1f} GPa, max |dE| vs reference = {float(np.nanmax(np.abs(np.array(S['eos']['E_ab_Ry']) - np.array(S['eos']['E_ref_Ry'])))):.1e} Ry.", ""]
    L += ["Literature: Haas, Tran, Blaha, PRB 79, 085104 (2009) for a0; Csonka et al., PRB 79, 155107 (2009), Table V for B0 (PBE and experiment); Lejaeghere et al., Science 351, aad3000 (2016) WIEN2k reference for the all-electron a0/B0/B0'; PBE gaps are the standard values at the experimental lattice constant.",
          "", "## Files", "", "- results/eos/a_*/scf.out, results/eos_ab/a_*/{scf.out,ab_profile.json}, results/bands/bands.out, results/summary.json, results/si_eos_bands.png"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())

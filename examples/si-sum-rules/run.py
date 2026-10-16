#!/usr/bin/env python
"""Example 5: Yambo IP-RPA optics of silicon against exact sum rules and the plasmon.

Yambo is run with the retarded Green's function (GrFnTpXd="R"): the default T-ordered form has
eps2(0) != 0 at finite broadening, which breaks the Kramers-Kronig static limit by an eta-proportional
amount. Yambo includes the non-local pseudopotential commutator [r,Vnl] in the velocity operator
whenever p2y wrote SAVE/ns.kb_pp_pwscf (it does here). With a non-local potential the f-sum is no
longer N: the Thomas-Reiche-Kuhn derivation uses [x, H] = i hbar p/m, which fails, and the exact sum
with the full velocity equals N plus a double-commutator term of the non-local part (Yabana and Bertsch,
"Oscillator strengths with pseudopotentials", arXiv:physics/9802017 (1998): the sum does not count the
valence electrons, and the correction can be large); the p/m-only sum obeys no rule at all. A control run
on a SAVE copy without the kb_pp files measures both numbers.

  QE SCF + NSCF (80 bands) with a norm-conserving PBE pseudopotential (SG15 ONCV; no USPP question for Yambo)
    -> p2y -> yambo setup -> yambo IP-RPA eps(w) on 0-40 eV with 20, 40 and 80 bands
    -> f-sum rule:        (2/pi) Int_0^Wmax w eps2(w) dw  ->  (hbar wp)^2,  hbar wp = sqrt(4 pi n) Ha
    -> KK static limit:   eps1(0) - 1 = (2/pi) Int eps2(w)/w dw   (T-ordered IP output is causal)
    -> energy-loss:       -Im 1/eps peaks at the plasmon (~16.7 eV for Si); eel file == -Im(1/eps) from eps
    -> Yambo's reported direct gap equals QE's Gamma gap

References: the f-sum and Kramers-Kronig relations are exact; hbar wp(Si, 8 valence e-/cell) = 16.6 eV;
measured Si bulk plasmon 16.7 eV (RPA gives 16-17 eV). The partial f-sum with a finite band set and
window must increase monotonically with the band count and stay below the exact value; without the
non-local pseudopotential commutator ([r,Vnl] off, Yambo's default) it saturates somewhat below 1.

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py   (~2 min)
"""

from __future__ import annotations

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
PSEUDO = Path(os.environ.get("MUPPET_SI_NC_PSEUDO", Path.home() / "qe-test/q-e-qe-7.4.1/QEHeat/examples/pseudo/Si_ONCV_PBE-1.1.upf"))
A_BOHR = 10.263                                  # experimental lattice constant
ECUT = 50.0
NBND = 80
EMAX, STEPS, BROAD = 40.0, 2000, 0.10
BANDS = (10, 20, 40, 80)
HA_EV = 27.211386245988
WP_EXP_EV = 16.7

QE_IN = """&CONTROL
  calculation = '{calc}'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = {a}
  nat = 2
  ntyp = 1
  ecutwfc = {ecut}
  force_symmorphic = .true.{nbnd}
/
&ELECTRONS
  conv_thr = 1.0d-9{diago}
/
ATOMIC_SPECIES
  Si 28.086 {pseudo}
ATOMIC_POSITIONS crystal
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4 0 0 0
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
  0.00000 | {emax:.5f} | eV
%
% DmRngeXd
 {broad:.5f} | {broad:.5f} | eV
%
ETStpsXd= {steps}
GrFnTpXd= "R"
% LongDrXd
 1.000000 | 0.000000 | 0.000000 |
%
"""


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def trap(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def main() -> int:
    exlib.reset_results(RES)
    checks = exlib.Checks()
    S: dict = {"provenance": exlib.provenance(), "pseudo": str(PSEUDO), "a_bohr": A_BOHR, "ecut": ECUT, "nbnd": NBND, "emax": EMAX}
    t0 = time.time()
    exlib.check_upf(PSEUDO)
    checks.add("norm-conserving pseudopotential", 'pseudo_type="NC"' in PSEUDO.read_text(errors="ignore")[:4000], PSEUDO.name, "UPF header says NC")

    # ---------------------------------------------------------------- 1. QE
    stage("1. QE SCF + NSCF (80 bands) with the NC ONCV PBE pseudo on a Gamma-centred 4x4x4 mesh")
    qe = RES / "qe"; (qe / "pseudo").mkdir(parents=True); os.symlink(PSEUDO, qe / "pseudo" / PSEUDO.name)
    (qe / "scf.in").write_text(QE_IN.format(calc="scf", a=A_BOHR, ecut=ECUT, nbnd="", diago="", pseudo=PSEUDO.name))
    rc, wall, text = exlib.run([exlib.QE_REF, "-in", "scf.in"], qe, qe / "scf.out", label="si nc scf")
    scf = exlib.parse_pw(text)
    (qe / "nscf.in").write_text(QE_IN.format(calc="nscf", a=A_BOHR, ecut=ECUT, nbnd=f"\n  nbnd = {NBND}", diago="\n  diago_full_acc = .true.", pseudo=PSEUDO.name))
    rc, wall, text = exlib.run([exlib.QE_REF, "-in", "nscf.in"], qe, qe / "nscf.out", label="si nc nscf 80 bands")
    nscf_text = text
    nscf = exlib.parse_pw(text)
    checks.add("QE SCF and NSCF done", scf.get("job_done") and nscf.get("job_done"), f"E = {scf.get('total_energy_Ry')} Ry, gap on mesh {nscf.get('gap_eV')} eV", "JOB DONE twice")
    m = re.search(r"number of electrons\s+=\s+([\d.]+)", text); nelec = float(m.group(1)) if m else 8.0
    vol = A_BOHR ** 3 / 4
    wp_ev = np.sqrt(4 * np.pi * nelec / vol) * HA_EV
    print(f"  {nelec:.0f} valence electrons in {vol:.2f} bohr^3 -> hbar wp = {wp_ev:.2f} eV (expt plasmon {WP_EXP_EV} eV)")
    S["wp_eV"] = wp_ev; S["nelec"] = nelec

    # ---------------------------------------------------------------- 2. p2y + setup
    stage("2. p2y and yambo setup")
    y = RES / "yambo"; y.mkdir()
    rc, wall, text = exlib.run([exlib.YAMBO_BIN / "p2y"], qe / "tmp" / "si.save", y / "p2y.log", label="p2y si nc")
    checks.add("p2y wrote SAVE", (qe / "tmp/si.save/SAVE").is_dir(), rc, "SAVE exists")
    l_p2y = qe / "tmp/si.save/l_p2y"
    rep_p2y = l_p2y.read_text(errors="ignore") if l_p2y.exists() else ""
    if l_p2y.exists():
        shutil.copy(l_p2y, y / "l_p2y")
    checks.add("p2y report found and free of pseudopotential complaints", bool(rep_p2y) and "not supported" not in rep_p2y.lower(),
               f"{len(rep_p2y.splitlines())} report lines", "l_p2y exists; no 'not supported'")
    shutil.move(str(qe / "tmp/si.save/SAVE"), y / "SAVE")
    exlib.run([exlib.YAMBO_BIN / "yambo"], y, y / "setup.log", label="yambo setup si nc")
    rep = (y / "r_setup").read_text() if (y / "r_setup").exists() else ""
    m = re.search(r"\[X\] Direct Gap\s+:\s+([\d.]+)", rep); gap_y = float(m.group(1)) if m else float("nan")
    m = re.search(r"\[X\] Indirect Gap\s+:\s+([\d.]+)", rep); gap_y_ind = float(m.group(1)) if m else float("nan")
    # QE Gamma eigenvalues (first k-point of the NSCF output) for the direct-gap comparison
    kblk = re.search(r"k =\s*0\.0000\s*0\.0000\s*0\.0000.*?bands \(ev\):\s*\n((?:[-\d.\s]+\n)+)", nscf_text)
    g_ev = [float(v) for v in re.findall(r"-?\d+\.\d+", kblk.group(1))] if kblk else []
    qe_gamma_gap = g_ev[4] - g_ev[3] if len(g_ev) >= 5 else float("nan")

    # ---------------------------------------------------------------- 3. optics for three band counts
    stage(f"3. IP-RPA eps(w) on 0-{EMAX:g} eV with {BANDS} bands")
    runs = {}
    for nb in BANDS:
        (y / f"ip{nb}.in").write_text(YAMBO_IP.format(nb=nb, emax=EMAX, broad=BROAD, steps=STEPS))
        rc, wall, text = exlib.run([exlib.YAMBO_BIN / "yambo", "-F", f"ip{nb}.in", "-J", f"ip{nb}"], y, y / f"yambo_ip{nb}.log", label=f"yambo ip {nb} bands")
        f = sorted(y.glob(f"o-ip{nb}.eps_q1*"))
        if not f:
            checks.add(f"yambo wrote eps for {nb} bands", False, rc, "o-ip*.eps_q1_* exists"); continue
        eps = exlib.read_yambo_eps(f[0]); E, e1, e2 = eps["energy_eV"], eps["eps1"], eps["eps2"]
        gm = re.search(r"Green`s Function\s*:\s*(\S+)", f[0].read_text(errors="ignore")[:6000]); green = gm.group(1) if gm else "?"
        pos = E > 0
        fsum = 2 / np.pi * trap(E[pos] * e2[pos], E[pos])
        kk = 2 / np.pi * trap(e2[pos] / E[pos], E[pos])
        loss = e2 / (e1 ** 2 + e2 ** 2)
        ip = int(np.argmax(loss[E > 5])) ; e_pl = float(E[E > 5][ip])
        runs[nb] = {"file": str(f[0]), "eps1_0": float(e1[0]), "fsum_eV2": fsum, "fsum_fraction": fsum / wp_ev ** 2, "kk_rhs": kk,
                    "kk_lhs": float(e1[0] - 1), "plasmon_eV": e_pl, "wall_s": wall, "causal_header": not eps["resonant_only"], "green": green}
        print(f"  {nb:3d} bands ({wall:.0f} s): eps1(0) = {e1[0]:.3f}; f-sum = {fsum:.1f} eV^2 = {fsum / wp_ev**2:.3f} of wp^2; "
              f"KK: eps1(0)-1 = {e1[0]-1:.3f} vs (2/pi)Int eps2/w = {kk:.3f}; -Im(1/eps) peak {e_pl:.2f} eV")
    S["runs"] = runs
    fr = [runs[nb]["fsum_fraction"] for nb in BANDS if nb in runs]

    # ---------------------------------------------------------------- 3b. control: 80 bands without [r,Vnl]
    # Yambo adds the non-local pseudopotential commutator to the dipoles whenever SAVE/ns.kb_pp_pwscf exists. In a copy
    # of SAVE without those files the velocity operator is p/m alone, for which the Thomas-Reiche-Kuhn sum is exactly N.
    stage("3b. Control: 80 bands with the commutator [r,Vnl] excluded (SAVE copy without ns.kb_pp_pwscf)")
    yc = RES / "yambo_novnl"; yc.mkdir()
    shutil.copytree(y / "SAVE", yc / "SAVE", ignore=shutil.ignore_patterns("ns.kb_pp_pwscf*"))
    exlib.run([exlib.YAMBO_BIN / "yambo"], yc, yc / "setup.log", label="yambo setup si nc (no kb_pp)")
    nbc = max(BANDS)
    (yc / f"ip{nbc}.in").write_text(YAMBO_IP.format(nb=nbc, emax=EMAX, broad=BROAD, steps=STEPS))
    rc, wall, text = exlib.run([exlib.YAMBO_BIN / "yambo", "-F", f"ip{nbc}.in", "-J", f"ip{nbc}"], yc, yc / f"yambo_ip{nbc}.log", label=f"yambo ip {nbc} bands no [r,Vnl]")
    fc = sorted(yc.glob(f"o-ip{nbc}.eps_q1*"))
    rep_c = (yc / f"r-ip{nbc}_optics_chi")
    rvnl_c = re.search(r"\[r,Vnl\] included\s*:\s*(\w+)", rep_c.read_text(errors="ignore")) if rep_c.exists() else None
    S["r_vnl_control"] = rvnl_c.group(1) if rvnl_c else "unknown"
    if fc:
        eps = exlib.read_yambo_eps(fc[0]); E, e1, e2 = eps["energy_eV"], eps["eps1"], eps["eps2"]; pos = E > 0
        fsum_c = 2 / np.pi * trap(E[pos] * e2[pos], E[pos]); frac_c = fsum_c / wp_ev ** 2
        kk_c = 2 / np.pi * trap(e2[pos] / E[pos], E[pos])
        S["control_novnl"] = {"file": str(fc[0]), "eps1_0": float(e1[0]), "fsum_fraction": frac_c, "kk_lhs": float(e1[0] - 1), "kk_rhs": kk_c}
        print(f"  {nbc} bands, [r,Vnl] {S['r_vnl_control']}: eps1(0) = {e1[0]:.3f}; f-sum = {frac_c:.3f} of wp^2 (with commutator: {runs[nbc]['fsum_fraction']:.3f})")
        checks.add("control run excluded the commutator", S["r_vnl_control"] == "no", f"[r,Vnl] included: {S['r_vnl_control']}", "report says no")
        checks.report("f-sum of the p/m-only control (informational: no sum rule holds for p/m with a non-local potential)",
                      f"{frac_c:.3f} of wp^2; eps1(0) = {e1[0]:.3f}", "reported; [x, H] = i hbar p/m fails for a non-local H, so this number obeys no sum rule")
        checks.add("the commutator changes the oscillator strength by more than a few per cent", abs(runs[nbc]["fsum_fraction"] - frac_c) > 0.02,
                   f"with [r,Vnl] {runs[nbc]['fsum_fraction']:.3f} vs p/m only {frac_c:.3f} of wp^2 ({(runs[nbc]['fsum_fraction'] / frac_c - 1) * 100:+.1f} %)",
                   "|difference| > 0.02 (Read and Needs 1991: the non-local term is a 10-20 % effect on Si matrix elements)")
    else:
        checks.add("control run wrote eps", False, rc, "o-ip*.eps_q1_* exists")
    frac_c = S.get("control_novnl", {}).get("fsum_fraction", float("nan"))
    checks.add("partial f-sum increases with the number of bands", len(fr) == len(BANDS) and all(b >= a - 1e-6 for a, b in zip(fr, fr[1:])) and fr[1] > fr[0] and fr[2] > fr[1],
               [round(x, 4) for x in fr], "non-decreasing; strictly increasing while added bands fall inside the window")
    checks.add("f-sum with the full velocity (p/m + i[Vnl, r]) is of order the exact value", bool(fr) and 0.9 < fr[-1] < 1.3, f"{fr[-1]:.3f}" if fr else "-",
               "0.9-1.3 of wp^2: with a non-local potential the sum equals N plus a double-commutator term of Vnl, pseudopotential-specific (Yabana and Bertsch 1998); here +17 % of N")
    checks.add("f-sum fraction at 80 bands and 40 eV", bool(fr) and fr[-1] > 0.60, f"{fr[-1]:.3f}" if fr else "-", "> 0.60 (finite window)")
    r80 = runs.get(80) or runs.get(max(runs))
    if r80:
        kk_rel = abs(r80["kk_lhs"] - r80["kk_rhs"]) / r80["kk_lhs"]
        checks.add("Kramers-Kronig static limit", kk_rel < 0.05, f"eps1(0)-1 = {r80['kk_lhs']:.3f} vs {r80['kk_rhs']:.3f} ({kk_rel*100:.1f}%)", "relative difference < 5% (window 0-40 eV)")
        checks.add("Yambo header is not Resonant-only", r80["causal_header"], f"Green`s Function: {r80['green']}", "Retarded (requested) or T-ordered")
        checks.add("plasmon from -Im(1/eps)", 15.0 < r80["plasmon_eV"] < 19.0, f"{r80['plasmon_eV']:.2f} eV", f"15-19 eV (hbar wp = {wp_ev:.1f}, expt 16.7; IP on an 8-k mesh sits high)")
        # eel file consistency
        eel = sorted(y.glob(f"o-ip{max(runs)}.eel_q1*"))
        if eel:
            rows = np.array([list(map(float, l.split()[:3])) for l in eel[0].read_text().splitlines() if l.strip() and not l.startswith("#")])
            eps = exlib.read_yambo_eps(Path(r80["file"])); loss = eps["eps2"] / (eps["eps1"] ** 2 + eps["eps2"] ** 2)
            n = min(len(rows), len(loss)); d = float(np.abs(rows[:n, 1] - loss[:n]).max() / loss.max())
            checks.add("eel file equals -Im(1/eps) computed from eps", d < 1e-2, f"max diff {d:.1e} of peak", "< 1% of the peak")
            S["eel_consistency"] = d
    checks.add("p2y transferred the eigenvalues: Yambo indirect gap == QE mesh gap", abs(gap_y_ind - (nscf.get("gap_eV") or 0)) < 2e-3,
               f"Yambo {gap_y_ind:.4f} eV vs QE HOMO/LUMO {nscf.get('gap_eV')}", "|dE| < 2 meV")
    checks.add("Yambo direct gap == QE Gamma gap (bands 5-4)", abs(gap_y - qe_gamma_gap) < 2e-3, f"Yambo {gap_y:.4f} eV vs QE {qe_gamma_gap:.4f} eV", "|dE| < 2 meV")
    rvnl = re.search(r"\[r,Vnl\] included\s*:\s*(\w+)", (y / f"r-ip{max(runs)}_optics_chi").read_text(errors="ignore")) if runs else None
    S["r_vnl_included"] = rvnl.group(1) if rvnl else "unknown"
    checks.report("non-local commutator in the dipoles (informational)", f"[r,Vnl] included: {S['r_vnl_included']}", "yes expected when SAVE/ns.kb_pp_pwscf exists")
    S["gap_yambo_eV"] = gap_y; S["gap_yambo_indirect_eV"] = gap_y_ind; S["qe_gamma_gap_eV"] = qe_gamma_gap

    # ---------------------------------------------------------------- 4. plot + results
    stage("4. Plot and RESULTS.md")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for nb in BANDS:
            if nb not in runs: continue
            eps = exlib.read_yambo_eps(Path(runs[nb]["file"])); E, e1, e2 = eps["energy_eV"], eps["eps1"], eps["eps2"]
            ax[0].plot(E, e2, label=f"eps2, {nb} bands"); ax[1].plot(E, e2 / (e1 ** 2 + e2 ** 2), label=f"{nb} bands")
            pos = E > 0; cum = 2 / np.pi * np.concatenate([[0], np.cumsum(0.5 * (E[pos][1:] * e2[pos][1:] + E[pos][:-1] * e2[pos][:-1]) * np.diff(E[pos]))]) / wp_ev ** 2
            ax[2].plot(E[pos], cum, label=f"{nb} bands")
        ax[0].set_xlabel("eV"); ax[0].set_title("Si IP-RPA eps2 (NC ONCV PBE)"); ax[0].legend(fontsize=8)
        ax[1].set_xlabel("eV"); ax[1].set_title("-Im 1/eps (plasmon)"); ax[1].axvline(wp_ev, color="k", ls="--", label=f"hbar wp {wp_ev:.1f} eV"); ax[1].legend(fontsize=8)
        ax[2].set_xlabel("eV"); ax[2].set_title("cumulative f-sum / wp^2"); ax[2].axhline(1, color="k", ls="--"); ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(RES / "si_sum_rules.png", dpi=140); print(f"  wrote {RES / 'si_sum_rules.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot skipped: {exc}")
    S["checks"] = checks.rows; S["all_pass"] = checks.all_pass; S["total_wall_s"] = time.time() - t0
    exlib.dump_json(RES / "summary.json", S)
    L = [f"# Example 5 results: Si sum rules and plasmon ({S['provenance']['date']})", "", f"pseudo {PSEUDO.name}, a = {A_BOHR} bohr, ecutwfc {ECUT} Ry, {NBND} NSCF bands, window 0-{EMAX:g} eV, broadening {BROAD} eV; hbar wp = {wp_ev:.2f} eV", "", checks.summary(), "",
         "## Checks", "", checks.markdown(), "", "## Numbers", "", "| bands | eps1(0) | f-sum / wp^2 | KK: eps1(0)-1 | (2/pi) Int eps2/w | plasmon (eV) |", "|---|---|---|---|---|---|"] + ([f"[r,Vnl] included: {S.get('r_vnl_included')}; Yambo gaps: indirect {S.get('gap_yambo_indirect_eV')} eV, direct {S.get('gap_yambo_eV')} eV (QE Gamma {S.get('qe_gamma_gap_eV')} eV)", f"Control without [r,Vnl] (80 bands): f-sum {S.get('control_novnl', {}).get('fsum_fraction', float('nan')):.3f} of wp^2, eps1(0) = {S.get('control_novnl', {}).get('eps1_0', float('nan')):.3f}"])
    for nb in BANDS:
        if nb in runs:
            r = runs[nb]; L.append(f"| {nb} | {r['eps1_0']:.3f} | {r['fsum_fraction']:.3f} | {r['kk_lhs']:.3f} | {r['kk_rhs']:.3f} | {r['plasmon_eV']:.2f} |")
    L += ["", "## Files", "", "- results/qe/{scf,nscf}.out, results/yambo/o-ip{20,40,80}.eps_q1_ip, results/yambo/r_setup, results/summary.json, results/si_sum_rules.png"]
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"\n{'ALL CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'} ({checks.summary()}) in {S['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Example 7: from the band structure of GaAs to the radiation and Joule heating of a mid-infrared antenna.

  A. QE (pw.x, NC ONCV pseudos, PBE): GaAs SCF, then the conduction band along Gamma-X and Gamma-L near
     Gamma -> electron effective mass m*_PBE and the PBE gap. A two-band Kane correction with the measured
     gap (the band-structure analogue of the pipeline's scissor step) gives m*.
  B. Drude permittivity of n = 1e19 cm^-3 GaAs: plasma frequency from n and m*, damping from the measured
     DC conductivity of Grant's 1000 nm membrane, eps_inf from experiment. Below the screened plasma
     frequency (~33 THz) the semiconductor is a lossy metal, so m* sets where the antenna resonates.
  C. Validation of the material in the FDTD code: MEEP 3D Drude sphere (radius 1 um) against exact Mie
     theory, absorption and scattering, across the Froehlich resonance.
  D. Transmit: MEEP 3D gap-fed GaAs dipole (5 um long, 0.6 um square arms, 0.2 um perfect-conductor feed
     terminals) from 15 to 40 THz: input impedance, radiated power, absorbed power omega Im(E* . D) inside
     the arms, radiation efficiency. Energy conservation P_in = P_rad + P_abs checks the absorbed power.
  E. Receive: the same 5 um GaAs rod (no gap) in an E_z-polarised plane wave from 8 to 20 THz, around its
     length resonance: absorption and scattering (re-radiation) cross-sections, and the heat per unit
     length along the rod, measured twice (box flux and the Yee-grid E . D sum).
  F. Steady 1D conduction along the rod bridging two heat-sunk contacts at its ends, under 1 kW/cm^2 at the
     absorption peak: temperature rise T(z).

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-sphere] [--skip-transmit] [--resolutions 10 15]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "mie-scattering"))
sys.path.insert(0, str(HERE.parent / "dipole-antenna"))
sys.path.insert(0, str(HERE))
import exlib  # noqa: E402
from antenna_theory import ETA0, x_zero_crossing  # noqa: E402
from gaas_theory import (THZ_PER_MEEP, band_mass, drude, eps_drude, fin_temperature, k_path, kane_mass,  # noqa: E402
                         parse_bands, rod_profile)
from mie_theory import mie_q  # noqa: E402

RES = HERE / "results"

# --------------------------------------------------------------------------- inputs and their sources
PSEUDOS = {"Ga": Path.home() / "pseudos/Ga_ONCV_PBE-1.2.upf", "As": Path.home() / "pseudos/As_ONCV_PBE-1.2.upf"}
ALAT_BOHR = 10.6829                 # 5.6533 A, experimental lattice constant (Vurgaftman et al., JAP 89, 5815 (2001))
ECUT_RY = 60.0
EG_EXP_0K, EG_EXP_300K = 1.519, 1.424   # eV, Vurgaftman et al. 2001
M_EXP = 0.067                       # electron mass at Gamma, Vurgaftman et al. 2001 (Table III)
PBE_GAP_REF = 0.5135                # eV, same ONCV pseudos: workflows/gaas-emi-shield/yambo r-gaas_scissors_optics_chi
N_DOPING = 1e25                     # m^-3 (1e19 cm^-3, the membranes' nominal doping)
SIGMA_DC = 8.01e4                   # S/m, measured on the 1000 nm membrane (examples/gaas-emi-membrane/run.py)
EPS_INF = 10.86                     # GaAs high-frequency permittivity (Adachi, JAP 58, R1 (1985))
KAPPA = 46.0                        # W/(m K), GaAs at 300 K (Adachi); heavy doping lowers it, so dT is a lower bound
ANTENNA = {"length": 5.0, "width": 0.6, "gap": 0.2, "pad": 0.2, "fmin": 15.0, "fmax": 40.0}   # um, THz
RECEIVE = {"fmin": 8.0, "fmax": 20.0, "nfreq": 49, "heat_every": 4, "pml_frac": 0.3}      # THz; brackets the 12.5 THz resonance
I0_W_M2 = 1e7                       # illumination for the thermal stage: 1 kW/cm^2 (a loosely focused CO2-laser class beam)
# Gates fixed before the first run (2026-10-14).
GATE_GAP = 0.10            # |E_g,PBE - 0.5135| eV (same pseudos, different k-mesh/cutoff)
GATE_ISO = 0.05            # |m*_GX / m*_GL - 1|: the s-like Gamma6 minimum is isotropic at small k
GATE_KANE = 0.20           # |m*_Kane / 0.067 - 1| at the 0 K gap: two-band Kane ignores spin-orbit and remote bands
GATE_MIE = 0.10            # rel rms of MEEP Q_abs and Q_sca vs Mie for the Drude sphere at 10 px/um
GATE_FROHLICH = 0.03       # relative shift of the Q_abs peak (Froehlich resonance)
GATE_ENERGY = 0.05         # |(P_rad + P_abs) / P_in - 1| at the heat frequencies
GATE_CONV = 0.10           # radiation efficiency at the resonance, coarse vs fine grid
GATE_FIN = 1e-3            # 1D conduction integrator vs the analytic uniform-source fin
# Receive-stage gates, fixed 2026-10-15 before its first 10 and 15 px/um runs (a 5 px/um plumbing scan from 5 to
# 40 THz had located the resonance and shown the two absorbed-power measurements within 1.1 %).
GATE_POYNTING = 0.05       # |P_abs(Yee E . D) / P_abs(box flux) - 1| at the heat frequencies, finest grid
GATE_RX_PEAK = 0.10        # peak sigma_abs, coarse vs fine grid
GATE_RX_FREQ = 0.03        # frequency of the sigma_abs peak, coarse vs fine grid (plus one frequency bin)


def stage(t):
    print(f"\n=== {t} ===", flush=True)


# --------------------------------------------------------------------------- A. QE
def pw_input(calc: str, kpoints: str, nbnd: int | None = None) -> str:
    sysx = f"  nbnd = {nbnd}\n" if nbnd else ""
    return f"""&CONTROL
  calculation = '{calc}'
  prefix = 'gaas'
  outdir = './out'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = {ALAT_BOHR}
  nat = 2
  ntyp = 2
  ecutwfc = {ECUT_RY}
{sysx}/
&ELECTRONS
  conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Ga 69.723 {PSEUDOS['Ga'].name}
  As 74.922 {PSEUDOS['As'].name}
ATOMIC_POSITIONS crystal
  Ga 0.00 0.00 0.00
  As 0.25 0.25 0.25
{kpoints}
"""


def stage_qe(checks: exlib.Checks, S: dict) -> dict | None:
    wd = RES / "qe"
    (wd / "pseudo").mkdir(parents=True, exist_ok=True)
    for path in PSEUDOS.values():
        exlib.check_upf(path)
        link = wd / "pseudo" / path.name
        if not link.exists():
            link.symlink_to(path)
    ref_ab = exlib.apple_bottom_symbols(exlib.QE_REF)
    checks.add("reference pw.x is apple-bottom-free", ref_ab == 0, f"{ref_ab} _ab_* symbols in {exlib.QE_REF}", "0 (nm)")
    (wd / "scf.in").write_text(pw_input("scf", "K_POINTS automatic\n  6 6 6 0 0 0"))
    rc, wall, text = exlib.run([exlib.QE_REF, "-in", "scf.in"], wd, wd / "scf.out", label="pw.x gaas scf")
    scf = exlib.parse_pw(text)
    checks.add("GaAs SCF converged", rc == 0 and scf.get("job_done") and "total_energy_Ry" in scf,
               f"E = {scf.get('total_energy_Ry')} Ry, {wall:.0f} s", "JOB DONE")
    if rc != 0:
        return None
    kp, kmag, labels = k_path()
    (wd / "bands.in").write_text(pw_input("bands", kp, nbnd=14))
    rc, wall, text = exlib.run([exlib.QE_REF, "-in", "bands.in"], wd, wd / "bands.out", label="pw.x gaas bands")
    bands = parse_bands(text)
    ok = rc == 0 and len(bands) == len(kmag) and all(len(b) >= 10 for b in bands)
    checks.add("GaAs bands near Gamma computed", ok, f"{len(bands)} k-points, {wall:.0f} s", f"{len(kmag)} k-points, >= 10 bands")
    if not ok:
        return None
    ev = np.array([b[8] for b in bands])        # band 9: valence band maximum (9 occupied bands with Ga 3d)
    ec = np.array([b[9] for b in bands])        # band 10: conduction band minimum at Gamma
    gap = float(ec[0] - ev[0])
    checks.add("PBE direct gap at Gamma", abs(gap - PBE_GAP_REF) < GATE_GAP, f"{gap:.4f} eV",
               f"{PBE_GAP_REF} +/- {GATE_GAP} eV (same pseudos, workflows/gaas-emi-shield)")
    masses = band_mass(kmag, labels, ec, ALAT_BOHR)
    m_pbe = 0.5 * (masses["X"] + masses["L"])
    iso = masses["X"] / masses["L"]
    checks.add("CBM mass isotropic (Gamma-X vs Gamma-L)", abs(iso - 1) < GATE_ISO,
               f"m*_GX = {masses['X']:.4f}, m*_GL = {masses['L']:.4f}", f"ratio 1 +/- {GATE_ISO}")
    # two-band Kane: 1/m* = 1 + E_P / E_g, so E_P from PBE, then m* at the measured gap
    m_kane_0k, e_p = kane_mass(m_pbe, gap, EG_EXP_0K)
    m_kane_rt, _ = kane_mass(m_pbe, gap, EG_EXP_300K)
    checks.add("Kane-corrected m* vs experiment", abs(m_kane_0k / M_EXP - 1) < GATE_KANE,
               f"m*_PBE = {m_pbe:.4f} -> E_P = {e_p:.1f} eV -> m* = {m_kane_0k:.4f} at 1.519 eV",
               f"{M_EXP} +/- {GATE_KANE:.0%} (Vurgaftman 2001; two-band, no spin-orbit)")
    S["qe"] = {"total_energy_Ry": scf["total_energy_Ry"], "gap_pbe_eV": gap, "m_pbe": m_pbe, "m_gx": masses["X"],
               "m_gl": masses["L"], "E_P_eV": e_p, "m_kane_0K": m_kane_0k, "m_kane_300K": m_kane_rt,
               "k_tpiba": kmag.tolist(), "E_c": ec.tolist(), "E_v": ev.tolist()}
    return S["qe"]


# --------------------------------------------------------------------------- C. Drude sphere
def stage_sphere(d: dict, checks: exlib.Checks, S: dict) -> None:
    out = RES / "sphere"
    xmin, xmax = 2 * np.pi * 22 / THZ_PER_MEEP, 2 * np.pi * 40 / THZ_PER_MEEP     # radius 1 um, 22-40 THz
    cmd = [exlib.MEEP_PYTHON, HERE.parent / "mie-scattering/meep_mie.py", out, "--resolution", 10, "--xmin", xmin,
           "--xmax", xmax, "--nfreq", 60, "--drude-fp", d["fp_meep"], "--drude-gamma", d["gamma_meep"],
           "--drude-eps-inf", d["eps_inf"], "--pml-frac", 0.5, "--air-frac", 0.5]
    rc, wall, _ = exlib.run(cmd, out, out / "meep.log", label="meep drude sphere", timeout=4 * 3600)
    ok = rc == 0 and (out / "spectrum.json").exists()
    checks.add("MEEP Drude sphere completed", ok, f"rc={rc}, {wall:.0f} s", "spectrum.json")
    if not ok:
        return
    sp = json.loads((out / "spectrum.json").read_text())
    f = np.array(sp["freq"])
    x = 2 * np.pi * f
    m = np.sqrt(eps_drude(f, d))
    m = np.where(m.imag < 0, -m, m)
    q = np.array([mie_q(mi, xi) for mi, xi in zip(m, x)])
    q_sca_mie, q_abs_mie = q[:, 1], q[:, 0] - q[:, 1]
    q_sca, q_abs = np.array(sp["qsca"]), np.array(sp["qabs"])
    err_abs = float(np.sqrt(np.mean((q_abs - q_abs_mie) ** 2)) / np.mean(q_abs_mie))
    err_sca = float(np.sqrt(np.mean((q_sca - q_sca_mie) ** 2)) / np.mean(q_sca_mie))
    checks.add("Drude sphere absorption vs Mie", err_abs < GATE_MIE, f"rel rms {err_abs:.4f}", f"< {GATE_MIE} (22-40 THz)")
    checks.add("Drude sphere scattering vs Mie", err_sca < GATE_MIE, f"rel rms {err_sca:.4f}", f"< {GATE_MIE}")
    ff = np.linspace(f.min(), f.max(), 4000)
    mf = np.sqrt(eps_drude(ff, d)); mf = np.where(mf.imag < 0, -mf, mf)
    qf = np.array([mie_q(mi, 2 * np.pi * fi) for mi, fi in zip(mf, ff)])
    f_peak_mie = ff[np.argmax(qf[:, 0] - qf[:, 1])]
    f_peak = f[np.argmax(q_abs)]
    shift = abs(f_peak / f_peak_mie - 1)
    checks.add("Froehlich resonance position", shift < GATE_FROHLICH + float(np.max(np.diff(f))) / f_peak_mie,
               f"MEEP {f_peak * THZ_PER_MEEP:.2f} THz vs Mie {f_peak_mie * THZ_PER_MEEP:.2f} THz", f"< {GATE_FROHLICH:.0%} + 1 bin")
    S["sphere"] = {"rel_rms_abs": err_abs, "rel_rms_sca": err_sca, "f_peak_THz": f_peak * THZ_PER_MEEP,
                   "f_peak_mie_THz": f_peak_mie * THZ_PER_MEEP, "wall_s": sp["wall_s"], "voxels": sp["voxels"],
                   "f_THz": (f * THZ_PER_MEEP).tolist(), "q_abs": q_abs.tolist(), "q_sca": q_sca.tolist(),
                   "f_mie_THz": (ff * THZ_PER_MEEP).tolist(), "q_abs_mie": (qf[:, 0] - qf[:, 1]).tolist(), "q_sca_mie": qf[:, 1].tolist()}


# --------------------------------------------------------------------------- D. antenna
def stage_antenna(d: dict, res: float) -> dict | None:
    out = RES / f"antenna_res{res:g}"
    cmd = [exlib.MEEP_PYTHON, HERE / "meep_gaas_dipole.py", out, "--fp", d["fp_meep"], "--gamma", d["gamma_meep"],
           "--eps-inf", d["eps_inf"], "--length", ANTENNA["length"], "--width", ANTENNA["width"], "--gap", ANTENNA["gap"],
           "--pad", ANTENNA["pad"], "--fmin-thz", ANTENNA["fmin"], "--fmax-thz", ANTENNA["fmax"], "--resolution", res]
    rc, wall, _ = exlib.run(cmd, out, out / "meep.log", label=f"meep gaas dipole res {res:g}", timeout=8 * 3600)
    if rc != 0 or not (out / "antenna.json").exists():
        return None
    return json.loads((out / "antenna.json").read_text())


def analyse_antenna(a: dict) -> dict:
    f = np.array(a["freq_thz"])
    p_in, p_rad = np.array(a["p_in"]), np.array(a["p_rad"])
    z = (np.array(a["z_re"]) + 1j * np.array(a["z_im"])) * ETA0
    hf = np.array([h["f"] for h in a["heat"]]) * THZ_PER_MEEP
    p_abs = np.array([h["p_abs_both_arms"] for h in a["heat"]])
    p_in_h, p_rad_h = np.interp(hf, f, p_in), np.interp(hf, f, p_rad)
    eta = p_rad / p_in
    i_res = int(np.argmax(p_rad / np.max(p_rad) * (z.real > 0)))       # strongest radiation for the fixed source
    x0 = x_zero_crossing(f, z)
    return {"f": f, "z": z, "eta": eta, "heat_f": hf, "balance": (p_rad_h + p_abs) / p_in_h, "p_abs": p_abs,
            "f_res": float(f[i_res]), "eta_res": float(eta[i_res]), "f_x0": x0[0] if x0 else None}


# --------------------------------------------------------------------------- E. receive
def stage_rod(d: dict, res: float) -> dict | None:
    out = RES / f"rod_res{res:g}"
    cmd = [exlib.MEEP_PYTHON, HERE / "meep_gaas_rod.py", out, "--fp", d["fp_meep"], "--gamma", d["gamma_meep"],
           "--eps-inf", d["eps_inf"], "--length", ANTENNA["length"], "--width", ANTENNA["width"],
           "--fmin-thz", RECEIVE["fmin"], "--fmax-thz", RECEIVE["fmax"], "--nfreq", RECEIVE["nfreq"],
           "--heat-every", RECEIVE["heat_every"], "--pml-frac", RECEIVE["pml_frac"], "--resolution", res]
    rc, wall, _ = exlib.run(cmd, out, out / "meep.log", label=f"meep gaas rod res {res:g}", timeout=12 * 3600)
    if rc != 0 or not (out / "rod.json").exists():
        return None
    return json.loads((out / "rod.json").read_text())


def analyse_rod(r: dict) -> dict:
    f = np.array(r["freq_thz"])
    s_abs, s_sca = np.array(r["sigma_abs_um2"]), np.array(r["sigma_sca_um2"])
    i = int(np.argmax(s_abs))
    poynting = np.array([h["p_abs_yee"] / h["p_abs_box"] for h in r["heat"]])
    heat_f = np.array([h["f"] for h in r["heat"]]) * THZ_PER_MEEP
    return {"f": f, "s_abs": s_abs, "s_sca": s_sca, "f_peak": float(f[i]), "s_abs_peak": float(s_abs[i]),
            "s_sca_at_peak": float(s_sca[i]), "poynting": poynting, "heat_f": heat_f,
            "k_heat": int(np.argmin(abs(heat_f - f[i])))}       # the heat map nearest the absorption peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sphere", action="store_true")
    ap.add_argument("--skip-transmit", action="store_true")
    ap.add_argument("--resolutions", type=float, nargs="+", default=[10, 15])
    args = ap.parse_args()
    exlib.reset_results(RES)
    checks = exlib.Checks()
    S: dict = {"provenance": exlib.provenance(), "inputs": {"n_m3": N_DOPING, "sigma_dc": SIGMA_DC, "eps_inf": EPS_INF,
               "kappa": KAPPA, "I0_W_m2": I0_W_M2, "antenna": ANTENNA, "receive": RECEIVE}}

    stage("F. thermal integrator self-check")
    s = np.linspace(0, 2.4e-6, 241)
    t = fin_temperature(s, np.full_like(s, 1.0), 2.0)
    t_exact = 1.0 * s[-1] ** 2 / (2 * 2.0)
    checks.add("fin integrator vs analytic uniform source", abs(t[-1] / t_exact - 1) < GATE_FIN, f"{t[-1] / t_exact:.6f}",
               f"tip dT = q L^2 / (2 kappa A), ratio 1 +/- {GATE_FIN}")

    stage("A. QE: GaAs conduction band near Gamma")
    qe = stage_qe(checks, S)
    if qe is None:
        return finish(checks, S)

    stage("B. Drude permittivity of n = 1e19 cm^-3 GaAs")
    d = S["drude"] = drude(qe["m_kane_300K"], N_DOPING, SIGMA_DC, EPS_INF)
    d_pbe = S["drude_pbe_mass"] = drude(qe["m_pbe"], N_DOPING, SIGMA_DC, EPS_INF)
    print(f"  m* = {d['m_star']:.4f} m_e: screened plasma frequency {d['f_p_screened_THz']:.1f} THz, "
          f"damping {d['gamma_THz']:.2f} THz  (with the bare PBE mass {qe['m_pbe']:.4f}: "
          f"{d_pbe['f_p_screened_THz']:.1f} THz)")

    if not args.skip_sphere:
        stage("C. MEEP Drude sphere vs Mie")
        stage_sphere(d, checks, S)

    runs = {}
    if not args.skip_transmit:
        stage("D. MEEP GaAs dipole antenna, transmitting")
        for r in sorted(args.resolutions):
            a = stage_antenna(d, r)
            checks.add(f"MEEP antenna resolution {r:g} completed", a is not None, "antenna.json" if a else "failed", "rc 0")
            if a:
                runs[r] = (a, analyse_antenna(a))
    if runs:
        fine = max(runs)
        a, an = runs[fine]
        worst = float(np.max(abs(an["balance"] - 1)))
        checks.add("energy conservation P_in = P_rad + P_abs", worst < GATE_ENERGY, f"worst |ratio - 1| = {worst:.3f}",
                   f"< {GATE_ENERGY} at {len(an['heat_f'])} frequencies, {ANTENNA['fmin']:g}-{ANTENNA['fmax']:g} THz")
        if len(runs) > 1:
            coarse = min(runs)
            e_c = float(np.interp(an["f_res"], runs[coarse][1]["f"], runs[coarse][1]["eta"]))
            checks.add("radiation efficiency converged", abs(e_c / an["eta_res"] - 1) < GATE_CONV,
                       f"eta({an['f_res']:.1f} THz) = {e_c:.3f} (res {coarse:g}) vs {an['eta_res']:.3f} (res {fine:g})",
                       f"+/- {GATE_CONV:.0%}")
        g0 = a["gaas_start_um"]
        near = []                                          # share of the heat within 0.3 um of the terminal
        for h in a["heat"]:
            z, q = np.array(h["z"]), np.array(h["q_z"])
            near.append(float(q[(z > g0) & (z < g0 + 0.3)].sum() / q.sum()))
        S["transmit"] = {"resolution": fine, "f_res_THz": an["f_res"], "eta_res": an["eta_res"], "f_X0_THz": an["f_x0"],
                         "heat_f_THz": an["heat_f"].tolist(), "heat_within_0.3um_of_terminal": near,
                         "wall_s": a["wall_s"], "voxels": a["voxels"]}
        print(f"  strongest radiation at {an['f_res']:.1f} THz, radiation efficiency {an['eta_res']:.3f}; X = 0 at "
              f"{an['f_x0'] if an['f_x0'] is None else round(an['f_x0'], 2)} THz; heat within 0.3 um of the terminal: "
              f"{min(near):.0%}-{max(near):.0%}")

    stage("E. MEEP GaAs rod antenna, receiving")
    rods = {}
    for r in sorted(args.resolutions):
        rd = stage_rod(d, r)
        checks.add(f"MEEP rod resolution {r:g} completed", rd is not None, "rod.json" if rd else "failed", "rc 0")
        if rd:
            rods[r] = (rd, analyse_rod(rd))
    if not rods:
        plot(S, runs, rods)
        return finish(checks, S)
    fine = max(rods)
    rd, ra = rods[fine]
    worst = float(np.max(abs(ra["poynting"] - 1)))
    checks.add("absorbed power: Yee E . D sum = box flux", worst < GATE_POYNTING, f"worst |ratio - 1| = {worst:.4f}",
               f"< {GATE_POYNTING} at {len(ra['poynting'])} frequencies, {RECEIVE['fmin']:g}-{RECEIVE['fmax']:g} THz")
    if len(rods) > 1:
        coarse = min(rods)
        rc_ = rods[coarse][1]
        bin_ = float(np.max(np.diff(ra["f"]))) / ra["f_peak"]
        checks.add("absorption peak height converged", abs(rc_["s_abs_peak"] / ra["s_abs_peak"] - 1) < GATE_RX_PEAK,
                   f"{rc_['s_abs_peak']:.2f} um^2 (res {coarse:g}) vs {ra['s_abs_peak']:.2f} um^2 (res {fine:g})", f"+/- {GATE_RX_PEAK:.0%}")
        checks.add("absorption peak frequency converged", abs(rc_["f_peak"] / ra["f_peak"] - 1) < GATE_RX_FREQ + bin_,
                   f"{rc_['f_peak']:.2f} THz (res {coarse:g}) vs {ra['f_peak']:.2f} THz (res {fine:g})",
                   f"< {GATE_RX_FREQ:.0%} + one bin ({bin_:.1%})")
    area_proj = ANTENNA["length"] * ANTENNA["width"]
    print(f"  absorption peak {ra['f_peak']:.2f} THz: sigma_abs = {ra['s_abs_peak']:.2f} um^2 ({ra['s_abs_peak'] / area_proj:.1f}x the "
          f"projected area), sigma_sca = {ra['s_sca_at_peak']:.2f} um^2 (re-radiated)")

    stage("F. Joule heating and temperature along the rod")
    k = ra["k_heat"]
    h = rd["heat"][k]
    s_um, q_w_per_m = rod_profile(h, ANTENNA["length"], I0_W_M2)   # from the end contact (0) to the middle (L/2)
    area = (ANTENNA["width"] * 1e-6) ** 2
    t_rise = fin_temperature(s_um * 1e-6, q_w_per_m, KAPPA * area)
    p_abs_w = h["p_abs_yee"] / h["incident"] * 1e-12 * I0_W_M2
    S["receive"] = {"resolution": fine, "f_peak_THz": ra["f_peak"], "sigma_abs_peak_um2": ra["s_abs_peak"],
                    "sigma_sca_at_peak_um2": ra["s_sca_at_peak"], "heat_freq_THz": float(ra["heat_f"][k]),
                    "p_abs_W": p_abs_w, "p_sca_W": float(np.interp(ra["heat_f"][k], ra["f"], ra["s_sca"])) * 1e-12 * I0_W_M2,
                    "peak_q_W_per_m": float(q_w_per_m.max()), "dT_max_K": float(t_rise.max()),
                    "s_um": s_um.tolist(), "q_W_per_m": q_w_per_m.tolist(), "dT_K": t_rise.tolist(),
                    "wall_s": rd["wall_s"], "voxels": rd["voxels"]}
    print(f"  at {ra['heat_f'][k]:.1f} THz under {I0_W_M2 / 1e7:g} kW/cm^2: the rod absorbs {p_abs_w * 1e6:.3g} uW and re-radiates "
          f"{S['receive']['p_sca_W'] * 1e6:.3g} uW; peak heat {q_w_per_m.max():.3g} W/m, temperature rise at the middle "
          f"{t_rise.max():.3g} K (1D conduction to the end contacts, no substrate)")
    plot(S, runs, rods)
    return finish(checks, S)


def plot(S: dict, runs: dict, rods: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
    if "qe" in S:
        q = S["qe"]
        k = np.array(q["k_tpiba"])
        ax[0, 0].plot(k, np.array(q["E_c"]) - q["E_c"][0], "o", ms=3, label="QE CBM (Γ–X then Γ–L)")
        ax[0, 0].set_xlabel("|k| (2π/a)")
        ax[0, 0].set_ylabel("E − E_c(Γ) (eV)")
        ax[0, 0].set_title(f"GaAs PBE: m* = {q['m_pbe']:.4f}, Kane → {q['m_kane_300K']:.4f} m_e", fontsize=10)
    if "sphere" in S:
        s = S["sphere"]
        ax[0, 1].plot(s["f_mie_THz"], s["q_abs_mie"], "C3-", lw=1, label="Mie Q_abs")
        ax[0, 1].plot(s["f_mie_THz"], s["q_sca_mie"], "C0-", lw=1, label="Mie Q_sca")
        ax[0, 1].plot(s["f_THz"], s["q_abs"], "C3o", ms=3, mfc="none", label="MEEP Q_abs")
        ax[0, 1].plot(s["f_THz"], s["q_sca"], "C0o", ms=3, mfc="none", label="MEEP Q_sca")
        ax[0, 1].set_xlabel("frequency (THz)")
        ax[0, 1].set_title("doped-GaAs sphere, r = 1 µm", fontsize=10)
        ax[0, 1].legend(fontsize=8, frameon=False)
    if runs:
        for r, (a, an) in sorted(runs.items()):
            ax[0, 2].plot(an["f"], an["eta"], label=f"η = P_rad/P_in, {r:g} px/µm")
        a, an = runs[max(runs)]
        ax[0, 2].plot(an["heat_f"], an["p_abs"] / np.interp(an["heat_f"], an["f"], np.array(a["p_in"])), "ks", ms=4,
                      label="P_abs/P_in, ω Im(E*·D)")
        ax[0, 2].set_xlabel("frequency (THz)")
        ax[0, 2].set_ylim(0, 1)
        ax[0, 2].set_title("transmitting, gap-fed: where the input power goes", fontsize=10)
        ax[0, 2].legend(fontsize=8, frameon=False)
    if rods:
        for r, (rd, ra) in sorted(rods.items()):
            ls = "-" if r == max(rods) else ":"
            ax[1, 0].plot(ra["f"], ra["s_abs"], "C3" + ls, label=f"σ_abs, {r:g} px/µm")
            ax[1, 0].plot(ra["f"], ra["s_sca"], "C0" + ls, label=f"σ_sca, {r:g} px/µm")
        ax[1, 0].axhline(ANTENNA["length"] * ANTENNA["width"], color="k", lw=0.8, ls="--", label="projected area")
        ax[1, 0].set_xlabel("frequency (THz)")
        ax[1, 0].set_ylabel("cross-section (µm²)")
        ax[1, 0].set_title("receiving: 5 µm GaAs rod, E ∥ rod", fontsize=10)
        ax[1, 0].legend(fontsize=8, frameon=False)
    if "receive" in S:
        A = S["receive"]
        ax[1, 1].plot(A["s_um"], A["q_W_per_m"], "C3-", label="heat")
        ax2 = ax[1, 1].twinx()
        ax2.plot(A["s_um"], A["dT_K"], "C0-", label="ΔT")
        ax[1, 1].set_xlabel("distance from the end contact (µm); rod middle at 2.5")
        ax[1, 1].set_ylabel("Joule heat per length (W/m)", color="C3")
        ax2.set_ylabel("temperature rise (K)", color="C0")
        ax[1, 1].set_title(f"{A['heat_freq_THz']:.1f} THz, {I0_W_M2 / 1e7:g} kW/cm²", fontsize=10)
    if runs and rods:
        a, _ = runs[max(runs)]
        rd, ra = rods[max(rods)]
        ht = a["heat"][0]                                         # lowest transmit heat frequency
        hr = rd["heat"][ra["k_heat"]]
        zt, qt = np.array(ht["z"]), np.array(ht["q_z"])
        zr, qr = np.array(hr["z"]), np.array(hr["q_z"])
        ax[1, 2].plot(zt, qt / qt.max(), "C1-", label=f"transmitting, {ht['f'] * THZ_PER_MEEP:.0f} THz (terminal at {a['gaas_start_um']:.1f} µm)")
        ax[1, 2].plot(zr, qr / qr.max(), "C3-", label=f"receiving, {hr['f'] * THZ_PER_MEEP:.0f} THz")
        ax[1, 2].set_xlim(0, ANTENNA["length"] / 2 + 0.2)
        ax[1, 2].set_xlabel("z along the rod from its centre (µm)")
        ax[1, 2].set_ylabel("heat per length (normalised)")
        ax[1, 2].set_title("where the heat goes", fontsize=10)
        ax[1, 2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "antenna_heating.png", dpi=150)


def finish(checks: exlib.Checks, S: dict) -> int:
    S["checks"] = checks.rows
    S["all_pass"] = checks.all_pass
    exlib.dump_json(RES / "summary.json", S)
    lines = [f"# Antenna heating results ({time.strftime('%Y-%m-%d %H:%M')})", "", f"All checks pass: **{checks.all_pass}**", "",
             checks.markdown(), ""]
    if "qe" in S:
        q = S["qe"]
        lines.append(f"- QE: PBE gap {q['gap_pbe_eV']:.3f} eV, m*_PBE = {q['m_pbe']:.4f}, E_P = {q['E_P_eV']:.1f} eV, "
                     f"Kane m* = {q['m_kane_0K']:.4f} (0 K gap) / {q['m_kane_300K']:.4f} (300 K gap, used below)")
    if "drude" in S:
        dd = S["drude"]
        lines.append(f"- Drude: screened plasma frequency {dd['f_p_screened_THz']:.1f} THz, damping {dd['gamma'] / 2 / np.pi / 1e12:.2f} THz")
    if "transmit" in S:
        A = S["transmit"]
        lines.append(f"- transmitting (gap-fed, {A['resolution']:g} px/um): strongest radiation at {A['f_res_THz']:.1f} THz with "
                     f"radiation efficiency {A['eta_res']:.3f}; {min(A['heat_within_0.3um_of_terminal']):.0%}-"
                     f"{max(A['heat_within_0.3um_of_terminal']):.0%} of the heat lies within 0.3 um of the feed terminals")
    if "receive" in S:
        A = S["receive"]
        lines.append(f"- receiving ({A['resolution']:g} px/um): absorption peak {A['f_peak_THz']:.2f} THz, sigma_abs = "
                     f"{A['sigma_abs_peak_um2']:.2f} um^2, sigma_sca = {A['sigma_sca_at_peak_um2']:.2f} um^2; under "
                     f"{I0_W_M2 / 1e7:g} kW/cm^2 at {A['heat_freq_THz']:.1f} THz the rod absorbs {A['p_abs_W'] * 1e6:.3g} uW, "
                     f"re-radiates {A['p_sca_W'] * 1e6:.3g} uW, and its middle warms by {A['dT_max_K']:.3g} K "
                     "(1D conduction to heat-sunk end contacts, no substrate: an illustration, not a device prediction)")
    lines += ["", "![antenna heating](antenna_heating.png)", "", "Provenance: see summary.json."]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'ALL PASS' if checks.all_pass else 'FAILURES'} -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

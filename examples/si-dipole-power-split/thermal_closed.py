"""Closed-form heating of the chip under the laser, with input uncertainty. Registered before Elmer runs.

Lumped chip on a chuck: P_abs(k) = P(k) T_train A(t_dev, t_box) erf(sqrt2 a/w) erf(sqrt2 b/w); R = t/(kappa A_c) + gap/(kappa_air A_c)
in parallel with the top loss; C = rho cp t A_c; T(t) from the actual sweep timeline, power stepping at each file's save time.
Membrane drop across the oxide and the beam's spreading resistance are added as separate terms. No RF value is used.
"""
import json

import numpy as np
from scipy.special import erf

import common as c


def absorbed_fraction(spc, t_dev_nm, t_box_um):
    p = spc.params_from_epw()
    out = np.empty(t_dev_nm.size)
    for i, (td, tb) in enumerate(zip(t_dev_nm, t_box_um)):
        p.L_nm, p.t_box_cm = td * 1e-7, tb * 1e-4
        out[i] = sum(spc.thin_film_optics("soi", p))
    return out


def chip_model(n=20000):
    """The sampled inputs and the lumped chip they give: R, C, absorbed fraction, absorbed power on the chip P(k), geometry."""
    spc = c.import_private("si_photocarriers")
    s = c.sample_inputs(n)
    idx = np.arange(0, n, 20)
    afrac = np.resize(absorbed_fraction(spc, s["t_device_nm"][idx], s["t_box_um"][idx]), n)
    chip = c.device()["stack"]["chip_m"] * s["chip_mm"]
    area, t = chip**2, 1e-3 * s["t_handle_mm"]
    kap = 148.0 * s["kappa_si_T"]
    R = 1.0 / (1.0 / (t / (kap * area) + 1e-6 * s["gap_chuck_um"] / (0.026 * area)) + s["h_top_W_m2K"] * area)
    C = 2329.0 * 712.0 * s["cp_si_T"] * t * area
    frac = erf(np.sqrt(2) * 0.5 * chip / (1e-3 * s["beam_w_mm"])) ** 2
    P = lambda k: c.laser_power(np.full(n, float(k)), s["power_amps"]) * s["train_transmission"] * afrac * frac
    return dict(s=s, R=R, C=C, afrac=afrac, frac=frac, P=P, kap=kap, area=area, t=t, gap_R=1e-6 * s["gap_chuck_um"] / (0.026 * area))


def sweep_temperatures(m, tl):
    """Rise at each sweep of the timeline, per sample. The setting of file i holds from file i-1's save to its own."""
    T = np.zeros((len(tl), m["R"].size))
    tau = m["R"] * m["C"]
    for i in range(1, len(tl)):
        p = m["P"](tl[i][1]) if tl[i][1] else 0 * tau
        T[i] = p * m["R"] + (T[i - 1] - p * m["R"]) * np.exp(-(tl[i][2] - tl[i - 1][2]) / tau)
    return T


def main():
    c.require_private_repo()
    m = chip_model()
    s, R, C, P, afrac, frac = m["s"], m["R"], m["C"], m["P"], m["afrac"], m["frac"]
    tau = R * C
    tl = c.x_timeline()
    names = [t[0] for t in tl]
    T = sweep_temperatures(m, tl)
    q = c.percentiles
    # spreading of the beam in the handle, above the lumped mean
    R_spread = 1.0 / (np.sqrt(2 * np.pi) * m["kap"] * 1e-3 * s["beam_w_mm"])
    I7 = 2 * P(7) / (afrac * np.pi * (1e-3 * s["beam_w_mm"]) ** 2)          # W/m^2 incident peak at setting 7
    drop_box = I7 * c.device()["stack"]["membrane_absorbed"] * 1e-6 * s["t_box_um"] / s["kappa_sio2"]

    # one-at-a-time decomposition of the spread of the steady rise at k = 7, in ln units
    terms = {"gap_chuck_um": m["gap_R"], "beam_w_mm": frac, "power_law": c.laser_power(np.full(R.size, 7.0), s["power_amps"]),
             "absorbed_fraction (fringe)": afrac, "train_transmission": s["train_transmission"], "chip_mm": m["area"], "t_handle_mm": m["t"]}
    budget = {k: float(np.std(np.log(v))) for k, v in terms.items()}

    res = {
        "what": "closed-form chip heating; prediction registered before the Elmer runs",
        "R_th_K_W": q(R), "C_th_J_K": q(C), "tau_th_s": q(tau),
        "P_abs_on_chip_W": {str(k): q(P(k)) for k in (3, 5, 7)},
        "dT_K_steady": {str(k): q(P(k) * R) for k in (3, 5, 7)},
        "dT_K_at_sweep": {names[i]: q(T[i]) for i in range(1, len(tl))},
        "timeline_s": {names[i]: tl[i][2] for i in range(len(tl))},
        "dT_spreading_K_at_7": q(P(7) * R_spread),
        "dT_box_drop_K_at_7": q(drop_box),
        "ln_spread_budget_of_dT7": dict(sorted(budget.items(), key=lambda kv: -kv[1])),
        "absorbed_fraction": q(afrac),
    }
    c.register("01_thermal_closed_form", res, code=[__file__, c.HERE / "common.py"])
    print(json.dumps({k: res[k] for k in ("R_th_K_W", "tau_th_s", "dT_K_steady", "ln_spread_budget_of_dT7")}, indent=0)[:1200])


if __name__ == "__main__":
    main()

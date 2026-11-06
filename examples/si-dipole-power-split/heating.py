"""What heating predicts in the existing data, from the dark fields, the materials and the thermal model. Registered before any comparison (prereg/03_fingerprints.json).

H_T: dY = sigma S dln(sigma)/dT dT (first order in sigma, uniform handle), checked against the resistivity scan of full runs.
H_TC: dY_k = dY_3 (P_k/P_3) g(T_k)/g(T_3), g = (mu_n + mu_p)(T) tau(T); anchored on the measured setting 3 and on magnitude only. That is the most favourable version for heating: the
anchor uses the measured size and the magnitude-only test forgives the phase change between settings. Deviation from the complex test in 00_PREREG, in heating's favour.
Input samples (no RF value) are the ones from common.sample_inputs; the steady rise is the closed form, whose agreement with Elmer is gated separately.
"""
import json

import numpy as np

import common as c
import thermal_closed as tc

SETTINGS = tuple(c.device()["settings"])


def load_dark():
    """{rho: {f_GHz: dict(Y, dY_dlnsigma, ...)}} from the dark full-wave runs that exist."""
    out = {}
    for rho in (1, 3, 10, 30, 100):
        p = c.OUT / "fullwave" / f"ps_dark_r{rho}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        out[rho] = {}
        for f, r in d["freqs"].items():
            sig = 100.0 / rho
            z = np.array(r["sens_handle"]["z_m"])
            s = np.array(r["sens_handle"]["by_z"]["re"]) + 1j * np.array(r["sens_handle"]["by_z"]["im"])
            top = -c.device()["stack"]["t_box_mm"] * 1e-3
            k = z <= top + 1e-9                       # raw edges: the plane at the oxide/handle face belongs to the handle's dual volume
            depth = np.maximum(top - z[k], 0.0)
            sk = s[k]
            near = depth < 150e-6                     # the photocarriers' depth, ~L_D, for the frequency fingerprint of H_TC
            out[rho][float(f)] = {"Y": complex(*r["Y_S"]), "dY_dlnsigma": complex(sig * sk.sum()), "dY_dlnsigma_top": complex(sig * sk[near].sum()),
                                  "P_rad": r["P_rad_A"] / r["P_in"], "P_handle": r["P_handle"] / r["P_in"], "P_sheet": r["P_sheet"] / r["P_in"], "balance": r["balance"]}
    return out


def main():
    c.require_private_repo()
    mat = json.load(open(c.PREREG / "02_materials.json"))
    dark = load_dark()
    tt = np.array(mat["T_K"])
    mu_sum = np.array(mat["mu_h_ibte"]) + np.array(mat["mu_e_ibte"])
    m = tc.chip_model()
    s, R, C = m["s"], m["R"], m["C"]
    n = R.size
    T0 = s["T_chuck_K"]
    tau = R * C
    tl = c.x_timeline()
    T = tc.sweep_temperatures(m, tl)                  # temperature at each sweep of the timeline, each sample
    day = c.raw_day("X")
    idx = {name: i for i, (name, _, _) in enumerate(tl)}
    dT = {k: T[idx[day["lit"].format(k=k)]] for k in SETTINGS}
    dT_after, dT_710 = T[idx[day["after"]]], T[idx[day["bias"]]]

    # sensitivities interpolated in ln(rho) across the sampled handle resistivity; p type
    rhos = sorted(dark)
    fs = sorted(dark[rhos[0]])
    def interp(key, f, rho):
        y = np.array([dark[r][f][key] for r in rhos])
        lr, lq = np.log(rhos), np.log(rho)
        return np.interp(lq, lr, y.real) + 1j * np.interp(lq, lr, y.imag)
    rho = s["rho_handle_ohmcm"]
    lay = mat["layers_vs_rho"]
    slope_rhos = np.array([float(k) for k in lay])
    slope_vals = np.array([v["dlnsigma_dT_300K"] for v in lay.values()])
    dlns = np.interp(np.log(rho), np.log(slope_rhos), slope_vals)

    q = c.percentiles
    pred = {"what": "fingerprint predictions; comparison only after this file is committed", "handle_rhos_run": rhos,
            "dark_sensitivity": {str(r): {f"{f:g}": [dark[r][f]["dY_dlnsigma"].real, dark[r][f]["dY_dlnsigma"].imag] for f in fs} for r in rhos}}

    # F1 + F2, direct heating at X and Ka centre
    for name, f in (("X", 10.0), ("Ka", 31.5)):
        dyd = interp("dY_dlnsigma", f, rho)
        pred[f"dY_dT_uS_per_K_{name}"] = {"re": q((dyd * dlns).real * 1e6), "im": q((dyd * dlns).imag * 1e6)}
        pred[f"phase_dlnsigma_deg_{name}"] = q(np.degrees(np.angle(dyd)))
        pred[f"H_T_dY_uS_{name}"] = {str(k): {"re": q((dyd * dlns * dT[k]).real * 1e6), "im": q((dyd * dlns * dT[k]).imag * 1e6)} for k in SETTINGS}
    pred["F2_H_T_exponent"] = "1 in dT, so dG proportional to P above threshold, sign negative until sigma(T) recovers: see 02_materials T_where_sigma_recovers_its_300K_value"
    # F3: in-band frequency dependence
    ratio = lambda key: interp(key, 11.75, rho) / interp(key, 8.25, rho)
    pred["F3_ratio_11p75_over_8p25"] = {"uniform_handle_H_T": {"abs": q(np.abs(ratio("dY_dlnsigma"))), "phase_deg": q(np.degrees(np.angle(ratio("dY_dlnsigma"))))},
                                        "top_150um_H_TC": {"abs": q(np.abs(ratio("dY_dlnsigma_top"))), "phase_deg": q(np.degrees(np.angle(ratio("dY_dlnsigma_top"))))}}
    # H_TC: steepening available from the temperature rise
    def g(temp):
        mu = np.interp(temp, tt, mu_sum) / np.interp(T0, tt, mu_sum)
        pow_ = (temp / T0) ** s["tau_exponent"]
        act = np.exp(s["tau_Ea_eV"] / 8.617333262e-5 * (1 / T0 - 1 / temp))
        return mu * np.where(s["tau_activated"], act, pow_)
    g3 = g(T0 + dT[3])
    ratio_tc = {k: (c.laser_power(np.full(n, float(k)), s["power_amps"]) / c.laser_power(np.full(n, 3.0), s["power_amps"])) * g(T0 + dT[k]) / g3 for k in SETTINGS}
    pred["H_TC_ratio_to_s3"] = {str(k): q(v) for k, v in ratio_tc.items()}
    pred["H_TC_steepening_g7_over_g3"] = q(g(T0 + dT[7]) / g3)
    samples = {**{f"ratio_{k}": v for k, v in ratio_tc.items()}, "R": R, "C": C, "amps": s["power_amps"], "dT_after": dT_after, "dT_710": dT_710, "rho": rho, "dlns": dlns,
               **{f"dT_{k}": v for k, v in dT.items()}}
    np.savez(c.OUT / "heating_samples.npz", **samples)
    pred["samples_sha256"] = c.arrays_hash(samples)
    # F4: the Ka day under the same laser and mount; the steepening carries over because dT depends on total absorbed power
    pred["F4_Ka_ratio_to_s3_if_H_TC"] = {str(k): q(v) for k, v in ratio_tc.items()}
    # D1, D2: time signatures
    dyd = interp("dY_dlnsigma", 10.0, rho)
    pred["D1_dark_after_uS_at_273s"] = {"re": q((dyd * dlns * dT_after).real * 1e6), "im": q((dyd * dlns * dT_after).imag * 1e6), "dT_K": q(dT_after)}
    pred["D2_7_10V_minus_7_0V_temperature_K"] = q(dT_710 - dT[7])
    pred["D2_response_change_fraction_if_H_TC"] = q(g(T0 + dT_710) / g(T0 + dT[7]) - 1)
    pred["tau_th_s"] = q(tau)
    c.register("03_fingerprints", pred, code=[__file__, c.HERE / "common.py"])
    print(json.dumps({k: pred[k] for k in ("dY_dT_uS_per_K_X", "phase_dlnsigma_deg_X", "H_TC_steepening_g7_over_g3", "D1_dark_after_uS_at_273s", "D2_7_10V_minus_7_0V_temperature_K", "D2_response_change_fraction_if_H_TC")}, indent=0)[:1600])


if __name__ == "__main__":
    main()

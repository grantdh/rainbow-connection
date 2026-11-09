"""R_th of the chip resting on the probe-station chuck, from contact physics (00d B); registers prereg/08_contact.json before it is compared with the flip map.

Gas gap after Yovanovich: mean-plane separation Y = 1.184 sigma (-ln(3.132 P / H_c))^0.547 under the chip's own weight P = rho g t, plus the gas rarefaction length M,
plus half the non-flatness (bow + chuck), plus one particle that wedges the chip (gap g0 at one edge, g0 + h at the other: effective gap h / ln(1 + h / g0)).
h = k_air / gap in parallel with the solid contact (negligible at ~15 Pa, reported), the probe fingers and the top surface's loss. Vacuum hold-down is not assumed:
without it the contact is at its worst, which favours heating. The records (photos of the mount, the wafer's spec sheet) and the handle thickness are in
device.json's deep section.
"""
import json

import numpy as np

import common as c

K_AIR = 0.026            # W/m/K at 300 K
K_SI = 148.0
K_FINGER, FINGER_A, FINGER_L = 110.0, 40e-6 * 20e-6, 0.6e-3      # BeCu finger, cross-section and length (ASM)


def sample(n=200_000, seed=c.SEED + 8):
    rng = np.random.default_rng(seed)
    return {"sigma_um": np.exp(rng.uniform(np.log(0.3), np.log(3.0), n)), "M_um": rng.uniform(0.2, 0.4, n), "H_GPa": rng.uniform(1.0, 4.0, n),
         "flat_um": rng.uniform(0.0, 20.0, n), "particle_um": rng.uniform(0.0, 100.0, n), "t_handle_um": rng.uniform(*c.device()["deep"]["handle"]["t_um_range"], n),
         "side_mm": c.device()["stack"]["chip_m"] * 1e3 * rng.uniform(0.8, 1.2, n), "h_top": rng.uniform(5.0, 25.0, n)}


def r_th(s):
    t = 1e-6 * s["t_handle_um"]
    A = (1e-3 * s["side_mm"]) ** 2
    P = 2329.0 * 9.81 * t                                              # Pa, the chip's own weight
    Y = 1.184 * s["sigma_um"] * (-np.log(3.132 * P / (1e9 * s["H_GPa"]))) ** 0.547
    g0 = Y + s["M_um"] + 0.5 * s["flat_um"]
    h = s["particle_um"]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(h > 1e-3, h / np.log1p(h / g0), g0)             # wedge-averaged conductance
    R_gap = 1e-6 * gap / (K_AIR * A) + t / (K_SI * A)
    R_fingers = FINGER_L / (K_FINGER * FINGER_A) / 2
    R = 1.0 / (1.0 / R_gap + 1.0 / R_fingers + s["h_top"] * A)
    C = 2329.0 * 712.0 * t * A
    return dict(R=R, C=C, gap_um=gap, P=P, R_fingers=R_fingers, Y_um=Y)


def main():
    c.require_private_repo()
    s = sample()
    r = r_th(s)
    q = c.percentiles
    q99 = lambda a: {**q(a), "q99": float(np.percentile(a, 99))}
    no_particle = r_th({**s, "particle_um": np.zeros_like(s["particle_um"])})
    d = c.device()["deep"]
    res = {"what": "R_th of the chip resting on the bare chuck without vacuum (00d B); exclusion level = 99th percentile",
           "records": d["text"]["08_records"],
           "R_th_K_W": q99(r["R"]), "R_th_K_W_without_the_particle": q99(no_particle["R"]), "effective_gap_um": q99(r["gap_um"]),
           "mean_plane_separation_um": q(r["Y_um"]), "contact_pressure_Pa": float(np.median(r["P"])),
           "tau_th_s": q(r["R"] * r["C"]), "C_J_K": q(r["C"]),
           "probe_fingers_R_K_W": float(r["R_fingers"]),
           "P_R_th_at_least": {f"{x:g}": float(np.mean(r["R"] >= x)) for x in d["contact_thresholds_K_W"]},
           "notes": ["Solid-spot conductance at ~15 Pa is below 1e-3 of the gas path and is left out.",
                     "A vacuum hold-down would raise the contact pressure by 10^3-10^4 and lower R_th further; not assumed."]}
    c.register("08_contact", res, code=[__file__, c.HERE / "common.py"])
    print(json.dumps({k: res[k] for k in ("R_th_K_W", "R_th_K_W_without_the_particle", "effective_gap_um", "tau_th_s", "P_R_th_at_least")}, indent=0)[:1500])


if __name__ == "__main__":
    main()

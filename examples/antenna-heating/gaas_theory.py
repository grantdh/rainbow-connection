"""Side-effect-free physics of the antenna-heating chain, shared by run.py and muppet-studio's nodes.

    k_path, parse_bands, band_mass   conduction-band mass near Gamma from a pw.x 'bands' run
    kane_mass                        two-band Kane mass at a corrected gap
    drude, eps_drude                 Drude permittivity of a doped semiconductor (MEEP units, 1 length unit = 1 um)
    fin_temperature, rod_profile     steady 1D conduction along a rod
"""

from __future__ import annotations

import re

import numpy as np

E, ME, EPS0, C0 = 1.602176634e-19, 9.1093837015e-31, 8.8541878128e-12, 299792458.0
RY_EV = 13.605693122994
THZ_PER_MEEP = 299.792458            # f[THz] = f[MEEP] x this (c / 1 um in THz), with 1 length unit = 1 um


def k_path(kmax: float = 0.04, n: int = 8) -> tuple[str, np.ndarray, list[str]]:
    """Gamma plus n points along [100] (Gamma-X) and [111] (Gamma-L): the pw.x K_POINTS tpiba block, |k| in
    units of 2 pi / a, and a direction label per point."""
    mags = np.linspace(kmax / n, kmax, n)
    pts, labels = [(0.0, 0.0, 0.0)], ["G"]
    for m in mags:
        pts.append((m, 0.0, 0.0)); labels.append("X")
    for m in mags:
        pts.append(tuple(m / np.sqrt(3) * np.ones(3))); labels.append("L")
    lines = ["K_POINTS tpiba", f"  {len(pts)}"] + [f"  {x:.8f} {y:.8f} {z:.8f} 1" for x, y, z in pts]
    return "\n".join(lines), np.array([np.linalg.norm(p) for p in pts]), labels


def parse_bands(text: str) -> list[np.ndarray]:
    """Eigenvalues per k-point from a pw.x bands/nscf output: the numeric lines after each 'bands (ev):'."""
    out = []
    for block in text.split("End of band structure calculation")[-1].split("bands (ev):")[1:]:
        vals: list[float] = []
        for line in block.splitlines()[1:]:
            if not line.strip():
                if vals:
                    break
                continue
            nums = re.findall(r"-?\d+\.\d+", line)
            if not nums or len("".join(nums)) < len(line.replace(" ", "")) - 2:   # a text line: block ended
                break
            vals += [float(v) for v in nums]
        out.append(np.array(vals))
    return out


def band_mass(kmag_tpiba: np.ndarray, labels: list[str], e_c: np.ndarray, alat_bohr: float) -> dict[str, float]:
    """Band-edge mass (m_e) per direction label: E = E_c + Ry k^2 / m* + c4 k^4, k in 1/bohr, fitted through
    Gamma (point 0) and that direction's points."""
    kb = np.asarray(kmag_tpiba) * 2 * np.pi / alat_bohr
    e_c = np.asarray(e_c)
    masses = {}
    for lab in dict.fromkeys(l for l in labels if l != "G"):
        sel = [0] + [i for i, l in enumerate(labels) if l == lab]
        k2 = kb[sel] ** 2
        (c2, _), *_ = np.linalg.lstsq(np.vstack([k2, k2 ** 2]).T, e_c[sel] - e_c[0], rcond=None)
        masses[lab] = RY_EV / c2
    return masses


def kane_mass(m_dft: float, gap_dft: float, gap: float) -> tuple[float, float]:
    """Two-band Kane model 1/m* = 1 + E_P / E_g: E_P from the DFT mass and gap, then m* at the corrected gap.
    Returns (m*, E_P in eV)."""
    e_p = gap_dft * (1 / m_dft - 1)
    return 1 / (1 + e_p / gap), e_p


def drude(m_star: float, n_m3: float, sigma_dc: float, eps_inf: float) -> dict:
    """eps(f) = eps_inf - f_p^2 / (f^2 + i f gamma): plasma frequency from n and m*, damping from the DC
    conductivity (gamma = n e^2 / (m* sigma_DC)); MEEP frequencies with 1 length unit = 1 um."""
    m = m_star * ME
    wp2 = n_m3 * E ** 2 / (EPS0 * m)
    gamma = n_m3 * E ** 2 / (m * sigma_dc)
    to_meep = 1e-6 / C0                  # ordinary frequency in Hz -> MEEP units
    return {"m_star": m_star, "n_m3": n_m3, "sigma_dc": sigma_dc, "eps_inf": eps_inf,
            "omega_p": float(np.sqrt(wp2)), "gamma": gamma,
            "f_p_screened_THz": float(np.sqrt(wp2 / eps_inf)) / (2 * np.pi) / 1e12, "gamma_THz": gamma / (2 * np.pi) / 1e12,
            "fp_meep": float(np.sqrt(wp2)) / (2 * np.pi) * to_meep, "gamma_meep": gamma / (2 * np.pi) * to_meep}


def eps_drude(f_meep: np.ndarray, d: dict) -> np.ndarray:
    return d["eps_inf"] - d["fp_meep"] ** 2 / (f_meep ** 2 + 1j * f_meep * d["gamma_meep"])


def fin_temperature(s: np.ndarray, q: np.ndarray, kappa_a: float) -> np.ndarray:
    """Steady 1D conduction, T(0) = T0 (anchored at s = 0), dT/ds = 0 at s = s[-1] (tip, or a symmetry plane).
    kappa A T'' = -q  ->  dT/ds = (1/kappa A) int_s^L q,  T(s) = int_0^s dT/ds (trapezoid on the grid)."""
    tail = np.concatenate([np.cumsum(((q[1:] + q[:-1]) / 2 * np.diff(s))[::-1])[::-1], [0.0]])   # int_s^L q
    return np.concatenate([[0.0], np.cumsum((tail[1:] + tail[:-1]) / 2 * np.diff(s))]) / kappa_a


def rod_profile(heat: dict, length_um: float, intensity_w_m2: float) -> tuple[np.ndarray, np.ndarray]:
    """Heat per length q (W/m) along a receiving rod bridging two contacts, from one heat entry of
    meep_gaas_rod.py (one-cell slabs above the rod's centre, heat per unit incident intensity). Returns s (um,
    distance from the end contact, 0 .. L/2) and q, padded with the end values at the contact face and at the
    middle (the symmetry plane)."""
    half = length_um / 2
    z = np.asarray(heat["z"])
    inside = z < half
    s = (half - z[inside])[::-1]
    q = (np.asarray(heat["q_z"])[inside] / heat["incident"])[::-1] * 1e-6 * intensity_w_m2   # um x W/m^2 -> W/m
    return np.concatenate([[0.0], s, [half]]), np.concatenate([[q[0]], q, [q[-1]]])

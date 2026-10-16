"""Transmission-line model of a rectangular microstrip patch (Balanis, "Antenna Theory", 4th ed., sec. 14.2).

All lengths in metres, frequencies in Hz.
"""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0


def eps_eff(eps_r: float, h: float, w: float) -> float:
    """Effective permittivity for W/h > 1 (Balanis eq. 14-1)."""
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / w) ** -0.5


def delta_l(eps_r: float, h: float, w: float) -> float:
    """Fringing length extension of each radiating edge (Hammerstad, Balanis eq. 14-2)."""
    ee = eps_eff(eps_r, h, w)
    return 0.412 * h * (ee + 0.3) * (w / h + 0.264) / ((ee - 0.258) * (w / h + 0.8))


def resonant_frequency(eps_r: float, h: float, w: float, l: float) -> float:
    """TM010 resonance including fringing (Balanis eq. 14-5)."""
    return C0 / (2 * (l + 2 * delta_l(eps_r, h, w)) * np.sqrt(eps_eff(eps_r, h, w)))


def design(eps_r: float, h: float, f_r: float) -> tuple[float, float]:
    """Width and length for a target resonance (Balanis eqs. 14-6, 14-7)."""
    w = C0 / (2 * f_r) * np.sqrt(2 / (eps_r + 1))
    l = C0 / (2 * f_r * np.sqrt(eps_eff(eps_r, h, w))) - 2 * delta_l(eps_r, h, w)
    return w, l


def edge_resistance(f: float, w: float, l: float) -> float:
    """Cavity-model input resistance at the radiating edge, 1 / (2 (G1 + G12)) (Balanis eqs. 14-12, 14-18, 14-17)."""
    from scipy.special import j0
    k0 = 2 * np.pi * f / C0
    th = np.linspace(1e-6, np.pi - 1e-6, 20001)
    slot = (np.sin(k0 * w / 2 * np.cos(th)) / np.cos(th)) ** 2 * np.sin(th) ** 3
    g1 = np.trapezoid(slot, th) / (120 * np.pi ** 2)
    g12 = np.trapezoid(slot * j0(k0 * l * np.sin(th)), th) / (120 * np.pi ** 2)
    return 1 / (2 * (g1 + g12))


def inset_resistance(f: float, w: float, l: float, y0: float) -> float:
    """R_in at a feed y0 inside the radiating edge: R_edge cos^2(pi y0 / L) (Balanis eq. 14-20a)."""
    return edge_resistance(f, w, l) * np.cos(np.pi * y0 / l) ** 2


if __name__ == "__main__":
    w, l = design(2.2, 0.1588e-2, 10e9)
    print(f"Balanis Example 14.1: W = {w * 100:.3f} cm (1.186), L = {l * 100:.3f} cm (0.906), "
          f"eps_eff = {eps_eff(2.2, 0.1588e-2, w):.3f} (1.972), dL = {delta_l(2.2, 0.1588e-2, w) * 100:.4f} cm (0.081)")
    print(f"round trip: f_r = {resonant_frequency(2.2, 0.1588e-2, w, l) / 1e9:.4f} GHz")
    print(f"edge resistance at 10 GHz: {edge_resistance(10e9, w, l):.1f} ohm")

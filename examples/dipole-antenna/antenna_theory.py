"""Closed-form linear-antenna results used as references (numpy/scipy only).

All formulas assume a centre-fed straight wire of total length l and radius a carrying the classic
sinusoidal current I(z) = I_m sin(k (l/2 - |z|)), as in Balanis, "Antenna Theory", 4th ed., ch. 4 and 8.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sici

ETA0 = 376.730313668          # free-space wave impedance (ohm)
EULER = 0.5772156649015329


def _si(x):
    return sici(x)[0]


def _ci(x):
    return sici(x)[1]


def induced_emf_impedance(l_over_lambda: np.ndarray | float, a_over_lambda: float) -> np.ndarray:
    """Input impedance (ohm) by the induced-EMF method, Balanis eqs. (8-60a), (8-60b), referred to the
    feed with R_in = R_m / sin^2(kl/2) (eq. 8-61). Exact for the assumed sinusoidal current, which wires of
    finite radius only approximate."""
    kl = 2 * np.pi * np.asarray(l_over_lambda, dtype=float)
    ka2_over_l = 2 * np.pi * a_over_lambda ** 2 / np.asarray(l_over_lambda, dtype=float)
    rm = ETA0 / (2 * np.pi) * (EULER + np.log(kl) - _ci(kl)
                               + 0.5 * np.sin(kl) * (_si(2 * kl) - 2 * _si(kl))
                               + 0.5 * np.cos(kl) * (EULER + np.log(kl / 2) + _ci(2 * kl) - 2 * _ci(kl)))
    xm = ETA0 / (4 * np.pi) * (2 * _si(kl) + np.cos(kl) * (2 * _si(kl) - _si(2 * kl))
                               - np.sin(kl) * (2 * _ci(kl) - _ci(2 * kl) - _ci(2 * ka2_over_l)))
    return (rm + 1j * xm) / np.sin(kl / 2) ** 2


def x_zero_crossing(f: np.ndarray, z: np.ndarray) -> tuple[float, float] | None:
    """Resonance of a sampled impedance: the first frequency where X = Im z changes sign (linear
    interpolation between samples) and R there, or None when X keeps one sign."""
    f, z = np.asarray(f, dtype=float), np.asarray(z)
    x = z.imag
    cross = np.where(np.sign(x[:-1]) != np.sign(x[1:]))[0]
    if not len(cross):
        return None
    i = cross[0]
    f_res = float(f[i] - x[i] * (f[i + 1] - f[i]) / (x[i + 1] - x[i]))
    return f_res, float(np.interp(f_res, f, z.real))


def pattern(theta: np.ndarray, l_over_lambda: float) -> np.ndarray:
    """Radiation intensity U(theta) (arbitrary scale) of the sinusoidal-current dipole (Balanis 4-58a)."""
    kl2 = np.pi * l_over_lambda
    s = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = ((np.cos(kl2 * np.cos(theta)) - np.cos(kl2)) / s) ** 2
    return np.nan_to_num(u)


def directivity(u: np.ndarray, theta: np.ndarray) -> float:
    """D = 4 pi U_max / P for an azimuthally symmetric pattern sampled on theta in [0, pi]."""
    p = 2 * np.pi * np.trapezoid(u * np.sin(theta), theta)
    return float(4 * np.pi * u.max() / p)


if __name__ == "__main__":
    z = induced_emf_impedance(0.5, 1e-4)
    print(f"half-wave, thin wire: Z = {z.real:.2f} + j{z.imag:.2f} ohm   (Balanis: 73.13 + j42.55)")
    th = np.linspace(0, np.pi, 20001)
    print(f"D(half-wave) = {directivity(pattern(th, 0.5), th):.4f}  (1.643)   "
          f"D(Hertzian, sin^2) = {directivity(np.sin(th) ** 2, th):.4f}  (1.5)")
    l = np.linspace(0.40, 0.55, 3001)
    x = induced_emf_impedance(l, 0.02).imag
    print(f"resonant length for a = lambda/50: {l[np.argmin(abs(x))]:.4f} lambda")

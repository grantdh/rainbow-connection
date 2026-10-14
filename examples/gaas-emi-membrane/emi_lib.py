"""Drude free carriers + characteristic-matrix TMM + shielding effectiveness.

Same conventions as MoreThanSkinDeep/mtsd (which has the pytest coverage): time
dependence e^{-iwt}, absorbing media have Im eps > 0, Born & Wolf characteristic
matrix with -i off-diagonals. Kept self-contained so the example has no import
outside this repo; the plan is to move it into rainbow-connection/lib.
"""

from __future__ import annotations

import numpy as np

e = 1.602176634e-19
m_e = 9.1093837015e-31
eps0 = 8.8541878128e-12
mu0 = 1.25663706212e-6
c = 299792458.0
Z0 = np.sqrt(mu0 / eps0)
HBAR_EV_S = 6.582119569e-16


def gamma_from_sigma(sigma_Sm: float, n_m3: float, m_star_kg: float) -> float:
    """Scattering rate [rad/s] that reproduces a measured DC conductivity: gamma = n e^2 / (m* sigma)."""
    return n_m3 * e ** 2 / (m_star_kg * sigma_Sm)


def gamma_from_mobility(mu_m2Vs: float, m_star_kg: float) -> float:
    return e / (mu_m2Vs * m_star_kg)


def omega_p2(n_m3: float, m_star_kg: float) -> float:
    return n_m3 * e ** 2 / (eps0 * m_star_kg)


def drude_eps(freq_Hz, n_m3: float, m_star_kg: float, gamma: float) -> np.ndarray:
    """Free-carrier contribution alone (add to the interband eps): -wp^2 / (w^2 + i gamma w)."""
    w = 2 * np.pi * np.atleast_1d(freq_Hz).astype(float)
    return -omega_p2(n_m3, m_star_kg) / (w ** 2 + 1j * gamma * w)


def _layer_matrix(freq_Hz, d_m: float, eps_c) -> np.ndarray:
    f = np.atleast_1d(freq_Hz).astype(float)
    w = 2 * np.pi * f
    eps_c = np.broadcast_to(np.asarray(eps_c, dtype=complex), f.shape)
    n = np.sqrt(eps_c)
    n = np.where(n.imag < 0, -n, n)
    phi = n * w * d_m / c
    M = np.empty((2, 2) + f.shape, dtype=complex)
    M[0, 0] = np.cos(phi)
    M[0, 1] = -1j * np.sin(phi) / n
    M[1, 0] = -1j * n * np.sin(phi)
    M[1, 1] = np.cos(phi)
    return M


def tmm(freq_Hz, layers) -> dict:
    """Stack in air at normal incidence. layers = [(thickness_m, eps_complex_or_array), ...] in light order."""
    f = np.atleast_1d(freq_Hz).astype(float)
    M = np.zeros((2, 2, f.size), dtype=complex)
    M[0, 0] = M[1, 1] = 1.0
    for d, eps_c in layers:
        M = np.einsum("ikf,kjf->ijf", _layer_matrix(f, d, eps_c), M)
    den = M[0, 0] + M[0, 1] + M[1, 0] + M[1, 1]
    t = 2.0 / den
    r = (M[0, 0] + M[0, 1] - M[1, 0] - M[1, 1]) / den
    T, R = np.abs(t) ** 2, np.abs(r) ** 2
    return {"T": T, "R": R, "A": 1 - T - R, "SE_dB": -10 * np.log10(T)}


def thin_film_SE(sigma_Sm: float, d_m: float) -> float:
    """Sheet-conductance limit (d << skin depth): SE = -20 log10 |2 / (2 + Z0 sigma d)|."""
    return -20 * np.log10(abs(2.0 / (2.0 + Z0 * sigma_Sm * d_m)))


def skin_depth(freq_Hz: float, sigma_Sm: float) -> float:
    return np.sqrt(2.0 / (2 * np.pi * freq_Hz * mu0 * sigma_Sm))


def count_peaks(y: np.ndarray) -> int:
    return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))

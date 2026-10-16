"""Mie scattering of a homogeneous sphere (Bohren & Huffman, ch. 4), numpy/scipy only.

    qext, qsca = mie_q(m, x)        m: relative refractive index (complex ok), x = 2 pi r n_medium / lambda
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def mie_coefficients(m: complex, x: float) -> tuple[np.ndarray, np.ndarray]:
    """a_n, b_n for n = 1..nmax (B&H eq. 4.53 with mu = mu_1), nmax from Wiscombe's criterion."""
    nmax = int(np.round(x + 4.05 * x ** (1 / 3) + 2))
    n = np.arange(1, nmax + 1)
    mx = m * x
    jx, jx_d = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    yx, yx_d = spherical_yn(n, x), spherical_yn(n, x, derivative=True)
    jm, jm_d = spherical_jn(n, mx), spherical_jn(n, mx, derivative=True)
    # Riccati-Bessel functions psi_n(z) = z j_n(z), xi_n(z) = z h1_n(z) and their derivatives
    psi, psi_d = x * jx, jx + x * jx_d
    psim, psim_d = mx * jm, jm + mx * jm_d
    hx, hx_d = jx + 1j * yx, jx_d + 1j * yx_d
    xi, xi_d = x * hx, hx + x * hx_d
    a = (m * psim * psi_d - psi * psim_d) / (m * psim * xi_d - xi * psim_d)
    b = (psim * psi_d - m * psi * psim_d) / (psim * xi_d - m * xi * psim_d)
    return a, b


def mie_q(m: complex, x: float) -> tuple[float, float]:
    """Extinction and scattering efficiencies Q = sigma / (pi r^2)."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, len(a) + 1)
    qext = 2 / x ** 2 * np.sum((2 * n + 1) * (a + b).real)
    qsca = 2 / x ** 2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2))
    return float(qext), float(qsca)


def rayleigh_qsca(m: complex, x: float) -> float:
    return 8 / 3 * x ** 4 * abs((m ** 2 - 1) / (m ** 2 + 2)) ** 2


if __name__ == "__main__":
    # Bohren & Huffman Appendix A sample run: m = 1.55, r = 0.525 um, lambda = 0.6328 um
    x = 2 * np.pi * 0.525 / 0.6328
    qext, qsca = mie_q(1.55 + 0j, x)
    print(f"B&H sample: x = {x:.4f}  Qext = {qext:.5f}  Qsca = {qsca:.5f}   (book: 3.10543 / 3.10543)")
    print(f"Rayleigh x=0.05: mie {mie_q(1.5, 0.05)[1]:.6e}  rayleigh {rayleigh_qsca(1.5, 0.05):.6e}")
    print(f"large sphere x=200, m=1.33: Qext = {mie_q(1.33, 200)[0]:.4f}  (-> 2, extinction paradox)")
    print(f"absorbing m=1.5+0.1j x=2: Qext={mie_q(1.5+0.1j, 2)[0]:.4f} Qsca={mie_q(1.5+0.1j, 2)[1]:.4f}")

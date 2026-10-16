"""Absorbed power omega Im(E* . D) on MEEP's Yee grid, binned into one-cell slabs along z.

Shared by meep_gaas_dipole.py and meep_gaas_rod.py (same interpreter, same directory). E and D of a
component sit at the same Yee point, so the sum is exactly zero in vacuum and in perfect conductors, and
nothing depends on interpolating the normal field across a material surface (the first antenna version
interpolated E onto a grid straddling the GaAs surface and over-counted the heat up to 4.8-fold).
"""

from __future__ import annotations

import numpy as np
import meep as mp

PAIRS = ((mp.Ex, mp.Dx), (mp.Ey, mp.Dy), (mp.Ez, mp.Dz))


def add_heat_region(sim, freqs, half_width: float, top: float, px: float):
    """DFT of E and D on the Yee grid over |x|, |y| <= half_width and 0 <= z <= top, each plus two cells.
    The region's edges sit a quarter cell off the grid planes: an edge on a grid plane gains or loses a
    bracketing sample with floating-point noise. Returns the DFT object and the number of whole slabs."""
    hx_px, top_px = int(np.ceil(half_width / px)) + 2, int(np.ceil(top / px)) + 2
    z_lo, z_hi, hx = -0.25 * px, (top_px + 0.25) * px, (hx_px + 0.25) * px
    dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Dx, mp.Dy, mp.Dz], freqs, yee_grid=True,
                             center=mp.Vector3(z=(z_lo + z_hi) / 2), size=mp.Vector3(2 * hx, 2 * hx, z_hi - z_lo))
    return dft, top_px + 1


def slab_heat(sim, dft, k: int, f: float, px: float, n_slab: int) -> tuple[np.ndarray, float]:
    """Absorbed power per one-cell slab [j px, (j + 1) px] above z = 0 at the k-th DFT frequency f, each Yee
    sample shared linearly between the two nearest slab centres (a cell-centred sample goes wholly to its
    slab, a face sample half to each). Entry n_slab collects anything above the top slab. Returns the slabs
    and the heat in samples below z = 0, which the slabs leave out."""
    zc = np.asarray(sim.get_array_metadata(dft_cell=dft)[2], float)
    q = np.zeros(n_slab + 1)
    total = 0.0
    for ce, cd in PAIRS:
        e = np.asarray(sim.get_dft_array(dft, ce, k))
        d = np.asarray(sim.get_dft_array(dft, cd, k))
        per_z = (2 * np.pi * f * np.imag(np.conj(e) * d) * px ** 3).sum(axis=(0, 1))   # absorbed power per Yee plane
        total += per_z.sum()
        # z of the samples: the cell-centred grid along the component's own axis; along the others the face
        # grid, one sample more, bracketing both region edges (checked against a material edge at a known
        # height at 10, 15 and 20 px/um)
        z = zc if ce == mp.Ez else np.append(zc - px / 2, zc[-1] + px / 2)
        if len(z) != len(per_z):
            raise SystemExit(f"{len(per_z)} Yee samples along z for component {ce}, expected {len(z)}")
        u = z / px - 0.5
        i0 = np.floor(u + 1e-6).astype(int)
        w1 = np.clip(u - i0, 0.0, 1.0)
        for idx, wt in ((i0, 1 - w1), (i0 + 1, w1)):
            ok = (idx >= 0) & (idx <= n_slab)
            np.add.at(q, idx[ok], (wt * per_z)[ok])
    return q, total - q.sum()

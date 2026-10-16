"""MEEP 3D: probe-fed rectangular microstrip patch on a finite ground plane.

Units: 1 MEEP length unit = 1 cm, c = 1, so f[MEEP] = f[GHz] / 29.9792458.
Geometry (Balanis Example 14.1 by default): substrate eps_r = 2.2, h = 0.1588 cm, patch W x L with the
resonant length L along x and the width W along y, centred on a square ground plane and substrate of
side G. The cell is a whole number of pixels, the conductor surfaces sit on grid planes exactly h apart,
the PEC sheets are 1.5 px thick, and subpixel averaging is off so metal is never blended into dielectric.
Feed: a z-directed current line from the ground to the patch at x = -L/2 + inset, y = 0 (an ideal probe).

Recorded:
  - Harminv on E_z under the far radiating edge: resonant frequencies and Q of the cavity modes (Q
    threshold lowered from MEEP's default 50, which hid the Q ~ 10 TM010 mode in the first run);
  - the complex power S_feed leaving a small box around the probe (the power it delivers), and that power
    per unit source strength, whose peak marks the resonance seen from the feed;
  - near2far at the design frequency: E- and H-plane patterns, directivity.
Mirror plane y = 0 (E_z even) halves the work.

Run with the meep_arm interpreter:
    python meep_patch.py OUT_DIR [--px-per-h 4] [--eps 2.2] [--h 0.1588] [--w 1.186] [--l 0.906] [--inset-frac 0.25]
Writes OUT_DIR/patch.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import meep as mp

GHZ = 1 / 29.9792458            # MEEP frequency of 1 GHz when the length unit is 1 cm

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--px-per-h", type=int, default=4)
p.add_argument("--eps", type=float, default=2.2)
p.add_argument("--h", type=float, default=0.1588)
p.add_argument("--w", type=float, default=1.186)
p.add_argument("--l", type=float, default=0.906)
p.add_argument("--ground", type=float, default=3.0, help="ground-plane / substrate side (cm)")
p.add_argument("--inset-frac", type=float, default=0.25, help="probe distance from the feed edge, as a fraction of L")
p.add_argument("--f0-ghz", type=float, default=10.0)
p.add_argument("--nfreq", type=int, default=81)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

res = a.px_per_h / a.h
px = 1 / res
f0 = a.f0_ghz * GHZ
fcen, df = f0, 0.5 * f0
# The cell is built in whole pixels so that grid planes (where tangential E lives) fall exactly on the
# ground-plane and patch surfaces, h = px_per_h pixels apart. One-pixel PEC sheets that straddle grid
# points are otherwise blended into finite dielectric by subpixel averaging: the first run of this
# example simulated a bare substrate that way (eps through the patch read 1.5, 6.6, 2.2, ..., 3.3, 3.0).
n_pml, n_up, n_down, n_xy = (int(np.ceil(v * res)) for v in (1.5, 1.2, 0.8, a.ground / 2 + 0.8 + 1.5))
# MEEP anchors its grid at the cell centre, so the planes counted in from the cell boundary are grid planes
# only when every side has an even number of pixels (an odd count shifts them half a cell and the sheets
# blur away: at 6 px per h, with 197 pixels in z, the patch and ground plane vanished from the eps map)
n_up += (2 * n_pml + n_down + a.px_per_h + n_up) % 2
dpml = n_pml * px
sx = 2 * n_xy * px
nz = 2 * n_pml + n_down + a.px_per_h + n_up
sz = nz * px
cell = mp.Vector3(sx, sx, sz)
Z0 = sz / 2 - (n_pml + n_down) * px             # cell z of the ground plane's top face is -Z0 (a grid plane)


def at(x=0.0, y=0.0, z=0.0):
    """Cell coordinates from a frame with z = 0 on the ground plane's top face."""
    return mp.Vector3(x, y, z - Z0)


# Metal sheets 1.5 px thick, overlapping their surface plane by px/4, so that the grid plane at the surface
# lies strictly inside the metal and the substrate between the conductors is exactly h.
geometry = [
    mp.Block(size=mp.Vector3(a.ground, a.ground, a.h), center=at(z=a.h / 2), material=mp.Medium(epsilon=a.eps)),
    mp.Block(size=mp.Vector3(a.ground, a.ground, 1.5 * px), center=at(z=px / 4 - 0.75 * px), material=mp.metal),   # ground
    mp.Block(size=mp.Vector3(a.l, a.w, 1.5 * px), center=at(z=a.h - px / 4 + 0.75 * px), material=mp.metal),      # patch
]
x_feed = -a.l / 2 + a.inset_frac * a.l
src = mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=at(x_feed, 0, a.h / 2),
                size=mp.Vector3(0, 0, a.h))
sim = mp.Simulation(resolution=res, cell_size=cell, boundary_layers=[mp.PML(dpml)], geometry=geometry,
                    sources=[src], symmetries=[mp.Mirror(mp.Y)], eps_averaging=False)

freqs = np.linspace(fcen - df / 2, fcen + df / 2, a.nfreq)
# Port: the power the probe delivers is the complex Poynting flux out of a small box around it, between the
# ground and patch planes (S_feed). No current normalisation is needed for that. An absolute input
# impedance needs the probe current, which an ideal line source in a cavity does not give cleanly (a loop
# around it also encloses the cavity's displacement current), so Z_in is left to a coax-pin port model.
# The side faces run one cell into each metal sheet, where the fields are zero: with faces ending on the
# metal planes, the end samples (interpolated half-way into the metal) got half weight on top, and the
# box read 7-10 % short of the radiated power at 3-4 cells across h.
# The box has no top or bottom face, so it must lie wholly under the patch.
fb = 3 * px
hb = a.h + 2 * px
if x_feed - fb <= -a.l / 2 or fb >= a.w / 2:
    raise SystemExit(f"the feed box (half-size {fb:.3f} cm around x = {x_feed:.3f} cm) reaches past the patch edge: "
                     "raise --inset-frac or --px-per-h")
COMPS = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
feed_faces = [(at(x_feed + fb, 0, a.h / 2), mp.Vector3(0, 2 * fb, hb), (1, 0, 0)), (at(x_feed - fb, 0, a.h / 2), mp.Vector3(0, 2 * fb, hb), (-1, 0, 0)),
              (at(x_feed, +fb, a.h / 2), mp.Vector3(2 * fb, 0, hb), (0, 1, 0)), (at(x_feed, -fb, a.h / 2), mp.Vector3(2 * fb, 0, hb), (0, -1, 0))]
feed_dft = [sim.add_dft_fields(COMPS, freqs, center=c, size=s) for c, s, _ in feed_faces]
# near2far box enclosing ground plane and patch
bx = a.ground / 2 + 0.3
z_lo, z_hi = -0.4, a.h + 0.6
zc, zs = (z_lo + z_hi) / 2, z_hi - z_lo
n2f = sim.add_near2far(f0, 0, 1,
                       mp.Near2FarRegion(center=at(+bx, 0, zc), size=mp.Vector3(0, 2 * bx, zs), weight=+1),
                       mp.Near2FarRegion(center=at(-bx, 0, zc), size=mp.Vector3(0, 2 * bx, zs), weight=-1),
                       mp.Near2FarRegion(center=at(0, +bx, zc), size=mp.Vector3(2 * bx, 0, zs), weight=+1),
                       mp.Near2FarRegion(center=at(0, -bx, zc), size=mp.Vector3(2 * bx, 0, zs), weight=-1),
                       mp.Near2FarRegion(center=at(0, 0, z_hi), size=mp.Vector3(2 * bx, 2 * bx, 0), weight=+1),
                       mp.Near2FarRegion(center=at(0, 0, z_lo), size=mp.Vector3(2 * bx, 2 * bx, 0), weight=-1))
harm = mp.Harminv(mp.Ez, at(a.l / 2 - 2 * px, 0, a.h / 2), fcen, df)
harm.Q_thresh = 1.0         # MEEP discards Q < 50 by default; this patch's TM010 has Q ~ 10

t0 = time.time()
sim.run(mp.after_sources(harm), until_after_sources=mp.stop_when_dft_decayed(tol=1e-7, minimum_run_time=150))
wall = time.time() - t0


s_feed = np.zeros(len(freqs), complex)
for dft, (_, _, nrm) in zip(feed_dft, feed_faces):
    w = np.squeeze(np.asarray(sim.get_array_metadata(dft_cell=dft)[3]))
    for k in range(len(freqs)):
        F = [np.squeeze(sim.get_dft_array(dft, c, k)) for c in COMPS]
        cr = (np.conj(F[1]) * F[5] - np.conj(F[2]) * F[4], np.conj(F[2]) * F[3] - np.conj(F[0]) * F[5],
              np.conj(F[0]) * F[4] - np.conj(F[1]) * F[3])
        s_feed[k] += np.sum(w * sum(nc * c for nc, c in zip(nrm, cr)))
# delivered power per unit source strength: proportional to the input resistance seen by a fixed current
spec = np.array([abs(src.src.fourier_transform(f)) ** 2 for f in freqs]) * freqs ** 2
delivered = s_feed.real / spec

R = 1000.0


def u_at(th, ph):
    """Radiation intensity r^2 Re(E* x H) . r_hat at (theta, phi), from near2far."""
    rhat = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    ff = np.array(sim.get_farfield(n2f, mp.Vector3(*(R * rhat))))
    return R ** 2 * np.real(np.cross(np.conj(ff[0:3]), ff[3:6])) @ rhat


# Far field. The mirror plane y = 0 makes U(theta, -phi) = U(theta, phi), so the H-plane cut and the
# directivity grid evaluate half their points and mirror the rest.
# E-plane (phi = 0, xz) and H-plane (phi = 90, yz), theta from -90 to 90 deg (broadside = 0)
theta = np.linspace(-np.pi / 2, np.pi / 2, 91)
u_h = np.array([u_at(th, np.pi / 2) for th in theta[len(theta) // 2:]])
pat = {"E": np.array([u_at(th, 0.0) for th in theta]), "H": np.concatenate([u_h[:0:-1], u_h])}
# directivity: full-sphere quadrature of U on a (theta, phi) grid
nt, npf = 61, 48
tq = np.linspace(0, np.pi, nt)
pq = np.linspace(0, 2 * np.pi, npf, endpoint=False)
U = np.zeros((nt, npf))
half = npf // 2
for i, th in enumerate(tq):
    if i in (0, nt - 1):                          # the poles: r_hat does not depend on phi
        U[i] = u_at(th, 0.0)
        continue
    U[i, :half + 1] = [u_at(th, ph) for ph in pq[:half + 1]]
    U[i, half + 1:] = U[i, half - 1:0:-1]         # phi_(npf - j) = -phi_j
p_rad = np.trapezoid(np.sin(tq) * U.mean(axis=1) * 2 * np.pi, tq)
p_in = s_feed.real                             # MEEP DFT convention: no factor 1/2, same as the far-field U

modes = [{"f_ghz": m.freq / GHZ, "Q": m.Q, "amp": abs(m.amp), "err": abs(m.err)} for m in harm.modes]
json.dump({
    "resolution": res, "px_per_h": a.px_per_h, "eps": a.eps, "h": a.h, "w": a.w, "l": a.l, "ground": a.ground,
    "inset": a.inset_frac * a.l, "cell": [sx, sx, sz], "voxels": int(round(sx * res)) ** 2 * int(round(sz * res)),
    "wall_s": round(wall, 1), "harminv": modes,
    "freq_ghz": (freqs / GHZ).tolist(), "delivered_norm": (delivered / delivered.max()).tolist(),
    "theta": theta.tolist(), "u_E": pat["E"].tolist(), "u_H": pat["H"].tolist(),
    "directivity_f0": float(4 * np.pi * U.max() / p_rad), "p_rad_f0": float(p_rad),
    "p_in_f0": float(np.interp(f0, freqs, p_in)), "f0_ghz": a.f0_ghz, "broadside_over_max": float(U[0].mean() / U.max()),
}, open(a.out / "patch.json", "w"), indent=1)
print(f"patch: res {res:.2f}, wall {wall:.0f} s, modes {[(round(m['f_ghz'], 3), round(m['Q'], 1)) for m in modes]}")

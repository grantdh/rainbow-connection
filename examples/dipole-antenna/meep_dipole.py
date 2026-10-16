"""MEEP 3D dipole antennas in free space.

Units: wavelength lambda0 = 1 at the design frequency f0 = 1 (c = 1, Z0 = 1 in MEEP units).

--mode hertzian   a point Jz source at the origin.
--mode halfwave   a perfectly conducting wire of radius a along z, total length l = lambda0/2, cut by a
                  delta gap of g lambda0 (rounded to whole cells) at the centre, where a Jz current source
                  fills the gap.

Both modes record:
  - the net power leaving a closed box of six flux planes around the antenna (P_box, all frequencies);
  - the far field at f0 on a theta sweep (near2far from the same box), giving the radiation intensity
    U(theta) = r^2 Re(E* x H).r, the far-field power integral P_far, and the directivity 4 pi U_max / P_far.
halfwave also records the port: the complex power S leaving a small box around the gap and the arm
current I (Ampere loop around the upper arm just beyond the gap), so Z_in(f) = S / |I|^2 and P_in = Re(S).
Power quantities use MEEP's DFT convention (flux = Re(E* x H), no factor 1/2), so R = P_box / |I|^2.

Mirror planes x and y (phase +1) and z (phase -1, E_z even in z) cut the work by eight.

Run with the meep_arm interpreter:
    python meep_dipole.py OUT_DIR --mode {hertzian,halfwave} [--resolution N] [--radius a] [--gap g]
Writes OUT_DIR/dipole.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import meep as mp

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--mode", choices=["hertzian", "halfwave"], required=True)
p.add_argument("--resolution", type=float, default=None)
p.add_argument("--radius", type=float, default=0.02, help="wire radius in lambda0 (halfwave)")
p.add_argument("--length", type=float, default=0.5, help="total wire length in lambda0 (halfwave)")
p.add_argument("--gap", type=float, default=0.02, help="feed gap in lambda0, rounded to whole cells (halfwave)")
p.add_argument("--nfreq", type=int, default=41)
p.add_argument("--ntheta", type=int, default=91)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
res = a.resolution or (40 if a.mode == "hertzian" else 100)

f0 = 1.0
fmin, fmax = (0.9, 1.1) if a.mode == "hertzian" else (0.65, 1.35)
fcen, df = 0.5 * (fmin + fmax), fmax - fmin
dpml = 0.5
half_l = 0.5 * a.length if a.mode == "halfwave" else 0.0
box = half_l + 0.15                          # flux / near2far box half-size (outside the antenna)
air = box + 0.25                             # free space before the PML
L = 2 * (air + dpml)
cell = mp.Vector3(L, L, L)
symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

geometry = []
gap = max(1, round(a.gap * res)) / res
if a.mode == "halfwave":
    arm = half_l - gap / 2
    for sign in (+1, -1):
        geometry.append(mp.Cylinder(radius=a.radius, height=arm, axis=mp.Vector3(z=1),
                                    center=mp.Vector3(z=sign * (gap / 2 + arm / 2)), material=mp.metal))
    src_size = mp.Vector3(2 * a.radius, 2 * a.radius, gap)   # a current sheet bundle filling the gap
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3(), size=src_size)]
else:
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3())]

sim = mp.Simulation(resolution=res, cell_size=cell, boundary_layers=[mp.PML(dpml)], sources=sources,
                    geometry=geometry, symmetries=symmetries)
faces = [(mp.Vector3(x=+box), mp.Vector3(0, 2 * box, 2 * box), +1), (mp.Vector3(x=-box), mp.Vector3(0, 2 * box, 2 * box), -1),
         (mp.Vector3(y=+box), mp.Vector3(2 * box, 0, 2 * box), +1), (mp.Vector3(y=-box), mp.Vector3(2 * box, 0, 2 * box), -1),
         (mp.Vector3(z=+box), mp.Vector3(2 * box, 2 * box, 0), +1), (mp.Vector3(z=-box), mp.Vector3(2 * box, 2 * box, 0), -1)]
flux = [sim.add_flux(fcen, df, a.nfreq, mp.FluxRegion(center=c, size=s)) for c, s, _ in faces]
n2f = sim.add_near2far(f0, 0, 1, *[mp.Near2FarRegion(center=c, size=s, weight=w) for c, s, w in faces])

feed = {}
if a.mode == "halfwave":
    freqs_dft = np.linspace(fmin, fmax, a.nfreq)
    px = 1 / res
    # Port: complex power S out of a small box around the gap, and the arm current I from Ampere loops around
    # the upper arm two cells beyond the gap (conduction current only), at two loop sizes. A loop in the
    # gap plane, used first, is ambiguous: the gap's fringing displacement current makes it grow with loop
    # size (14 % from 1 to 8 cells at 100 px/lambda), and the on-axis gap voltage was 13 % low there.
    z_loop = gap / 2 + 2 * px
    for n in (2, 4):
        sl = a.radius + n * px
        feed[f"loop{n}"] = [
            sim.add_dft_fields([mp.Hx], freqs_dft, center=mp.Vector3(0, -sl, z_loop), size=mp.Vector3(2 * sl, 0, 0)),
            sim.add_dft_fields([mp.Hy], freqs_dft, center=mp.Vector3(+sl, 0, z_loop), size=mp.Vector3(0, 2 * sl, 0)),
            sim.add_dft_fields([mp.Hx], freqs_dft, center=mp.Vector3(0, +sl, z_loop), size=mp.Vector3(2 * sl, 0, 0)),
            sim.add_dft_fields([mp.Hy], freqs_dft, center=mp.Vector3(-sl, 0, z_loop), size=mp.Vector3(0, 2 * sl, 0))]
    fx, fz = a.radius + 4 * px, gap / 2 + 2 * px
    COMPS = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
    feed_faces = [(mp.Vector3(x=+fx), mp.Vector3(0, 2 * fx, 2 * fz), (1, 0, 0)), (mp.Vector3(x=-fx), mp.Vector3(0, 2 * fx, 2 * fz), (-1, 0, 0)),
                  (mp.Vector3(y=+fx), mp.Vector3(2 * fx, 0, 2 * fz), (0, 1, 0)), (mp.Vector3(y=-fx), mp.Vector3(2 * fx, 0, 2 * fz), (0, -1, 0)),
                  (mp.Vector3(z=+fz), mp.Vector3(2 * fx, 2 * fx, 0), (0, 0, 1)), (mp.Vector3(z=-fz), mp.Vector3(2 * fx, 2 * fx, 0), (0, 0, -1))]
    feed["box"] = [sim.add_dft_fields(COMPS, freqs_dft, center=c, size=sz) for c, sz, _ in feed_faces]

t0 = time.time()
sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-7))
wall = time.time() - t0

freqs = np.array(mp.get_flux_freqs(flux[0]))
p_box = sum(sign * np.array(mp.get_fluxes(fl)) for fl, (_, _, sign) in zip(flux, faces))

# far field at f0: theta sweep at phi = 0, and for the Hertzian dipole also at 45 deg (its azimuthal-symmetry check)
R = 1000.0
theta = np.linspace(0, np.pi, a.ntheta)
u = {}
for phi in ((0.0, np.pi / 4) if a.mode == "hertzian" else (0.0,)):
    vals = []
    for th in theta:
        rhat = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
        ff = np.array(sim.get_farfield(n2f, mp.Vector3(*(R * rhat))))
        e, h = ff[0:3], ff[3:6]
        vals.append(R ** 2 * np.real(np.cross(np.conj(e), h)) @ rhat)
    u[phi] = np.array(vals)
u0 = u[0.0]
p_far = 2 * np.pi * np.trapezoid(u0 * np.sin(theta), theta)
out = {"mode": a.mode, "resolution": res, "cell": L, "voxels": int(round(L * res)) ** 3, "wall_s": round(wall, 1),
       "freq": freqs.tolist(), "p_box": p_box.tolist(), "f0": f0, "p_box_f0": float(np.interp(f0, freqs, p_box)),
       "theta": theta.tolist(), "u_phi0": u0.tolist(), "p_far_f0": float(p_far),
       "directivity": float(4 * np.pi * u0.max() / p_far)}
if np.pi / 4 in u:
    out["u_phi45"] = u[np.pi / 4].tolist()

if a.mode == "halfwave":
    def line_integral(dft, comp):
        w = np.squeeze(np.asarray(sim.get_array_metadata(dft_cell=dft)[3]))
        return np.array([np.sum(w * np.squeeze(sim.get_dft_array(dft, comp, k))) for k in range(a.nfreq)])
    def loop_current(m):
        return line_integral(m[0], mp.Hx) + line_integral(m[1], mp.Hy) - line_integral(m[2], mp.Hx) - line_integral(m[3], mp.Hy)

    def box_power():
        """Complex power out of the feed box: sum over faces of (E* x H) . n dA (MEEP DFT convention)."""
        s = np.zeros(a.nfreq, complex)
        for dft, (_, _, nrm) in zip(feed["box"], feed_faces):
            w = np.squeeze(np.asarray(sim.get_array_metadata(dft_cell=dft)[3]))
            for k in range(a.nfreq):
                F = [np.squeeze(sim.get_dft_array(dft, c, k)) for c in COMPS]
                cr = (np.conj(F[1]) * F[5] - np.conj(F[2]) * F[4], np.conj(F[2]) * F[3] - np.conj(F[0]) * F[5],
                      np.conj(F[0]) * F[4] - np.conj(F[1]) * F[3])
                s[k] += np.sum(w * sum(nc * c for nc, c in zip(nrm, cr)))
        return s

    current = loop_current(feed["loop2"])
    current_outer = loop_current(feed["loop4"])
    s_feed = box_power()
    z_in = s_feed / abs(current) ** 2
    p_box_dft = np.interp(freqs_dft, freqs, p_box)
    out.update({"radius": a.radius, "length": a.length, "gap": gap, "freq_dft": freqs_dft.tolist(),
                "z_re": z_in.real.tolist(), "z_im": z_in.imag.tolist(),
                # MEEP's DFT flux is Re(E* x H) with no factor 1/2, so the same convention holds for S and I
                "r_from_power": (p_box_dft / abs(current) ** 2).tolist(),
                "p_in": s_feed.real.tolist(), "p_box_dft": p_box_dft.tolist(),
                "current_loop_ratio": (abs(current_outer) / abs(current)).tolist()})
json.dump(out, open(a.out / "dipole.json", "w"), indent=1)
print(f"{a.mode}: resolution {res}, {out['voxels']:,} voxels, {wall:.0f} s, D = {out['directivity']:.4f}")

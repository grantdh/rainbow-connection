"""Full-wave power split of the Si dipole with openEMS: flux through closed boxes, heat by region, q(x) along the antenna.

Uses the private geometry builder (workflows/nm-meander-dipole/sim/run_meander.py) unchanged: its build() is wrapped to add dumps,
so one piece of code defines the device. Added frequency-domain dumps:
  FluxA/FluxB   E and H on two closed 5-face boxes in air (the chuck closes the sixth face), for P_rad and its independence of the box;
  Evol/Jvol     E and conduction current J over the whole stack, raw values on the Yee edges.
Heat density is Re(J*.E)/2 from openEMS's own J, so lit runs need no conductivity map on the analysis side.

  fullwave_openems.py run --tag NAME [run_meander flags ...]       run through run_meander.main() (private results/ directory)
  fullwave_openems.py analyse --tag NAME                           balance + q(x), writes OUT/fullwave/NAME.json
"""
import argparse
import json
import sys

import numpy as np

import common as c

DEV = c.device()
DUMP_F_GHZ = DEV["dump_f_ghz"]
BOX_A, BOX_B = 5.0, 9.0                                    # flux boxes, mm outside the chip (and above it)
RESULTS = c.WF / "results"


def add_dumps(CSX, t_box_mm, t_sub_mm, t_sheet):
    fl = [f * 1e9 for f in DUMP_F_GHZ]
    x0, x1, y0, y1 = DEV["stack"]["sample_mm"]
    ztop = -t_box_mm
    zbot = ztop - t_sub_mm
    def box(name, typ, a, b, mode=2):
        d = CSX.AddDump(name, dump_type=typ, dump_mode=mode, file_type=1, frequency=fl)
        d.AddBox(a, b)
    for nm, m in (("A", BOX_A), ("B", BOX_B)):
        X0, X1, Y0, Y1, Z = x0 - m, x1 + m, y0 - m, y1 + m, t_sheet + m
        faces = {"xm": ([X0, Y0, zbot], [X0, Y1, Z]), "xp": ([X1, Y0, zbot], [X1, Y1, Z]),
                 "ym": ([X0, Y0, zbot], [X1, Y0, Z]), "yp": ([X0, Y1, zbot], [X1, Y1, Z]), "zp": ([X0, Y0, Z], [X1, Y1, Z])}
        for f, (a, b) in faces.items():
            box(f"E{nm}_{f}", 10, a, b, mode=1)
            box(f"H{nm}_{f}", 11, a, b, mode=1)
    # One volume over the whole stack, raw values on the Yee edges (dump_mode 0): J E times an edge's whole dual volume is that edge's G V^2, so the sum is the discrete Joule
    # heat exactly. The box runs a little past the chip so that edges on its faces keep their whole dual volume. Cell-centred values (mode 2) overcounted the heat by 5 %.
    m = 0.3
    box("Evol", 10, [x0 - m, y0 - m, zbot], [x1 + m, y1 + m, t_sheet + 0.05], mode=0)
    box("Jvol", 12, [x0 - m, y0 - m, zbot], [x1 + m, y1 + m, t_sheet + 0.05], mode=0)


def run(argv):
    sys.path.insert(0, str(c.WF / "sim"))
    sys.path.insert(0, str(c.WF / "materials"))
    import run_meander as rm
    orig = rm.build

    def build(args):
        out = orig(args)
        CSX = out[1]
        t_box = max(args.box_cell_um, args.t_box_um) * 1e-3
        add_dumps(CSX, t_box, args.t_sub, rm.T_SHEET)
        return out
    rm.build = build
    sys.argv = ["run_meander.py", *argv, "--dump-f", ",".join(map(str, DUMP_F_GHZ))]
    rm.main()


def h5_field(path):
    """(x, y, z mesh in m, complex fields [nf] of shape (3, nx, ny, nz), frequencies) from an openEMS FD hdf5 dump."""
    import h5py
    with h5py.File(path, "r") as h:
        mesh = [np.asarray(h["Mesh"][a]) for a in "xyz"]
        fd = h["FieldData/FD"]
        keys = sorted((k for k in fd if k.startswith("f")), key=lambda k: int(k[1:]))
        return mesh, [fd[k][()] for k in keys], [float(np.ravel(fd[k].attrs["frequency"])[0]) for k in keys]


def edge_volumes(lines):
    """Dual volume of every Yee edge, per component, shape (3, nx, ny, nz), for raw (dump_mode 0) fields indexed by the node at the edge's start."""
    prim = [np.diff(l, append=l[-1]) for l in lines]                  # primary length from node i to i+1 (zero past the last line)
    dual = []
    for l in lines:
        d = np.diff(l)
        dual.append(0.5 * (np.r_[0.0, d] + np.r_[d, 0.0]))            # half of each neighbouring cell; one-sided at the dump's faces
    n = [l.size for l in lines]
    out = np.empty((3, *n))
    for c_ in range(3):
        f = [dual[a] if a != c_ else prim[a] for a in range(3)]
        out[c_] = f[0][:, None, None] * f[1][None, :, None] * f[2][None, None, :]
    return out


def heat(me, E, J, zr):
    """P = 1/2 sum over Yee edges with z in zr (m) of Re(J* E) V_dual (W at the run's port drive), with projections on x, y, z. openEMS's J holds no port resistor current:
    a run with the port's cells masked out gave the same sum to all printed digits."""
    p = 0.5 * np.real(np.conj(J) * E) * edge_volumes(me)
    keep = np.zeros(p.shape[1:], bool)
    keep[:, :, (me[2] >= zr[0]) & (me[2] <= zr[1])] = True
    d = p.sum(axis=0) * keep
    proj = {"xyz"[k]: (me[k].tolist(), d.sum(axis=tuple(a for a in range(3) if a != k)).tolist()) for k in range(3)}
    return float(d.sum()), proj


def sensitivity(me, E, chip_m, z_top, V):
    """First-order dY per unit dsigma of the handle: (1/V^2) sum E.E dV (no conjugate), total, by depth and by x. Only the handle's share of each edge's dual volume counts."""
    ix = (me[0] >= chip_m[0] - 1e-9) & (me[0] <= chip_m[1] + 1e-9)
    iy = (me[1] >= chip_m[2] - 1e-9) & (me[1] <= chip_m[3] + 1e-9)
    iz = me[2] <= z_top + 1e-9
    lines = [me[0][ix], me[1][iy], me[2][iz]]
    e = E[np.ix_(np.ones(3, bool), ix, iy, iz)]
    ee = ((e**2) * edge_volumes(lines)).sum(axis=0) / V**2
    return lines[2], ee.sum(axis=(0, 1)), ee.sum(axis=(1, 2)), lines[0]


def oxide_e2(me, E, z_top):
    """Int |E|^2 dV in the oxide: its z edges whole, the tangential edges on its two faces with the oxide's half of their dual volume."""
    iz = (me[2] >= z_top - 1e-9) & (me[2] <= 1e-9)
    lines = [me[0], me[1], me[2][iz]]
    return float((np.abs(E[np.ix_(np.ones(3, bool), np.ones(me[0].size, bool), np.ones(me[1].size, bool), iz)]) ** 2 * edge_volumes(lines)).sum())


def probe_column(me, E, x_mm, y_mm):
    """E along z at one (x, y): the transmit field that reciprocity turns into the port voltage from a current source there."""
    ix, iy = int(np.argmin(abs(me[0] * 1e3 - x_mm))), int(np.argmin(abs(me[1] * 1e3 - y_mm)))
    return {"x_mm": float(me[0][ix] * 1e3), "y_mm": float(me[1][iy] * 1e3), "z_m": me[2].tolist(), "Ez": {"re": E[2, ix, iy].real.tolist(), "im": E[2, ix, iy].imag.tolist()}}


def flux(sim, nm, i):
    """Outward Poynting power through the 5 faces of box nm: node-interpolated E and H on each face, trapezoid in area."""
    total, per = 0.0, {}
    for f, ax, sign in (("xm", 0, -1), ("xp", 0, 1), ("ym", 1, -1), ("yp", 1, 1), ("zp", 2, 1)):
        me, E, _ = h5_field(sim / f"E{nm}_{f}.h5")
        _, H, _ = h5_field(sim / f"H{nm}_{f}.h5")
        s = sign * 0.5 * np.real(np.cross(E[i], np.conj(H[i]), axis=0))[ax]
        for k in (2, 1, 0):                                  # integrate over the two in-plane axes; the normal axis has length 1
            s = np.trapezoid(s, me[k], axis=k) if me[k].size > 1 else np.take(s, 0, axis=k)
        per[f] = float(s)
        total += float(s)
    return total, per


def analyse(tag):
    c.require_private_repo()
    res = RESULTS / tag
    sim = res / "sim"
    port = np.load(res / "port_at_dump.npz")
    meta = json.load(open(res / "meta.json"))
    args = meta["args"]
    t_box = max(args["box_cell_um"], args["t_box_um"]) * 1e-6
    z_top, z_bot = -t_box, -t_box - args["t_sub"] * 1e-3
    chip = [v * 1e-3 for v in DEV["stack"]["sample_mm"]]
    me, E, _ = h5_field(sim / "Evol.h5")
    _, J, _ = h5_field(sim / "Jvol.h5")
    out = {"tag": tag, "meta_args": args, "freqs": {}}
    cplx = lambda a: {"re": np.real(a).tolist(), "im": np.imag(a).tolist()}
    for i, f in enumerate(port["f"]):
        V, I = port["V"][i], port["I"][i]
        pin = 0.5 * float(np.real(V * np.conj(I)))
        ph, proj = heat(me, E[i], J[i], (z_bot - 1e-9, z_top + 1e-9))
        ps, projs = heat(me, E[i], J[i], (-1e-9, me[2].max()))
        fa, per_a = flux(sim, "A", i)
        fb, per_b = flux(sim, "B", i)
        zh, s_z, s_x, xh = sensitivity(me, E[i], chip, z_top, V)
        rec = {"probe_column": probe_column(me, E[i], DEV["reciprocity"]["x_mm"], DEV["reciprocity"]["y_mm"]), "V": [float(V.real), float(V.imag)], "I": [float(I.real), float(I.imag)],
               "sens_handle": {"z_m": zh.tolist(), "by_z": cplx(s_z), "x_m": xh.tolist(), "by_x": cplx(s_x)},
               "Y_S": [float((I / V).real), float((I / V).imag)], "P_in": pin, "P_handle": ph, "P_sheet": ps, "P_rad_A": fa, "P_rad_B": fb, "int_E2_dV_box": oxide_e2(me, E[i], z_top),
               "flux_faces_A": per_a, "balance": (ph + ps + fa) / pin - 1,
               "q_handle_x": proj["x"], "q_handle_y": proj["y"], "q_handle_z": proj["z"], "q_sheet_x": projs["x"], "q_sheet_y": projs["y"]}
        out["freqs"][f"{f/1e9:g}"] = rec
        print(f"{f/1e9:7.2f} GHz  Y {I/V*1e3:.4f} mS  P_rad/P_in {fa/pin:.5f} (box B {fb/pin:.5f})  handle {ph/pin:.4f}  sheet {ps/pin:.5f}  balance {rec['balance']:+.5f}")
    (c.OUT / "fullwave").mkdir(exist_ok=True, parents=True)
    json.dump(out, open(c.OUT / "fullwave" / f"{tag}.json", "w"))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--tag", required=True)
        a = ap.parse_args()
        analyse(a.tag)

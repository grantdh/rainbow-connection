"""Lorentz reciprocity and the Kirchhoff/Nyquist form of it, on the real stack, with a second lumped port in the handle.

Two runs on the 25 um mesh: port 1 (feed) driven with port 2 passive, then port 2 driven with port 1 passive. S21 must equal S12.
Then the transmit field of a run without port 2 must predict Z21: reciprocity gives V_port1 = -(Int E.dl / I_1) I_2 for a current source I_2 at r in the handle.
That is the statement behind Kirchhoff here: thermal currents 4 k T sigma dV at r reach the port with exactly the weight of the transmit heat map at r.
  reciprocity_openems.py run 1|2|0         one direction, or 0: the transmit field on the same mesh with no port in the handle (through the private run_meander builder)
  reciprocity_openems.py analyse           S21, S12, Z21 vs the transmit field; writes OUT/reciprocity.json
  reciprocity_openems.py queue             the three runs that are not on disk yet (yield mode), then analyse
"""
import json
import sys

import h5py
import numpy as np

import common as c

R = c.device()["reciprocity"]
DUT_X, DUT_Y, D = R["x_mm"], R["y_mm"], R["half_width_mm"]
Z_TOP, Z_BOT = R["z_top_mm"], R["z_bot_mm"]               # mm below the handle's top
F = np.array([10e9, 31.5e9])
SIGMA_HANDLE = R["sigma_handle_S_m"]
LOOP_AREA = R["loop_area_m2"]                             # m^2: the H loop on the dual grid around the port's edges
LENGTH = R["length_m"]                                    # m: port length
BASE = c.device()["fullwave_runs"][R["run_base"]]
STACK = c.device()["stack"]


def done(which):
    res = c.WF / "results" / f"ps_recip_{which}"
    return (res / "sim/port_ut_1").exists() and (res / ("sim/Ecol.h5" if which == 0 else "port2.npz")).exists()


def run(which):
    sys.path.insert(0, str(c.WF / "sim"))
    sys.path.insert(0, str(c.WF / "materials"))
    import run_meander as rm
    port2 = []

    class Driven(rm.openEMS):                      # the builder calls AddLumpedPort for the feed; choose here which port is driven
        def AddLumpedPort(self, port_nr, R, start, stop, p_dir, excite=0, **kw):
            return super().AddLumpedPort(port_nr, R, start, stop, p_dir, excite=(1.0 if port_nr == max(which, 1) else 0.0), **kw)
    rm.openEMS = Driven
    orig_build = rm.build

    def build(args):
        out = orig_build(args)
        t_box = args.box_cell_um * 1e-3
        grid = out[1].GetGrid()

        def on_line(axis, v):
            """A value on the mesh: an existing line if one is within 2 um (a new line beside it would leave a sliver cell and an unstable run)."""
            lines = np.asarray(grid.GetLines(axis))
            near = lines[np.argmin(abs(lines - v))]
            if abs(near - v) < 2e-3:
                return float(near)
            grid.AddLine(axis, v)
            return v
        zs = [on_line("z", -t_box - Z_TOP), on_line("z", -t_box - Z_BOT)]
        xs = [on_line("x", DUT_X - D), on_line("x", DUT_X + D)]
        ys = [on_line("y", DUT_Y - D), on_line("y", DUT_Y + D)]
        if which == 0:                             # transmit field on the same mesh, no port in the handle
            d = out[1].AddDump("Ecol", dump_type=10, dump_mode=0, file_type=1, frequency=list(F))
            d.AddBox([xs[0], ys[0], zs[1]], [xs[1], ys[1], zs[0]])
        else:
            port2.append(out[0].AddLumpedPort(2, 50, [xs[0], ys[0], zs[1]], [xs[1], ys[1], zs[0]], "z", priority=30))
        return out
    rm.build = build
    tag = f"ps_recip_{which}"
    sys.argv = ["run_meander.py", *c.replay(BASE, tag)]
    rm.main()
    sim = c.WF / "results" / tag / "sim"
    if which == 0:
        return
    f = np.linspace(6e9, 35e9, 291)
    port2[0].CalcPort(str(sim), f)
    np.savez(c.WF / "results" / tag / "port2.npz", f=f, V=port2[0].uf_tot, I=port2[0].if_tot)


def dft(t, v, f):
    """The port records' spectrum, in the convention of openEMS's CalcPort."""
    return np.array([np.sum(v * np.exp(-2j * np.pi * fi * t)) * (t[1] - t[0]) for fi in np.atleast_1d(f)]) * 2


def analyse():
    """S21 against S12 from the two runs; then Z12 from the run driven at port 2 against the transmit field of the port-less run."""
    runs = {}
    for w in (1, 2):
        d = c.WF / "results" / f"ps_recip_{w}"
        p2 = np.load(d / "port2.npz")
        ut, it = np.loadtxt(d / "sim/port_ut_1", comments="%"), np.loadtxt(d / "sim/port_it_1", comments="%")
        runs[w] = dict(V1=dft(ut[:, 0], ut[:, 1], p2["f"]), I1=dft(it[:, 0], it[:, 1], p2["f"]), V2=p2["V"], I2=p2["I"], f=p2["f"])
    f = runs[1]["f"]
    s21 = 0.5 * (runs[1]["V2"] - 50 * runs[1]["I2"]) / (0.5 * (runs[1]["V1"] + 50 * runs[1]["I1"]))
    s12 = 0.5 * (runs[2]["V1"] - 50 * runs[2]["I1"]) / (0.5 * (runs[2]["V2"] + 50 * runs[2]["I2"]))
    pick = [int(np.argmin(abs(f - x))) for x in F]
    rel = np.abs(s21 / s12 - 1)
    out = {"f": F.tolist(), "max_rel_S21_S12_6_35GHz": float(rel.max()), "rel_S21_S12": rel[pick].tolist(), "S21": [[float(s21[i].real), float(s21[i].imag)] for i in pick]}
    r0 = c.WF / "results" / "ps_recip_0"
    with h5py.File(r0 / "sim/Ecol.h5", "r") as h:
        z = np.asarray(h["Mesh/z"])
        fd = h["FieldData/FD"]
        ez = {j: fd[f"f{j}"][()][2].mean(axis=(0, 1)) for j in range(len(F))}          # E_z at each height, averaged over the port's cross-section
    ut, it = np.loadtxt(r0 / "sim/port_ut_1", comments="%"), np.loadtxt(r0 / "sim/port_it_1", comments="%")
    pred = []
    for j, (i, fk) in enumerate(zip(pick, F)):
        I1, V1 = dft(it[:, 0], it[:, 1], fk)[0], dft(ut[:, 0], ut[:, 1], fk)[0]
        z_pred = -np.sum(ez[j][:-1] * np.diff(z)) / I1                                  # raw E_z[k] is the edge from z[k] to z[k+1]
        # Mutual impedance from the run driven at port 2. Its current probe is a loop of H around the port's edges, so it also holds the handle's own conduction and displacement current through
        # that area, which flows against the source. The impressed current is the probe's plus that part. V1 = Z11 I1 + Z12 I_s avoids port 2's reference impedance altogether.
        r2 = runs[2]
        open_v = r2["V1"][i] - V1 / I1 * r2["I1"][i]
        y_si = SIGMA_HANDLE + 1j * 2 * np.pi * fk * 8.8541878128e-12 * STACK["eps_si"]
        z12 = open_v / (r2["I2"][i] + y_si * LOOP_AREA / LENGTH * r2["V2"][i])
        z12_raw = open_v / r2["I2"][i]
        pred.append({"f": f"{fk/1e9:g}", "Z12_from_port_2_drive": [float(z12.real), float(z12.imag)], "Z12_without_handle_current_in_loop": [float(z12_raw.real), float(z12_raw.imag)],
                     "Z21_from_transmit_field": [float(z_pred.real), float(z_pred.imag)], "rel": float(abs(z_pred / z12 - 1))})
    out["kirchhoff_form"] = pred
    json.dump(out, open(c.OUT / "reciprocity.json", "w"), indent=1)
    print(json.dumps(out, indent=0)[:1500])


def queue():
    for which in (1, 2, 0):
        if not done(which):
            tag, log = f"ps_recip_{which}", c.OUT / f"recip{which}.log"
            if c.run_solver([sys.executable, __file__, "run", str(which)], c.HERE, log, lock="yield") == 0:
                c.record_stop(BASE, tag, log)
    analyse()


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2])), "analyse": analyse, "queue": queue}[sys.argv[1]]()

"""FEM cross-check of the dark dipole with Palace (Nedelec elements, frequency domain): a different method class from FDTD.

Same stack, chuck and port as the openEMS run, built from the same geometry file, except that the oxide is resolved at its true thickness and the pads
lie directly on it. Handle: lossy dielectric. Outer air faces: second-order absorbing boundary. Chuck: PEC. Membrane: a sheet impedance from the vdP value.

Heat by region follows from Palace's domain energy, E_elec = 1/2 Int Re(eps) |E|^2 with RMS phasors, so P_J = 2 (sigma/eps) E_elec. P_rad is the power flux through the outer
absorbing surface. P_in = |V|^2 Re(Y) from the port. The balance P_in = P_J + P_rad is then a check on the solve, as it is for the FDTD run.

  fem_palace.py mesh --name N --lc-pad 0.015 [--arms]        gmsh mesh into OUT/fem/N/
  fem_palace.py solve --name N --rho 10 --order 2 --f 10,31.5     Palace run
  fem_palace.py read --name N --rho 10                         Y, P_rad, P_J, balance
"""
import argparse
import csv
import json
import subprocess

import numpy as np

import common as c

PALACE = c.HOME / "opt/palace/bin/palace"
GMSH_PY = c.HOME / ".venvs/cptopo/bin/python"
EPS0 = 8.8541878128e-12
STACK = c.device()["stack"]
EPS_SI, EPS_BOX, TAN_BOX = STACK["eps_si"], STACK["eps_box"], STACK["tan_box"]
SAMPLE = STACK["sample_mm"]

GMSH = r"""
import json, sys
import numpy as np
import gmsh
g = json.load(open(sys.argv[1])); lc_pad = float(sys.argv[2]); arms = int(sys.argv[3]); out = sys.argv[4]
T_BOX, T_SUB, MX, MT = %(T_BOX)r, %(T_SUB)r, %(MX)r, %(MT)r
x0, x1, y0, y1 = %(SAMPLE)r
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 1); gmsh.model.add("dip")
occ = gmsh.model.occ
zt, zb = -T_BOX, -T_BOX - T_SUB
air = occ.addBox(x0 - MX, y0 - MX, zb, x1 - x0 + 2 * MX, y1 - y0 + 2 * MX, MT - zb)
box = occ.addBox(x0, y0, zt, x1 - x0, y1 - y0, T_BOX)
han = occ.addBox(x0, y0, zb, x1 - x0, y1 - y0, T_SUB)
pads = g["pads"]; gx = pads[1][0]; py = pads[0][3]
surf = []
for xa, xb, ya, yb in pads:
    surf.append(occ.addRectangle(xa, ya, 0.0, xb - xa, yb - ya))
port = occ.addRectangle(-gx, 0.0, 0.0, 2 * gx, py)
surf.append(port)
if arms:
    rects = [occ.addRectangle(a, c, 0.0, b - a, d - c) for a, b, c, d in g["si"]]
    surf += rects
out_map, _ = occ.fragment([(3, air), (3, box), (3, han)], [(2, s) for s in surf])
occ.synchronize()
def cls_vol(tag):
    b = gmsh.model.getBoundingBox(3, tag)
    if b[5] <= 1e-6 and b[2] >= zt - 1e-6: return "box"
    if b[5] <= zt + 1e-6 and b[0] >= x0 - 1e-6 and b[3] <= x1 + 1e-6: return "handle"
    return "air"
vols = {"air": [], "box": [], "handle": []}
for _, t in gmsh.model.getEntities(3):
    vols[cls_vol(t)].append(t)
def inrect(cx, cy, r): return r[0] - 1e-6 <= cx <= r[1] + 1e-6 and r[2] - 1e-6 <= cy <= r[3] + 1e-6
sel = {"pec": [], "port": [], "abs": [], "sheet": []}
XA, XB, YA, YB, ZT = x0 - MX, x1 + MX, y0 - MX, y1 + MX, MT
for _, t in gmsh.model.getEntities(2):
    b = gmsh.model.getBoundingBox(2, t)
    cx, cy = gmsh.model.occ.getCenterOfMass(2, t)[:2]
    flat = abs(b[5] - b[2]) < 1e-5
    if flat and abs(0.5 * (b[2] + b[5]) - zb) < 1e-5: sel["pec"].append(t); continue
    if flat and abs(0.5 * (b[2] + b[5]) - ZT) < 1e-5: sel["abs"].append(t); continue
    if abs(b[3] - b[0]) < 1e-5 and (abs(b[0] - XA) < 1e-5 or abs(b[0] - XB) < 1e-5): sel["abs"].append(t); continue
    if abs(b[4] - b[1]) < 1e-5 and (abs(b[1] - YA) < 1e-5 or abs(b[1] - YB) < 1e-5): sel["abs"].append(t); continue
    if flat and abs(0.5 * (b[2] + b[5])) < 1e-6:
        if any(inrect(cx, cy, r) for r in pads): sel["pec"].append(t)
        elif inrect(cx, cy, (-gx, gx, 0.0, py)): sel["port"].append(t)
        elif arms and any(inrect(cx, cy, r) for r in g["si"]): sel["sheet"].append(t)
for i, (n, ts) in enumerate((("air", vols["air"]), ("box", vols["box"]), ("handle", vols["handle"]))):
    gmsh.model.addPhysicalGroup(3, ts, i + 1); gmsh.model.setPhysicalName(3, i + 1, n)
for i, n in enumerate(("pec", "port", "abs", "sheet")):
    if sel[n]:
        gmsh.model.addPhysicalGroup(2, sel[n], 10 + i); gmsh.model.setPhysicalName(2, 10 + i, n)
print({k: len(v) for k, v in sel.items()}, {k: len(v) for k, v in vols.items()})
# size field: fine at the feed, graded to the wavelength in Si and air
f = gmsh.model.mesh.field
d = f.add("Distance"); f.setNumbers(d, "CurvesList", [abs(c[1]) for t in sel["pec"] + sel["port"] for c in gmsh.model.getBoundary([(2, t)], oriented=False)
                                                      if abs(gmsh.model.getBoundingBox(2, t)[2]) < 1e-6]); f.setNumber(d, "Sampling", 40)
th = f.add("Threshold"); f.setNumber(th, "InField", d); f.setNumber(th, "SizeMin", lc_pad); f.setNumber(th, "SizeMax", 1.6); f.setNumber(th, "DistMin", 0.06); f.setNumber(th, "DistMax", 9.0)
bx = f.add("Box"); f.setNumber(bx, "VIn", 0.35); f.setNumber(bx, "VOut", 1.6)
f.setNumber(bx, "XMin", x0); f.setNumber(bx, "XMax", x1); f.setNumber(bx, "YMin", y0); f.setNumber(bx, "YMax", y1); f.setNumber(bx, "ZMin", zb); f.setNumber(bx, "ZMax", 0.5)
mn = f.add("Min"); f.setNumbers(mn, "FieldsList", [th, bx]); f.setAsBackgroundMesh(mn)
for o in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
    gmsh.option.setNumber(o, 0)
gmsh.option.setNumber("Mesh.Algorithm3D", 10); gmsh.option.setNumber("Mesh.Optimize", 1); gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(out)
print("tets", sum(len(x) for x in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()
""" % dict(T_BOX=STACK["t_box_mm"], T_SUB=STACK["t_handle_mm"], MX=STACK["margin_xy_mm"], MT=STACK["margin_top_mm"], SAMPLE=SAMPLE)


def fdir(name):
    d = c.OUT / "fem" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def mesh(a):
    d = fdir(a.name)
    (d / "mesh.py").write_text(GMSH)
    geo = c.WF / "geometry/meander_geometry.json"
    subprocess.run([str(GMSH_PY), "mesh.py", str(geo), str(a.lc_pad), str(int(a.arms)), "dip.msh"], cwd=d, check=True)


def config(a):
    sigma = 100.0 / a.rho
    bc = {"PEC": {"Attributes": [10]}, "Absorbing": {"Attributes": [12], "Order": 2},
          "LumpedPort": [{"Index": 1, "R": 50.0, "Excitation": True, "Attributes": [11], "Direction": "+X"}],
          "Postprocessing": {"SurfaceFlux": [{"Index": 1, "Attributes": [12], "Type": "Power", "Center": [0.5 * (SAMPLE[0] + SAMPLE[1]), 0.5 * (SAMPLE[2] + SAMPLE[3]), -1.0]}]}}
    if a.arms:
        bc["Impedance"] = [{"Attributes": [13], "Rs": a.rs_sheet}]
    samples = [{"Type": "Point", "Freq": [float(x) for x in a.f.split(",")], "SaveStep": 0}]
    return {"Problem": {"Type": "Driven", "Verbose": 2, "Output": f"post_r{a.rho:g}_o{a.order}"},
            "Model": {"Mesh": "dip.msh", "L0": 1e-3},
            "Domains": {"Materials": [{"Attributes": [1], "Permeability": 1.0, "Permittivity": 1.0},
                                      {"Attributes": [2], "Permeability": 1.0, "Permittivity": EPS_BOX, "LossTan": TAN_BOX},
                                      {"Attributes": [3], "Permeability": 1.0, "Permittivity": EPS_SI, "Conductivity": sigma}],
                        "Postprocessing": {"Energy": [{"Index": 1, "Attributes": [1]}, {"Index": 2, "Attributes": [2]}, {"Index": 3, "Attributes": [3]}]}},
            "Boundaries": bc,
            "Solver": {"Order": a.order, "Device": "CPU", "Driven": {"Samples": samples},
                       "Linear": {"Type": "Default", "KSPType": "GMRES", "Tol": 1e-7, "MaxIts": 400, "ComplexCoarseSolve": True}}}


def solve(a):
    d = fdir(a.name)
    cf = d / f"cfg_r{a.rho:g}_o{a.order}.json"
    cf.write_text(json.dumps(config(a), indent=1))
    rc = c.run_solver([PALACE, "-np", str(a.np), cf.name], d, d / f"palace_r{a.rho:g}_o{a.order}.log", lock="yield", env={"OMP_NUM_THREADS": "1"})
    print("palace rc", rc)
    return rc


def table(path):
    rows = list(csv.reader(open(path)))
    head = [h.strip() for h in rows[0]]
    return head, np.array([[float(x) for x in r] for r in rows[1:]])


def results(name, rho, order):
    """Y, radiated fraction, Joule fractions and balance per frequency, from Palace's CSV output."""
    d = fdir(name) / f"post_r{rho:g}_o{order}"
    sh, sp = table(d / "port-S.csv")
    eh, en = table(d / "domain-E.csv")
    fh, fl = table(d / "surface-F.csv")
    vh, vo = table(d / "port-V.csv")
    col = lambda head, pat: next(i for i, h in enumerate(head) if pat in h)
    sigma = 100.0 / rho
    out = {}
    for i, f in enumerate(sp[:, 0]):
        mag, arg = sp[i, col(sh, "|S[1][1]|")], sp[i, col(sh, "arg(S[1][1])")]
        s11 = 10 ** (mag / 20) * np.exp(1j * np.radians(arg))
        y = (1 - s11) / (1 + s11) / 50.0
        v = vo[i, col(vh, "Re{V[1]}")] + 1j * vo[i, col(vh, "Im{V[1]}")]
        # Palace's phasors are RMS-normalised (V_inc = 7.07 V for 1 W incident on 50 ohm): its flux Re{E x H*} carries no factor 1/2, and its E_elec = 1/2 Re(eps) |E|^2 is the time average
        pin = abs(v) ** 2 * y.real
        e_handle = en[i, col(eh, "E_elec[3]")]
        e_box = en[i, col(eh, "E_elec[2]")]
        w = 2 * np.pi * f * 1e9
        p_h = 2 * sigma / (EPS0 * EPS_SI) * e_handle
        p_b = 2 * w * TAN_BOX * e_box
        p_r = fl[i, 1]
        out[f"{f:g}"] = {"Y_S": [float(y.real), float(y.imag)], "P_in_W": float(pin), "rad": float(p_r / pin), "handle": float(p_h / pin), "box": float(p_b / pin), "balance": float((p_r + p_h + p_b) / pin - 1)}
    return out


def read(a):
    r = results(a.name, a.rho, a.order)
    for f, row in r.items():
        print(f, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mesh", "solve", "read"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--lc-pad", type=float, default=0.02)
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--rs-sheet", type=float, default=STACK["rs_sheet_ohm_sq"])
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--f", default="10")
    ap.add_argument("--np", type=int, default=6)
    a = ap.parse_args()
    c.require_private_repo()
    {"mesh": mesh, "solve": solve, "read": read}[a.cmd](a)


if __name__ == "__main__":
    main()

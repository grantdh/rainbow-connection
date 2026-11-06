"""Transient and steady T(r) of the chip under the laser, with Elmer: kappa(T), cp(T), the actual sweep timeline, a Gaussian surface source.

Quarter-symmetric chip, hexahedra graded toward the beam centre and the top. Boundary conditions: absorbed Gaussian flux and Robin loss on top,
Robin contact to the chuck below (h = kappa_air/gap), adiabatic on the symmetry planes, Robin on the outer sides. Light is absorbed within 1/alpha ~ 3 um
of the handle's top, far below the element size, so it enters as a surface flux. The oxide's drop is the closed-form term in 01_thermal_closed_form.

  thermal_elmer.py [--steady-only | --summary]    every case in device.json; writes OUT/thermal/<case>/ and prereg/06_thermal_elmer.json
Checks: boundary heat flux = absorbed power; steady T vs the closed form.
"""
import shutil
import subprocess
import sys

import numpy as np
from scipy.special import erf

import common as c

ELMER = c.HOME / "elmer/bin"
GMSH_PY = c.HOME / ".venvs/cptopo/bin/python"

GMSH = """
import gmsh, sys
a, b, t = {a}, {b}, {t}
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("chip")
box = gmsh.model.occ.addBox(0, 0, 0, a, b, t)
gmsh.model.occ.synchronize()
for dim, tag in gmsh.model.getEntities(1):
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, tag)
    if abs(z1 - z0) > 1e-5:
        d = gmsh.model.getBoundary([(1, tag)], oriented=True)
        p0 = gmsh.model.getValue(0, abs(d[0][1]), [])
        pr = 1.15 if p0[2] > 0.5 * t else 1 / 1.15
        gmsh.model.mesh.setTransfiniteCurve(tag, {nz}, "Progression", pr)
    else:
        d = gmsh.model.getBoundary([(1, tag)], oriented=True)
        p0 = gmsh.model.getValue(0, abs(d[0][1]), [])
        p1 = gmsh.model.getValue(0, abs(d[1][1]), [])
        along_x = abs(x1 - x0) > 1e-5
        start = p0[0] if along_x else p0[1]
        end = p1[0] if along_x else p1[1]
        pr = 1.06 if start < end else 1 / 1.06
        gmsh.model.mesh.setTransfiniteCurve(tag, {nxy}, "Progression", pr)
for dim, tag in gmsh.model.getEntities(2):
    gmsh.model.mesh.setTransfiniteSurface(tag); gmsh.model.mesh.setRecombine(2, tag)
gmsh.model.mesh.setTransfiniteVolume(box)
names = dict()
for dim, tag in gmsh.model.getEntities(2):
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tag)
    if abs(z1 - z0) < 1e-5: names.setdefault("top" if z0 > 0.5 * t else "bottom", []).append(tag)
    elif abs(x1 - x0) < 1e-5: names.setdefault("symx" if x0 < 0.5 * a else "side", []).append(tag)
    else: names.setdefault("symy" if y0 < 0.5 * b else "side", []).append(tag)
for i, n in enumerate(("top", "bottom", "symx", "symy", "side")):
    gmsh.model.addPhysicalGroup(2, names[n], i + 1); gmsh.model.setPhysicalName(2, i + 1, n)
gmsh.model.addPhysicalGroup(3, [box], 1); gmsh.model.setPhysicalName(3, 1, "si")
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("chip.msh"); gmsh.finalize()
"""

SIF = """
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "chip"
End
Simulation
  Coordinate System = Cartesian
  Simulation Type = {simtype}
  {time_block}
  Steady State Max Iterations = 30
  Output Intervals = 0
  Post File = case.vtu
  Max Output Level = 3
End
Constants
  Stefan Boltzmann = 5.67e-8
End
Body 1
  Equation = 1
  Material = 1
  Initial condition = 1
End
Initial Condition 1
  Temperature = {T0}
End
Equation 1
  Active Solvers(2) = 1 2
End
Material 1
  Density = 2329.0
  Heat Capacity = Variable Temperature
    Real MATC "712.0*{cpf}*(1.0 + 0.00062*(tx-300.0))"
  Heat Conductivity = Variable Temperature
    Real MATC "{kf}*148.0*(300.0/tx)^1.3"
End
Solver 1
  Equation = Heat Equation
  Procedure = "HeatSolve" "HeatSolver"
  Variable = Temperature
  Calculate Loads = True
  Stabilize = False
  Bubbles = False
  Nonlinear System Max Iterations = 20
  Nonlinear System Convergence Tolerance = 1e-7
  Nonlinear System Newton After Iterations = 3
  Nonlinear System Newton After Tolerance = 1e-3
  Nonlinear System Relaxation Factor = 1.0
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 2000
  Linear System Convergence Tolerance = 1e-10
  Linear System Preconditioning = ILU1
  Steady State Convergence Tolerance = 1e-7
End
Solver 2
  Exec Solver = After Timestep
  Procedure = "SaveData" "SaveScalars"
  Filename = scalars.dat
  Variable 1 = Temperature
  Operator 1 = max
  Variable 2 = Temperature
  Operator 2 = min
  Variable 3 = Temperature
  Operator 3 = mean
  Variable 4 = Temperature
  Operator 4 = boundary int
End
Boundary Condition 1
  Target Boundaries(1) = 1
  Name = "top"
  Heat Flux BC = True
  Heat Flux = Variable Time, Coordinate 1, Coordinate 2
    Real MATC "({power_chain})*exp(-2*(tx(1)*tx(1)+tx(2)*tx(2))/({w}*{w}))"
  Heat Transfer Coefficient = {htop}
  External Temperature = {T0}
End
Boundary Condition 2
  Target Boundaries(1) = 2
  Name = "bottom"
  Heat Flux BC = True
  Heat Transfer Coefficient = {hbot}
  External Temperature = {T0}
  Save Scalars = True
End
Boundary Condition 3
  Target Boundaries(1) = 5
  Name = "side"
  Heat Flux BC = True
  Heat Transfer Coefficient = {htop}
  External Temperature = {T0}
End
"""


def power_of_setting(k, law):
    return float(c.laser_power(np.array(float(k)), np.array(law == "amps")))


def chain(cs, tl):
    """Peak absorbed flux (W/m^2) vs time, as a MATC sum of steps; the setting of file i holds from file i-1's save time to its own."""
    terms, last = [], 0.0
    for i in range(1, len(tl)):
        k = tl[i][1]
        pk = 2 * power_of_setting(k, cs["law"]) * cs["train"] * cs["afrac"] / (np.pi * cs["w"] ** 2) if k else 0.0
        if pk != last:
            terms.append(f"(tx(0)>{tl[i-1][2]:.1f})*({pk - last:.6e})")
            last = pk
    return "+".join(terms)


def closed_form_steady(cs, P):
    area = cs["chip"] ** 2
    r = 1 / (1 / (cs["t"] / (148.0 * area) + cs["gap"] / (0.026 * area)) + cs["htop"] * area)
    frac = erf(np.sqrt(2) * 0.5 * cs["chip"] / cs["w"]) ** 2
    return r, P * frac


def run_case(name, cs, tl, modes=("transient", "steady")):
    d = c.OUT / "thermal" / name
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "chip").exists():
        half = cs["chip"] / 2
        (d / "mesh.py").write_text(GMSH.format(a=half, b=half, t=cs["t"], nz=17, nxy=33))
        subprocess.run([str(GMSH_PY), "mesh.py"], cwd=d, check=True)
        subprocess.run([str(ELMER / "ElmerGrid"), "14", "2", "chip.msh", "-autoclean", "-out", "chip"], cwd=d, check=True, capture_output=True)
    hbot = 0.026 / cs["gap"]
    t_end = tl[-1][2]
    for mode in modes:
        sub = d / mode
        done = sub / "scalars.dat"
        if done.exists() and len(np.atleast_2d(np.loadtxt(done))) >= (int(t_end / 0.5) if mode == "transient" else 1):
            continue
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir()
        shutil.copytree(d / "chip", sub / "chip")
        if mode == "transient":
            tb = f"Timestepping Method = BDF\n  BDF Order = 2\n  Timestep Intervals(1) = {int(t_end / 0.5)}\n  Timestep Sizes(1) = 0.5"
            sim, pc = "Transient", chain(cs, tl)
        else:
            tb, sim = "", "Steady State"
            p7 = power_of_setting(7, cs["law"]) * cs["train"] * cs["afrac"]
            pc = f"{2 * p7 / (np.pi * cs['w']**2):.6e}"
        sif = SIF.format(power_chain=pc, w=cs["w"], simtype=sim, time_block=tb, T0=cs["T0"], cpf=cs["cpf"], kf=cs["kf"], htop=cs["htop"], hbot=hbot)
        (sub / "case.sif").write_text(sif)
        (sub / "ELMERSOLVER_STARTINFO").write_text("case.sif\n1\n")
        rc = c.run_solver([ELMER / "ElmerSolver", "case.sif"], sub, sub / "elmer.log", lock="yield")
        if rc != 0 or not (sub / "scalars.dat").exists():
            sys.exit(f"{name} {mode}: Elmer failed, rc {rc}: {(sub / 'elmer.log').read_text()[-600:]}")


def summarise(cases, tl):
    """Register the results: T(t) at each sweep, the steady rise against the closed form, and the heat through the chuck against the absorbed power."""
    out, gates = {"what": "Elmer T(r,t) on the sweep timeline, kappa(T), cp(T)"}, []
    for name, cs in cases.items():
        d = c.OUT / "thermal" / name
        sd = np.atleast_2d(np.loadtxt(d / "steady/scalars.dat"))[-1]
        area_q = (cs["chip"] / 2) ** 2
        tmax, tmin, tmean, int_bot = sd[:4]
        heat_bot = 4 * 0.026 / cs["gap"] * (int_bot - cs["T0"] * area_q)
        p7 = power_of_setting(7, cs["law"]) * cs["train"] * cs["afrac"]
        r, p_chip = closed_form_steady(cs, p7)
        heat_top = cs["htop"] * (tmean - cs["T0"]) * (4 * area_q + 4 * cs["chip"] * cs["t"])      # top face and the four sides, at the mean rise
        rec = {"inputs": cs, "P_abs_on_chip_W": p_chip, "T_max_K": tmax, "T_mean_K": tmean, "dT_mean_K": tmean - cs["T0"], "closed_form_dT_K": p_chip * r, "heat_through_chuck_W": heat_bot, "top_and_side_loss_W": heat_top}
        tr = d / "transient/scalars.dat"
        if tr.exists():
            t = np.atleast_2d(np.loadtxt(tr))
            at = {n: float(t[min(int(ts / 0.5) - 1, len(t) - 1), 2] - cs["T0"]) for n, _, ts in tl[1:]}
            off = next(ts for _, k, ts in tl[::-1] if k > 0)
            cool = t[int(off / 0.5):, 2] - cs["T0"]
            pk = t[int(off / 0.5) - 1, 2] - cs["T0"]
            below = np.where(cool < pk / np.e)[0]
            rec.update({"dT_mean_at_sweep_K": at, "cooling_1_over_e_s": float((below[0] + 1) * 0.5) if len(below) else None})
        out[name] = rec
        gates.append({"name": f"Elmer heat through chuck, top and sides = absorbed power, {name}", "value": (heat_bot + heat_top) / p_chip - 1, "limit": 0.01, "note": "steady, k = 7"})
        gates.append({"name": f"Elmer steady mean rise vs closed form, {name}", "value": (tmean - cs["T0"]) / (p_chip * r) - 1, "limit": 0.10, "note": "kappa(T), Gaussian source"})
    out["gates"] = gates
    c.register("06_thermal_elmer", out, code=[__file__, c.HERE / "common.py"])
    return out


def main():
    c.require_private_repo()
    tl = c.x_timeline()
    cases = c.device()["thermal_cases"]
    modes = ("steady",) if "--steady-only" in sys.argv else ("transient", "steady")
    if "--summary" not in sys.argv:
        for name, cs in cases.items():
            print("case", name, flush=True)
            run_case(name, cs, tl, modes)
    res = summarise(cases, tl)
    for n in cases:
        print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[n].items() if k in ("dT_mean_K", "closed_form_dT_K", "heat_through_chuck_W", "P_abs_on_chip_W", "cooling_1_over_e_s")})
    print(res["gates"])


if __name__ == "__main__":
    main()

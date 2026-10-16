#!/usr/bin/env python
"""Example 6: Metal FDTD kernels against the exact discrete dispersion relation of the Yee scheme.

A rectangular PEC box has analytic TM_mnp modes. On a Yee grid those modes remain exact eigenmodes,
but their frequencies obey the discrete dispersion relation

    (2/dt) sin(w dt/2) = c sqrt( sum_i [ (2/dx) sin(k_i dx/2) ]^2 ),   k_i = m_i pi / L_i ,

so a kernel is correct only if its resonances land on these discrete frequencies (slightly below the
continuum values). The example rings an empty 64 x 64 x 128 box (dx = 2.5 cm) for 32,000 steps with a
soft source (the programs' Gaussian-modulated sine at 300 MHz, envelope spread shortened from 3 to 0.5
periods so that modes from ~120 to ~480 MHz are driven above 10 %; centred at exactly 10 periods, so the
net injected charge vanishes) and estimates the ring-down lines (t > 90 ns, after the drive has ended) with a matrix-pencil
harmonic inversion, which resolves lines closer than an FFT bin and reports their damping:

  - loopfdtd        LoopFDTD/dipole_s11.swift, conductor removed (as coded)
  - loopfdtd_fixed  same, with a per-component E-field boundary guard (see GUARD_FIX)
  - meephistopheles MEEPhistopheles/src/metal/dipole_s11.swift, conductor removed (Ez/Hx/Hy only)
  - meep            MEEP 3D, one Yee step per sample (meep_dipole_box.py --empty)

What each engine must satisfy:
  meep             every line is an N*dx Yee mode (MEEP's metallic cell boundary is exactly that box)
  loopfdtd_fixed   every line is an (N-1)*dx Yee mode: the kernel holds tangential E to zero on the
                   outermost node planes, so the PEC walls sit one cell inside the nominal box
  meephistopheles  every line is a 2D (p = 0) TM mode of the (N-1)*dx box (Ez/Hx/Hy scheme)
  loopfdtd         as coded, the one guard line also clamps the NORMAL E component on the far half-planes
                   (Ex at i = Nx-1, Ey at j = Ny-1, Ez at k = Nz-1), so the box is not a Yee PEC cavity
                   of any size; its lines sit 0.5-1.5 % below the (N-1)*dx modes in a mode-dependent way.
                   This is reported as a finding, not gated.
The box is lossless: the isolated (3,1,0) line must be undamped and an undamped model of all poles must fit the
whole ring-down (a half-record rms ratio is not a loss measure: mode pairs 0.3 % apart beat over 2.4 us). Gates apply
to lines >= 10 % of the strongest (on synthetic 32,000-step Yee-mode signals with float32 round-off the pencil
places those to 5e-4 %); weaker lines down to 3 % are reported. Poles closer than 0.3 % are one cluster whose
gated line is the least-damped pole: the source sits half a cell off the box centre, so even-parity modes such as
(4,1,2) are weakly excited and appear as faint poles next to strong odd-parity lines, where the Hann amplitude
cannot separate them. Those twin poles are matched against the full-parity mode list and reported.

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-meep] [--steps N]   (~5 min, MEEP dominates)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import exlib  # noqa: E402

RES = HERE / "results"
LOOP = Path.home() / "Dev/Claude/LoopFDTD"
MEEPH = Path.home() / "Dev/Claude/MEEPhistopheles"
CROSS = HERE.parent / "metal-fdtd-crosscheck"
SWIFTC = ["swiftc", "-O", "-framework", "Metal", "-framework", "Foundation"]
NX, NY, NZ, DX = 64, 64, 128, 0.025
C_MEEP = 299792458.0
C_LOOP = 3.0e8                                     # LoopFDTD: c0 = 3e8, eps0 = 1/(mu0 c0^2) -> c = 3e8 exactly
C_MEEPH = 1.0 / np.sqrt(4e-7 * np.pi * 8.854e-12)  # MEEPhistopheles: eps0 literal 8.854e-12 -> 2.997956e8
F_BAND = (150e6, 650e6)
T_RINGDOWN = 90e-9                                 # drive: Gaussian centred at 10/f0 = 33 ns, spread 3/f0 = 10 ns; < 1e-13 of peak by 90 ns
AMP_THRESH = 0.03                                  # lines reported
SPREAD_PERIODS = 0.5                               # Gaussian envelope spread in the generated variants (as coded: 3 periods = 10 ns)
CLUSTER_PCT = 0.3                                  # poles closer than this are one cluster (distinct excited modes are >= 0.08 % apart; see README)
AMP_GATE = 0.10                                    # lines gated (weak lines near the report threshold carry ~0.03 % pencil error)
GUARD_FIX_OLD = re.compile(
    r"if \(i == 0 \|\| i >= Nx-1 \|\| j == 0 \|\| j >= Ny-1 \|\| k == 0 \|\| k >= Nz-1\) return;\s*\n"
    r"(?P<mid>.*?)"
    r"// Ex: dHz/dy - dHy/dz\s*\n\s*Ex\[idx\] \+= Cb \* \(\(Hz\[idx_jp\] - Hz\[idx\]\) - \(Hy\[idx_kp\] - Hy\[idx\]\)\);\s*\n"
    r"\s*// Ey: dHx/dz - dHz/dx\s*\n\s*Ey\[idx\] \+= Cb \* \(\(Hx\[idx_kp\] - Hx\[idx\]\) - \(Hz\[idx_ip\] - Hz\[idx\]\)\);\s*\n"
    r"\s*// Ez: dHy/dx - dHx/dy\s*\n\s*Ez\[idx\] \+= Cb \* \(\(Hy\[idx_ip\] - Hy\[idx\]\) - \(Hx\[idx_jp\] - Hx\[idx\]\)\);",
    re.S)
GUARD_FIX_NEW = """uint idx = i + j * Nx + k * Nx * Ny;

    if (isPEC[idx]) {
        Ex[idx] = 0.0;
        Ey[idx] = 0.0;
        Ez[idx] = 0.0;
        return;
    }

    // Per-component PEC guard: a component is held at zero only on the walls it is tangential to
    // (and where its curl neighbours would fall outside the grid). The normal component on the far
    // half-plane is updated like any interior node.
    if (!(j == 0 || j >= Ny-1 || k == 0 || k >= Nz-1)) {
        uint idx_jp = i + (j+1) * Nx + k * Nx * Ny;
        uint idx_kp = i + j * Nx + (k+1) * Nx * Ny;
        Ex[idx] += Cb * ((Hz[idx_jp] - Hz[idx]) - (Hy[idx_kp] - Hy[idx]));
    }
    if (!(i == 0 || i >= Nx-1 || k == 0 || k >= Nz-1)) {
        uint idx_kp = i + j * Nx + (k+1) * Nx * Ny;
        uint idx_ip = (i+1) + j * Nx + k * Nx * Ny;
        Ey[idx] += Cb * ((Hx[idx_kp] - Hx[idx]) - (Hz[idx_ip] - Hz[idx]));
    }
    if (!(i == 0 || i >= Nx-1 || j == 0 || j >= Ny-1)) {
        uint idx_ip = (i+1) + j * Nx + k * Nx * Ny;
        uint idx_jp = i + (j+1) * Nx + k * Nx * Ny;
        Ez[idx] += Cb * ((Hy[idx_ip] - Hy[idx]) - (Hx[idx_jp] - Hx[idx]));
    }"""


def stage(t):
    print(f"\n=== {t} ===", flush=True)


# --------------------------------------------------------------------------- theory

def yee_freq(k, dt, dx, c):
    s = np.sqrt(sum((2 / dx * np.sin(ki * dx / 2)) ** 2 for ki in k))
    arg = c * dt / 2 * s
    return float("nan") if arg >= 1 else 2 / dt * np.arcsin(arg) / (2 * np.pi)


def modes(L, dt, c, two_d=False):
    """Yee frequencies of the TM_mnp modes a central Ez source excites (m, n odd; p even) in a box of sides L."""
    Lx, Ly, Lz = L
    out = []
    for m in (1, 3, 5, 7):
        for n in (1, 3, 5, 7):
            for p in ((0,) if two_d else (0, 2, 4, 6, 8, 10, 12, 14)):
                k = (m * np.pi / Lx, n * np.pi / Ly, p * np.pi / Lz)
                out.append((f"({m},{n},{p})", yee_freq(k, dt, DX, c), c / 2 * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2 + (p / Lz) ** 2)))
    return out


def box_models(dt, c, two_d=False):
    return {"N*dx": modes((NX * DX, NY * DX, NZ * DX), dt, c, two_d),
            "(N-1)*dx": modes(((NX - 1) * DX, (NY - 1) * DX, (NZ - 1) * DX), dt, c, two_d)}


def modes_all_parity(L, dt, c, two_d=False):
    """Every TM_mnp Yee mode (m, n <= 8, p <= 15) of a box with sides L, for matching weakly excited even-parity lines."""
    Lx, Ly, Lz = L
    out = []
    for m in range(1, 9):
        for n in range(1, 9):
            for p in ((0,) if two_d else range(0, 16)):
                k = (m * np.pi / Lx, n * np.pi / Ly, p * np.pi / Lz)
                out.append((f"({m},{n},{p})", yee_freq(k, dt, DX, c), c / 2 * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2 + (p / Lz) ** 2)))
    return out


# --------------------------------------------------------------------------- estimators

def pencil_lines(t, v, t_start, svd_rel=3e-4, amp_thresh=AMP_THRESH, band=F_BAND, max_rank=160):
    """Matrix-pencil harmonic inversion of the ring-down: returns [(f_Hz, damping_1/s, amplitude_rel)].

    The ring-down (t >= t_start, after the drive) is low-pass filtered and decimated so that the new
    Nyquist frequency is >= 1.5 x the top of the band, then fitted as a sum of complex exponentials
    (Hua & Sarkar, IEEE TASSP 38, 814 (1990)); the rank is chosen from the singular-value spectrum."""
    dt0 = float(t[1] - t[0])
    m = t >= t_start
    y = (v[m] - v[m].mean()).astype(float)
    decim = max(1, int(1.0 / (2 * dt0 * 1.5 * band[1])))
    Y = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), dt0)
    Y[f > 0.45 / (dt0 * decim)] = 0                      # brick wall below the new Nyquist
    y = np.fft.irfft(Y, len(y))[::decim]
    dt1 = dt0 * decim
    N = len(y); Lp = N // 3
    if Lp < 8:
        return []
    H = np.lib.stride_tricks.sliding_window_view(y, Lp + 1)[: N - Lp]
    Y0, Y1 = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(Y0, full_matrices=False)
    r = int(np.sum(S > svd_rel * S[0]))
    r = max(2, min(r, max_rank, Lp - 1))
    A = np.diag(1 / S[:r]) @ U[:, :r].conj().T @ Y1 @ Vh[:r].conj().T
    z = np.linalg.eigvals(A)
    freqs = np.angle(z) / (2 * np.pi * dt1)
    damp = np.log(np.abs(z)) / dt1
    # amplitudes from the Hann spectrum of the ring-down (robust; the Vandermonde solve is ill-conditioned
    # when the pencil also returns weak or spurious poles). Poles far from the unit circle are spurious.
    y0 = (v[m] - v[m].mean()).astype(float)
    Yw = np.abs(np.fft.rfft(y0 * np.hanning(len(y0)), 8 * len(y0))); fw = np.fft.rfftfreq(8 * len(y0), dt0)
    keep = (freqs > band[0]) & (freqs < band[1]) & (np.abs(damp) < 1e6)
    freqs, damp = freqs[keep], damp[keep]
    amp = np.interp(freqs, fw, Yw)
    if len(freqs) == 0 or amp.max() <= 0:
        return []
    amp = amp / amp.max()
    lines = [(float(fr), float(d), float(a)) for fr, d, a in zip(freqs, damp, amp) if a >= amp_thresh]
    # Poles closer than CLUSTER_PCT form a cluster (the Hann amplitude cannot tell them apart, so a faint, poorly
    # conditioned twin inherits a strong line's amplitude). The box is lossless, so the physical line of a cluster is
    # its least-damped pole; the others are returned separately in `twins` for the record.
    lines.sort()
    clusters, cur = [], []
    for ln in lines:
        if cur and abs(ln[0] - cur[0][0]) > CLUSTER_PCT / 100 * cur[0][0]:
            clusters.append(cur); cur = []
        cur.append(ln)
    if cur:
        clusters.append(cur)
    kept, twins = [], []
    for cl in clusters:
        best = min(cl, key=lambda ln: abs(ln[1]))
        kept.append(best); twins += [ln for ln in cl if ln is not best]
    pencil_lines.twins = twins
    return kept


def fit_residuals(t, v, t_start, svd_rel=3e-4, max_rank=160, band=F_BAND):
    """Fit the decimated ring-down with every pencil pole (0 < f < 0.45 Nyquist) first with all dampings set to zero,
    then with the pencil's dampings; returns (relative residual undamped, relative residual damped, n_poles).
    A lossless box must be fitted by the undamped model to a few per cent (float32 round-off, unmodelled weak lines)."""
    dt0 = float(t[1] - t[0]); m = t >= t_start; y = (v[m] - v[m].mean()).astype(float)
    decim = max(1, int(1.0 / (2 * dt0 * 1.5 * band[1])))
    Y = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), dt0); Y[f > 0.45 / (dt0 * decim)] = 0
    y = np.fft.irfft(Y, len(y))[::decim]; dt1 = dt0 * decim; N = len(y); Lp = N // 3
    H = np.lib.stride_tricks.sliding_window_view(y, Lp + 1)[: N - Lp]; Y0, Y1 = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(Y0, full_matrices=False); r = int(np.sum(S > svd_rel * S[0])); r = max(2, min(r, max_rank, Lp - 1))
    z = np.linalg.eigvals(np.diag(1 / S[:r]) @ U[:, :r].conj().T @ Y1 @ Vh[:r].conj().T)
    fr = np.angle(z) / (2 * np.pi * dt1); dp = np.log(np.abs(z)) / dt1
    k = (fr > 1e6) & (fr < 0.45 / (dt0 * decim)) & (np.abs(dp) < 1e6); fr, dp = fr[k], dp[k]
    n = np.arange(N)[:, None]
    B = np.hstack([np.cos(2 * np.pi * fr[None, :] * n * dt1), np.sin(2 * np.pi * fr[None, :] * n * dt1)])
    c0, *_ = np.linalg.lstsq(B, y, rcond=None)
    Bd = B * np.exp(dp[None, :] * n * dt1).repeat(2, axis=1)
    c1, *_ = np.linalg.lstsq(Bd, y, rcond=None)
    return float(np.linalg.norm(y - B @ c0) / np.linalg.norm(y)), float(np.linalg.norm(y - Bd @ c1) / np.linalg.norm(y)), int(len(fr))


def load_series(path):
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


def fft_spectrum(t, v, t_start=0.0, pad=4):
    m = t >= t_start
    y = v[m] - v[m].mean()
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)), len(y) * pad))
    f = np.fft.rfftfreq(len(y) * pad, t[1] - t[0])
    return f, X / X.max()


def match_lines(lines, mode_list):
    rows = []
    for fr, d, a in lines:
        lab, fy, fc = min(mode_list, key=lambda mm: abs(mm[1] - fr) if np.isfinite(mm[1]) else 1e30)
        rows.append({"f_MHz": fr / 1e6, "damping_per_us": d * 1e-6, "amp": a, "mode": lab, "f_yee_MHz": fy / 1e6,
                     "f_cont_MHz": fc / 1e6, "err_pct": (fr - fy) / fy * 100, "cont_err_pct": (fr - fc) / fc * 100})
    return rows


# --------------------------------------------------------------------------- engines

def generate_swift(src: Path, dst: Path, nsteps: int, guard_fix: bool) -> dict:
    text = src.read_text(); n = {}
    n["conductor"] = text.count("if isUpperArm || isLowerArm {")
    text = text.replace("if isUpperArm || isLowerArm {", "if false && (isUpperArm || isLowerArm) {")
    text, n["steps"] = re.subn(r"let numSteps = \d+", f"let numSteps = {nsteps}", text)
    text, n["snapshots"] = re.subn(r"let snapshotInterval = \d+", "let snapshotInterval = 1000000", text)
    text, n["spread"] = re.subn(r"let spread: Float = 3\.0 / designFreq", f"let spread: Float = {SPREAD_PERIODS} / designFreq", text)
    if guard_fix:
        text, n["guard"] = GUARD_FIX_OLD.subn(lambda mo: GUARD_FIX_NEW, text)
    dst.write_text(text)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-meep", action="store_true"); ap.add_argument("--steps", type=int, default=32000)
    args = ap.parse_args()
    exlib.reset_results(RES); (RES / "bin").mkdir()
    checks = exlib.Checks()
    S: dict = {"provenance": exlib.provenance(), "steps": args.steps, "grid": [NX, NY, NZ], "dx": DX, "t_ringdown_s": T_RINGDOWN}
    t0 = time.time()

    # ---------------------------------------------------------------- 1. Swift engines
    stage(f"1. Empty PEC box on the Swift/Metal engines, {args.steps} steps")
    engines = {}
    variants = [("loopfdtd", LOOP / "dipole_s11.swift", C_LOOP, False, False),
                ("loopfdtd_fixed", LOOP / "dipole_s11.swift", C_LOOP, True, False),
                ("meephistopheles", MEEPH / "src/metal/dipole_s11.swift", C_MEEPH, False, True)]
    for eng, src, c, fix, two_d in variants:
        gen = RES / "bin" / f"{eng}_empty.swift"
        n = generate_swift(src, gen, args.steps, fix)
        ok_gen = n["conductor"] == 1 and n["steps"] == 1 and n["snapshots"] == 1 and n["spread"] == 1 and (not fix or n.get("guard") == 1)
        checks.add(f"{eng}: variant generated", ok_gen, str(n), f"conductor removed, steps set, snapshots off, pulse spread {SPREAD_PERIODS} periods" + (", per-component guard" if fix else ""))
        if not ok_gen:
            continue
        exe = RES / "bin" / f"{eng}_empty"
        rc, wall, _ = exlib.run(SWIFTC + [str(gen), "-o", str(exe)], RES / "bin", RES / "bin" / f"build_{eng}.log", label=f"swiftc {eng} empty", nice=5)
        if rc != 0:
            checks.add(f"{eng}: builds", False, rc, "swiftc rc 0"); continue
        wdir = RES / f"{eng}_run"; wdir.mkdir()
        rc, wall, out = exlib.run([exe], wdir, wdir / "run.log", label=f"{eng} empty box {args.steps} steps", timeout=3600)
        produced = next(wdir.glob("dipole_*_output"), None)
        ok = rc == 0 and produced is not None and (produced / "timedomain.txt").exists()
        checks.add(f"{eng}: ran", ok, f"rc={rc}, {wall:.0f}s", "timedomain.txt written")
        if ok:
            shutil.move(str(produced), RES / eng)
            m = re.search(r"Throughput:\s*([\d.,]+)\s*Mcells/s", out)
            engines[eng] = {"c": c, "two_d": two_d, "throughput": float(m.group(1).replace(",", "")) if m else None}
    # ---------------------------------------------------------------- 2. MEEP
    if not args.skip_meep:
        stage(f"2. MEEP 3D empty box, {args.steps} steps, one Yee step per sample")
        rc, wall, out = exlib.run([exlib.MEEP_PYTHON, CROSS / "meep_dipole_box.py", RES / "meep", "--empty", "--steps", str(args.steps), "--spread", str(SPREAD_PERIODS)], RES,
                                  RES / "meep_run.log", label=f"meep empty box {args.steps} steps", timeout=7200)
        ok = rc == 0 and (RES / "meep" / "timedomain.txt").exists()
        checks.add("meep: ran", ok, f"rc={rc}, {wall:.0f}s", "timedomain.txt written")
        if ok:
            m = re.search(r"([\d.]+) Mcells/s", out)
            engines["meep"] = {"c": C_MEEP, "two_d": False, "throughput": float(m.group(1)) if m else None}

    # ---------------------------------------------------------------- 3. lines vs discrete dispersion
    stage("3. Ring-down lines (matrix pencil) vs the discrete Yee modes")
    S["engines"] = {}
    expected_box = {"meep": "N*dx", "loopfdtd_fixed": "(N-1)*dx", "meephistopheles": "(N-1)*dx"}
    for eng, info in engines.items():
        t_rec, v = load_series(RES / eng / "timedomain.txt")
        n0 = int(round(t_rec[0] / (t_rec[-1] / (len(t_rec) - 1))))     # MEEP's first sample is step 1, the Swift programs' is step 0
        dt_file = float(t_rec[-1] / (len(t_rec) - 1 + n0))
        t = (np.arange(len(t_rec)) + n0) * dt_file
        tol = 1e-9 if eng == "meep" else 2e-6                             # float64 axis vs a float32 axis printed with 6 digits
        uniform = bool(np.max(np.abs(t_rec - t)) < tol * t[-1])
        checks.add(f"{eng}: uniform time axis (one step per sample)", uniform, f"dt = {dt_file*1e12:.4f} ps, max |t_rec - i dt| = {np.max(np.abs(t_rec - t))/t[-1]:.1e} of T",
                   f"< {tol:.0e} of T")
        lines = pencil_lines(t, v, T_RINGDOWN); twins = list(pencil_lines.twins)
        rd = v[t >= T_RINGDOWN] - v[t >= T_RINGDOWN].mean(); h = len(rd) // 2
        rms_ratio = float(np.sqrt(np.mean(rd[h:] ** 2)) / np.sqrt(np.mean(rd[:h] ** 2)))   # reported only: slow beats of close pairs move it
        res_u, res_d, n_poles = fit_residuals(t, v, T_RINGDOWN)
        bm = box_models(dt_file, info["c"], info["two_d"])
        per_model = {name: match_lines(lines, ml) for name, ml in bm.items()}
        worst = {name: max((abs(r["err_pct"]) for r in rows if r["amp"] >= AMP_GATE), default=float("nan")) for name, rows in per_model.items()}
        maxdamp = max((abs(d) for _, d, a in lines if a >= AMP_GATE), default=0.0)
        n_gated = sum(1 for _, _, a in lines if a >= AMP_GATE)
        print(f"  {eng}: {len(lines)} lines above {AMP_THRESH:.0%} ({n_gated} above {AMP_GATE:.0%}, gated) in the ring-down (c = {info['c']:.6e}, dt = {dt_file*1e12:.4f} ps); "
              f"worst |error| of gated lines: N*dx {worst['N*dx']:.3f} %, (N-1)*dx {worst['(N-1)*dx']:.3f} %; max |damping| {maxdamp*1e-6:.4f} /us")
        best = min(worst, key=lambda k: worst[k] if np.isfinite(worst[k]) else 1e9)
        for r in per_model[best]:
            print(f"      {r['f_MHz']:9.4f} MHz  amp {r['amp']:.3f}  damp {r['damping_per_us']:+.4f}/us  -> {r['mode']:10s} yee {r['f_yee_MHz']:9.4f} ({r['err_pct']:+.3f} %)  continuum {r['f_cont_MHz']:9.3f} ({r['cont_err_pct']:+.3f} %)")
        S["engines"][eng] = {"dt_s": dt_file, "c": info["c"], "throughput": info["throughput"], "lines": lines, "models": per_model, "worst_pct": worst, "max_damping_per_s": maxdamp, "n_gated": n_gated, "rms_ratio": rms_ratio, "twins": twins, "fit_residual_undamped": res_u, "fit_residual_damped": res_d, "n_poles": n_poles}
        if eng in expected_box:
            box = expected_box[eng]
            rows = per_model[box]
            n_min = 2 if info["two_d"] else 3                   # the 2D box has only (3,1,0), (3,3,0), (5,1,0), (5,3,0) in the band
            checks.add(f"{eng}: every ring-down line >= {AMP_GATE:.0%} is a discrete Yee mode of the {box} box", n_gated >= n_min and worst[box] < 0.05,
                       f"worst {worst[box]:.4f} % over {n_gated} gated lines ({len(rows)} reported)", f"< 0.05 % on >= {n_min} lines (exact discrete eigenmodes; pencil precision ~1e-3 %)")
            other = [k for k in bm if k != box][0]
            gated_other = [r for r in per_model[other] if r["amp"] >= AMP_GATE]
            frac_other = float(np.mean([abs(r["err_pct"]) < 0.05 for r in gated_other])) if gated_other else 1.0
            checks.add(f"{eng}: the {other} box is excluded", bool(rows) and frac_other < 0.5,
                       f"{frac_other:.0%} of the lines fall within 0.05 % of a {other} mode (worst {worst[other]:.3f} %)", "< 50 % (the two boxes differ by 1/64 = 1.6 % in every mode)")
            f31 = [mm[1] for mm in bm[box] if mm[0] == "(3,1,0)"][0]
            present = any(abs(fr - f31) / f31 < 5e-4 for fr, _, _ in lines)
            checks.add(f"{eng}: the (3,1,0) line is present", present, f"expected {f31/1e6:.4f} MHz", "a line within 0.05 % (the matcher alone cannot fail on missing lines)")
            if twins:
                nb = {"N*dx": (NX * DX, NY * DX, NZ * DX), "(N-1)*dx": ((NX - 1) * DX, (NY - 1) * DX, (NZ - 1) * DX)}[box]
                trows = match_lines(twins, modes_all_parity(nb, dt_file, info["c"], info["two_d"]))
                S["engines"][eng]["twin_rows"] = trows
                checks.add(f"{eng}: poles set aside by the cluster rule also lie on Yee modes of the {box} box (any parity)",
                           all(abs(r["err_pct"]) < 0.1 for r in trows),
                           "; ".join(f"{r['f_MHz']:.3f} -> {r['mode']} {r['err_pct']:+.3f} %" for r in trows),
                           "< 0.1 % (weak even-parity modes excited by the half-cell source offset, or close odd-parity neighbours)")
            f31_line = min(lines, key=lambda ln: abs(ln[0] - f31))
            checks.add(f"{eng}: the isolated (3,1,0) line is undamped (lossless box)", abs(f31_line[0] - f31) / f31 < 5e-4 and abs(f31_line[1]) < 1e4,
                       f"damping {f31_line[1]*1e-6:+.5f} /us", "|damping| < 0.01 /us (< 1.4 % decay over the 1.4 us ring-down; nearest other mode 4 % away)")
            checks.add(f"{eng}: an undamped model of all {n_poles} poles fits the whole ring-down", res_u < 0.05 and res_u < res_d,
                       f"relative residual {res_u:.4f} undamped vs {res_d:.4f} with the pencil's dampings; rms(second half)/rms(first half) = {rms_ratio:.4f}",
                       "< 0.05 and below the damped fit (the per-pole dampings of close pairs are fit noise, not loss)")
            if eng == "meep":
                cont_worst = max(abs(r["cont_err_pct"]) for r in rows if r["amp"] >= AMP_GATE)
                checks.add("meep: the continuum formula is measurably worse than the Yee formula", cont_worst > 3 * worst[box], f"continuum {cont_worst:.3f} % vs Yee {worst[box]:.4f} %", "numerical dispersion resolved")
        else:
            rows = [r for r in per_model["(N-1)*dx"] if r["amp"] >= AMP_GATE]
            checks.report("loopfdtd (as coded): deviation from the (N-1)*dx Yee box (finding, not gated)",
                          f"lines off by {min((r['err_pct'] for r in rows), default=float('nan')):+.3f} .. {max((r['err_pct'] for r in rows), default=float('nan')):+.3f} %",
                          "the single boundary guard clamps the normal E on the far half-planes; see loopfdtd_fixed")

    # ---------------------------------------------------------------- 4. plot + results
    stage("4. Plot and RESULTS.md")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(engines), 1, figsize=(11, 2.6 * len(engines)), sharex=True, squeeze=False)
        for ax, (eng, e) in zip(axes[:, 0], S["engines"].items()):
            t, v = load_series(RES / eng / "timedomain.txt")
            f, X = fft_spectrum(t, v, T_RINGDOWN)
            ax.plot(f / 1e6, X, lw=0.8, color="C0", label=f"{eng}: ring-down FFT")
            box = expected_box.get(eng, "(N-1)*dx")
            for lab, fy, fc in box_models(e["dt_s"], e["c"], engines[eng]["two_d"])[box]:
                if F_BAND[0] < fy < F_BAND[1]:
                    ax.axvline(fy / 1e6, color="k", lw=0.4, alpha=0.5)
            for fr, d, a in e["lines"]:
                ax.plot(fr / 1e6, a, "r|", ms=12)
            ax.set_yscale("log"); ax.set_ylim(1e-3, 1.5); ax.set_xlim(F_BAND[0] / 1e6, F_BAND[1] / 1e6); ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("|V(f)|")
        axes[-1, 0].set_xlabel("MHz"); axes[0, 0].set_title("Empty PEC box: ring-down spectra; black lines = discrete Yee modes of the expected box; red = pencil lines")
        fig.tight_layout(); fig.savefig(RES / "yee_dispersion.png", dpi=140); print(f"  wrote {RES / 'yee_dispersion.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot skipped: {exc}")
    S["checks"] = checks.rows; S["all_pass"] = checks.all_pass; S["total_wall_s"] = time.time() - t0
    exlib.dump_json(RES / "summary.json", S)
    L = [f"# Example 6 results: Yee dispersion ({S['provenance']['date']})", "", f"{S['provenance']['hardware']}, box {NX}x{NY}x{NZ} cells of {DX} m, {args.steps} steps, ring-down from {T_RINGDOWN*1e9:.0f} ns", "", checks.summary(), "",
         "## Checks", "", checks.markdown(), ""]
    for eng, e in S["engines"].items():
        box = expected_box.get(eng, "(N-1)*dx")
        L += [f"## {eng}  (c = {e['c']:.6e} m/s, dt = {e['dt_s']*1e12:.4f} ps, throughput {e['throughput']} Mcells/s; compared with the {box} box)", "",
              "| f measured (MHz) | amplitude | damping (1/us) | mode | f Yee (MHz) | error | f continuum (MHz) | continuum error |", "|---|---|---|---|---|---|---|---|"]
        for r in e["models"][box]:
            L.append(f"| {r['f_MHz']:.4f} | {r['amp']:.3f} | {r['damping_per_us']:+.4f} | {r['mode']} | {r['f_yee_MHz']:.4f} | {r['err_pct']:+.4f} % | {r['f_cont_MHz']:.3f} | {r['cont_err_pct']:+.3f} % |")
        L += ["", f"worst error of gated lines: N*dx {e['worst_pct']['N*dx']:.4f} %, (N-1)*dx {e['worst_pct']['(N-1)*dx']:.4f} %; undamped-model fit residual {e['fit_residual_undamped']:.4f} (damped {e['fit_residual_damped']:.4f}); ring-down rms ratio {e['rms_ratio']:.4f}"]
        if e.get("twin_rows"):
            L += ["", "Twin poles set aside by the cluster rule (full-parity match): " + "; ".join(f"{r['f_MHz']:.3f} MHz -> {r['mode']} ({r['err_pct']:+.3f} %)" for r in e["twin_rows"])]
        L.append("")
    L += ["## Files", "", "- results/bin/*_empty.swift (generated copies; loopfdtd_fixed carries the per-component guard), results/<engine>/timedomain.txt, results/summary.json, results/yee_dispersion.png"]
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"\n{'ALL CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'} ({checks.summary()}) in {S['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

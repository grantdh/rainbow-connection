"""Compare the dipole-in-a-box time traces from LoopFDTD, MEEPhistopheles and MEEP.

python analyze.py results/   (any interpreter with numpy + matplotlib)
Reads results/<engine>/timedomain.txt and metadata.txt, writes results/comparison.json,
results/crosscheck.png and prints a table. The probes are uncalibrated in all three programs, so the
comparison is of normalised spectra and resonance positions, not absolute impedance.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

res = Path(sys.argv[1])
ENGINES = ["loopfdtd_empty", "meephistopheles_empty", "meep_empty", "loopfdtd_shipped", "loopfdtd_fixed", "meephistopheles_shipped", "meephistopheles_fixed", "meep_shipped", "meep_fixed"]


def load(engine: str):
    d = res / engine
    if not (d / "timedomain.txt").exists():
        return None
    data = np.loadtxt(d / "timedomain.txt", comments="#")
    meta = {}
    for line in (d / "metadata.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return {"t": data[:, 0], "v": data[:, 1], "i": data[:, 2], "meta": meta}


def spectrum(t, x):
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft((x - x.mean()) * w)
    f = np.fft.rfftfreq(n, d=t[1] - t[0])
    return f, np.abs(X)


def peaks(f, mag, fmin=100e6, fmax=600e6, n=6):
    m = (f > fmin) & (f < fmax)
    fi, mi = f[m], mag[m]
    idx = [k for k in range(1, len(mi) - 1) if mi[k] > mi[k - 1] and mi[k] > mi[k + 1] and mi[k] > 0.08 * mi.max()]
    idx = sorted(idx, key=lambda k: -mi[k])[:n]
    return sorted(float(fi[k]) for k in idx)


data = {e: load(e) for e in ENGINES}
out = {"engines": {}, "pairs": {}}
for e, d in data.items():
    if d is None:
        out["engines"][e] = None
        continue
    f, mag = spectrum(d["t"], d["v"])
    pk = peaks(f, mag)
    thr = d["meta"].get("throughput_Mcells_per_s") or d["meta"].get("Throughput") or ""
    thr = float(re.sub(r"[^\d.]", "", thr) or 0)
    out["engines"][e] = {"steps": len(d["t"]), "dt_s": float(d["t"][1] - d["t"][0]), "peaks_Hz": pk,
                         "throughput_Mcells_s": thr, "engine": d["meta"].get("engine", e),
                         "v_rms": float(np.sqrt(np.mean(d["v"] ** 2)))}
    print(f"{e:24s} steps={len(d['t'])} dt={d['t'][1] - d['t'][0]:.3e}s  peaks(MHz)={[round(p / 1e6, 1) for p in pk]}  {thr:.0f} Mcells/s")


def compare(a, b):
    da, db = data[a], data[b]
    if da is None or db is None:
        return None
    n = min(len(da["t"]), len(db["t"]))
    va, vb = da["v"][:n], db["v"][:n]
    corr = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else 0.0
    fa, ma = spectrum(da["t"][:n], va); fb, mb = spectrum(db["t"][:n], vb)
    ma, mb = ma / ma.max(), mb / mb.max()
    band = (fa > 100e6) & (fa < 1.2e9)
    spec_rms = float(np.sqrt(np.mean((ma[band] - mb[band]) ** 2)))
    pa, pb = out["engines"][a]["peaks_Hz"], out["engines"][b]["peaks_Hz"]
    dpk = [min(abs(p - q) / p for q in pb) if pb else 1.0 for p in pa]
    return {"time_corr": corr, "norm_spectrum_rms": spec_rms, "peak_rel_mismatch_max": float(max(dpk) if dpk else 1.0)}


for a, b in (("loopfdtd_empty", "meep_empty"), ("loopfdtd_fixed", "meep_fixed"), ("loopfdtd_shipped", "meep_shipped"), ("meephistopheles_fixed", "meep_fixed"),
             ("loopfdtd_fixed", "meephistopheles_fixed"), ("loopfdtd_shipped", "loopfdtd_fixed")):
    out["pairs"][f"{a}_vs_{b}"] = compare(a, b)
    if out["pairs"][f"{a}_vs_{b}"]:
        c = out["pairs"][f"{a}_vs_{b}"]
        print(f"{a} vs {b}: V(t) corr={c['time_corr']:.3f}, |spectrum| rms={c['norm_spectrum_rms']:.3f}, "
              f"max peak mismatch={c['peak_rel_mismatch_max'] * 100:.1f}%")
json.dump(out, open(res / "comparison.json", "w"), indent=2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for e, d in data.items():
        if d is None:
            continue
        ax[0].plot(d["t"] * 1e9, d["v"] / (np.abs(d["v"]).max() or 1), lw=0.8, label=e)
        f, mag = spectrum(d["t"], d["v"])
        ax[1].plot(f / 1e6, mag / mag.max(), lw=0.9, label=e)
    ax[0].set_xlabel("time (ns)"); ax[0].set_ylabel("gap Ez (normalised)"); ax[0].legend(); ax[0].set_title("dipole in a PEC box: gap field")
    ax[1].set_xlim(0, 1200); ax[1].set_xlabel("frequency (MHz)"); ax[1].set_ylabel("|V(f)| (normalised)"); ax[1].legend()
    ax[1].set_title("spectrum: box + dipole resonances")
    fig.tight_layout(); fig.savefig(res / "crosscheck.png", dpi=140)
    print("wrote", res / "crosscheck.png")
except Exception as exc:  # noqa: BLE001
    print("plot skipped:", exc)

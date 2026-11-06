"""Y(f) of the Si dipole from the raw .s1p sweeps and calibration standards, with the standards' definitions explicit.

The files are WinCal CALIBRATED_DATA. The raw receiver data no longer exist, so the only calibration information left is the
calibrated readback of the three standards, taken right after the calibration. Y is computed under three definitions of the
reference plane, which bound the calibration term in u_D:

  A  as saved: the tool's own definition of the standards;
  B  analytic re-reference to the probe kit's open capacitance: Y' = Y + j w C (1 - Y Z0) (bilinear, not a shunt);
  C  B plus a second-tier one-port SOL from the readbacks (that open, ideal short, ideal load), for the residual error.

A fixed-delay ripple is removed from every sweep by the same bandstop time gate, per day; the gate's band-edge transient sets the band (delays and bands in device.json).
u_D per setting = rss(noise, calibration |C - B|, gate |3 ns - 6 ns span|, drift bound). Writes OUT/measured_y.npz + .md.
"""
import json

import numpy as np
import skrf as rf
from scipy.signal import savgol_filter

import common as c

Z0 = 50.0
RAW = c.device()["raw"]
C_OPEN = RAW["c_open_F"]
DAYS = {name: c.raw_day(name) for name in ("X", "Ka")}


def y_of(s):
    return (1 - s) / (1 + s) / Z0


def net(p):
    n = rf.Network(str(p))
    n.when = c.stamp(p)
    return n


def gate(n, center, span):
    return rf.time.time_gate(n, center=center, span=span, mode="bandstop", window=("kaiser", 6), t_unit="s")


def rereference(f, y):
    return y + 2j * np.pi * f * C_OPEN * (1 - y * Z0)


def second_tier(day):
    op, sh, lo = [net(day["root"] / f"{n}.s1p") for n in day["cal"]]
    w = 2 * np.pi * op.f
    g_open = (1 - 1j * w * C_OPEN * Z0) / (1 + 1j * w * C_OPEN * Z0)
    ideal = lambda g: rf.Network(frequency=op.frequency, s=np.asarray(g).reshape(-1, 1, 1), z0=Z0)
    one = np.ones(len(w))
    cal = rf.calibration.OnePort(measured=[op, sh, lo], ideals=[ideal(g_open), ideal(-one), ideal(0 * one)])
    return cal


def process(name):
    day = DAYS[name]
    cal = second_tier(day)
    res, meta = {}, {}
    files = {"dark": day["dark"], **{k: day["lit"].format(k=k) for k in range(1, 8)}}
    if day["after"]:
        files["after"] = day["after"]
    if "bias" in day:
        files["bias"] = day["bias"]
    for key, fn in files.items():
        n = net(day["dev"] / fn)
        g3, g6 = gate(n, day["gate_s"], 3e-9), gate(n, day["gate_s"], 6e-9)
        ya, yb = y_of(g3.s[:, 0, 0]), rereference(n.f, y_of(g3.s[:, 0, 0]))
        yc = y_of(cal.apply_cal(g3).s[:, 0, 0])
        yc6 = y_of(cal.apply_cal(g6).s[:, 0, 0])
        res[key] = dict(a=ya, b=yb, c=yc, c6=yc6)
        meta[key] = n.when
    f = n.f
    m = (f >= day["band_hz"][0]) & (f <= day["band_hz"][1])
    t0 = meta["dark"]
    # drift + persistence together, from dark-after where it exists; otherwise the worst rate of the other day's dark-after pairs
    if "after" in res:
        rate = np.mean(np.abs(res["after"]["c"][m] - res["dark"]["c"][m])) / max((meta["after"] - t0).total_seconds(), 1.0)
    else:
        rate = RAW["carried_drift_S_per_s"]
    table = {}
    for key in res:
        if key == "dark":
            continue
        dy = res[key]["c"] - res["dark"]["c"]
        noise = np.std((dy - savgol_filter(dy.real, day["smooth"], 3) - 1j * savgol_filter(dy.imag, day["smooth"], 3))[m])
        ucal = np.mean(np.abs(dy - (res[key]["b"] - res["dark"]["b"]))[m])
        ugate = np.mean(np.abs(dy - (res[key]["c6"] - res["dark"]["c6"]))[m])
        udrift = rate * (meta[key] - t0).total_seconds()
        u = float(np.sqrt(noise**2 / m.sum() + ucal**2 + ugate**2 + udrift**2))
        table[key] = dict(dy=complex(np.mean(dy[m])), u=u, noise=float(noise), cal=float(ucal), gate=float(ugate), drift=float(udrift),
                          t_s=float((meta[key] - t0).total_seconds()))
    return f[m], {k: {kk: vv[m] for kk, vv in v.items()} for k, v in res.items()}, table


def extra_pairs():
    """Raw differences outside the dipole's X series: the Ka day's ambient sweeps around the lit run, and the control devices' response in the same laser (X day)."""
    day = DAYS["Ka"]
    a, b = net(day["dev"] / day["ambient"][0]), net(day["dev"] / day["ambient"][1])
    m = (a.f >= day["band_hz"][0]) & (a.f <= day["band_hz"][1])
    ka = complex(np.mean((y_of(gate(b, day["gate_s"], 3e-9).s[:, 0, 0]) - y_of(gate(a, day["gate_s"], 3e-9).s[:, 0, 0]))[m]))
    last = net(day["dev"] / day["lit"].format(k=7))
    out = {"Ka_ambient_after_minus_before": [ka.real, ka.imag], "Ka_ambient_s_since_last_lit": (b.when - last.when).total_seconds(), "controls": {}}
    x = DAYS["X"]
    cal = second_tier(x)
    for ctl in RAW["controls"]:
        d = x["root"] / ctl["folder"]
        y = lambda fn: y_of(cal.apply_cal(gate(net(d / fn), x["gate_s"], 3e-9)).s[:, 0, 0])
        y0 = y(ctl["dark"])
        f = net(d / ctl["dark"]).f
        mx = (f >= x["band_hz"][0]) & (f <= x["band_hz"][1])
        rows = {}
        for k in c.device()["settings"]:
            dy = complex(np.mean((y(ctl["lit"].format(k=k)) - y0)[mx]))
            rows[k] = [dy.real, dy.imag]
        out["controls"][ctl["name"]] = rows
    return out


def main():
    c.require_private_repo()
    json.dump(extra_pairs(), open(c.OUT / "measured_extra.json", "w"), indent=1)
    arrays, lines = {}, ["# Y(f) recomputed from the raw .s1p (definition C: gated, kit open, second-tier SOL)", ""]
    for name in DAYS:
        f, res, table = process(name)
        arrays[f"{name}_f"] = f
        for k, v in res.items():
            for kk, vv in v.items():
                arrays[f"{name}_{k}_{kk}"] = vv
        dark = np.mean(res["dark"]["c"])
        wc = 2 * np.pi * np.mean(f)
        z = 1 / dark
        lines += [f"## {name} band: {f[0]/1e9:.2f}-{f[-1]/1e9:.2f} GHz, {len(f)} points", "",
                  f"dark, def C: G = {dark.real*1e3:.4f} mS, C = {dark.imag/wc*1e15:.2f} fF (series R {z.real:.0f} ohm, C {-1/(wc*z.imag)*1e15:.1f} fF)",
                  f"dark, def A (as saved): G = {np.mean(res['dark']['a']).real*1e3:.4f} mS, C = {np.mean(res['dark']['a']).imag/wc*1e15:.2f} fF",
                  "",
                  "| state | t (s) | dY mS | phase deg | u_D uS | noise | cal | gate | drift |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in table.items():
            d = v["dy"]
            lines.append(f"| {k} | {v['t_s']:.0f} | {d.real*1e3:+.4f} {d.imag*1e3:+.4f}j | {np.degrees(np.angle(d)):.1f} | {v['u']*1e6:.2f} | {v['noise']*1e6:.2f} | {v['cal']*1e6:.2f} | {v['gate']*1e6:.2f} | {v['drift']*1e6:.2f} |")
            arrays[f"{name}_table_{k}"] = np.array([d.real, d.imag, v["u"], v["t_s"]])
        lines.append("")
    np.savez(c.OUT / "measured_y.npz", **arrays)
    (c.OUT / "measured_y.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

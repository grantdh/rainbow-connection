"""The deeper heating question (prereg/00d-00g): time fingerprints (09), the GaAs controls (10), model evidence (12), and the predictions for the measurements
that would settle it (13, with out/PROTOCOL.md). Two scenarios carry every prediction:

  nominal  the laser's threshold in device.json, median absorbed power, R_th and C at the 08 contact median, lifetime (T/T0)^2
  corner   the one region 07c leaves open: its threshold, absorbed power at the 97.5th percentile, 07b's smallest R_th there with E_a <= 0.40 eV,
           and E_a 0.40 eV
Device values and record text come from device.json's deep section in the private repository, the protocol's prose from its templates/.

  deep.py fingerprints | gaas | evidence | redrows | protocol | contact
"""
import json
import sys

import numpy as np
from scipy.special import erf

import common as c
import contact
import flip_map as fm
import thermal_elmer as te

T0 = fm.T0
CHOP_HZ = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4]


def prereg(name):
    return json.load(open(c.PREREG / f"{name}.json"))


def rho_spec():
    """The handle's resistivity range on the wafer's spec sheet (ohm cm)."""
    return c.device()["deep"]["handle"]["rho_spec_ohm_cm"]


def light(name):
    """Beam-train transmission and beam radius w (m) of a scenario: the 01 nominal values, or for the corner the end of each 01 prior that favours heating."""
    li = c.inputs()["light"]
    tr, w = li["train_transmission"], li["beam_w_mm"]
    return (tr["value"], w["value"] * 1e-3) if name == "nominal" else (tr["dist"]["hi"], w["dist"]["lo"] * 1e-3)


def corner_row():
    """07c's open threshold and 07b's row for it (EPW mu, absorbed power at the 97.5th percentile, E_a <= 0.40 eV)."""
    ith = float(min(prereg("07c_one_laser")["joint_thresholds"]))
    return ith, prereg("07b_flip_threshold")["table"][f"EPW|Ith{ith:.1f}|q97.5"]["Ea<=0.40"]


def scenarios(e):
    con = prereg("08_contact")
    ith, fb = corner_row()
    C = con["C_J_K"]["q50"]
    return {"nominal": dict(ith=c.device()["laser"]["i_th_A"], fac=e["facs"]["median"], R=con["R_th_K_W"]["q50"], C=C, law=("pow", 2.0)),
            "corner": dict(ith=ith, fac=e["facs"]["q97.5"], R=fb["R_th_min_K_W"], C=C, law=("Ea", 0.40))}


def g_of(mt, law):
    tau = fm.tau_law(*law)
    return lambda T: mt["mu_sum"](T0 + T) * tau(T)


def sigma_spec(mt):
    """sigma(T) at both ends of the spec handle's resistivity range, interpolated in ln rho between the neighbouring tables of 02."""
    spec = rho_spec()
    a, b = max(r for r in mt["sigma"] if r <= min(spec)), min(r for r in mt["sigma"] if r >= max(spec))
    lo, hi = np.log(a), np.log(b)
    w = lambda r: (np.log(r) - lo) / (hi - lo)
    return {r: np.exp((1 - w(r)) * np.log(mt["sigma"][a]) + w(r) * np.log(mt["sigma"][b])) for r in spec}


def handle_shift(e, mt, day, rise, sig):
    """flip_map.handle_shift with rises beyond 02's table clamped to its end (as registered in 09 and 12)."""
    return fm.handle_shift(e["ysig"], mt, day, np.atleast_1d(rise), sig, right=None)


# ------------------------------------------------------------------------------------------------ 09
def fingerprints():
    c.require_private_repo()
    e = fm.setup()
    mt = e["mats"]["EPW"]
    sc = scenarios(e)
    tl, idx = e["tl"], e["idx"]
    X, KA = c.raw_day("X"), c.raw_day("Ka")
    res = {"what": "09: time fingerprints for the nominal and corner scenarios (00d C)", "scenarios": sc}
    d = c.device()["deep"]
    for name, s in sc.items():
        g = g_of(mt, s["law"])
        Tx = fm.temps(tl["X"], np.array(s["R"]), np.array(s["C"]), s["ith"], s["fac"])
        Tk = fm.temps(tl["Ka"], np.array(s["R"]), np.array(s["C"]), s["ith"], s["fac"])
        t7, tb = Tx[idx["X"][X["lit"].format(k=7)]], Tx[idx["X"][X["bias"]]]
        r = {"dT_at_7_0V_K": float(t7), "dT_at_7_10V_K": float(tb), "warming_change_7_10V_vs_7_0V": float(g(tb) / g(t7) - 1)}
        for rho, sig in sigma_spec(mt).items():
            d1 = complex(handle_shift(e, mt, "X", Tx[idx["X"][X["after"]]], sig)[0])
            amb = complex(handle_shift(e, mt, "Ka", Tk[idx["Ka"][KA["ambient"][1]]], sig)[0])
            r[f"D1_dark_after_uS_rho{rho:g}"] = [d1.real * 1e6, d1.imag * 1e6]
            r[f"Ka_ambient_after_minus_before_uS_rho{rho:g}"] = [amb.real * 1e6, amb.imag * 1e6]
        # drift inside one trace: the recorded averaged sweeps (deep.json; the Ka-day record, assumed for X), starting t_s after the setting changes
        # from 6 to 7; the change of |dY| from the first to the last 5 % of the band
        p6, p7 = s["fac"] * fm.power(6, s["ith"]), s["fac"] * fm.power(7, s["ith"])
        t6 = Tx[idx["X"][X["lit"].format(k=6)]]
        tilt = {}
        sw = d["sweep"]
        m = int(0.05 * sw["points"])
        frac = np.linspace(0, 1, sw["points"])
        for ts in (0.0, 2.0, 5.0, 10.0, 15.0):
            tt = ts + np.arange(sw["averages"])[:, None] * sw["seconds"] + frac[None, :] * sw["seconds"]
            T = p7 * s["R"] + (t6 - p7 * s["R"]) * np.exp(-tt / (s["R"] * s["C"]))
            y = np.mean(g(T), axis=0)
            tilt[f"{ts:g}"] = float(np.mean(y[-m:]) / np.mean(y[:m]) - 1)
        r["in_band_tilt_from_drift_s7_by_start_delay_s"] = tilt
        res[name] = r
    # the bias sweep: the membrane is the only DC path; its conductivity under setting 7 at most sigma_max, at least squares_min squares
    bb = d["bias_bound"]
    rs_min = 1.0 / (bb["sigma_max_S_m"] * (c.inputs()["geometry"]["t_device_nm"]["value"] * 1e-9))
    p_bias = bb["V"]**2 / (bb["squares_min"] * rs_min)
    res["bias_joule_bound"] = {"P_max_W": p_bias, "dT_max_K_at_08_q99": p_bias * prereg("08_contact")["R_th_K_W"]["q99"],
                               "note": d["text"]["09_bias_note"]}
    c.register("09_time_fingerprints", res, code=[__file__, fm.__file__, c.HERE / "common.py"])
    print(json.dumps(res, indent=0)[:2500])


# ------------------------------------------------------------------------------------------------ 10
def gaas():
    c.require_private_repo()
    e = fm.setup()
    sc = scenarios(e)
    rng = np.random.default_rng(c.SEED + 10)
    n = 50_000
    d = c.device()["deep"]
    gc, bw = d["gaas_controls"], c.inputs()["light"]["beam_w_mm"]["dist"]      # priors on the controls' membranes and slides (ASM; see its _source)
    a = rng.uniform(*gc["radius_m"], n)           # radius of the absorbing membrane
    k_glass = rng.uniform(*gc["k_glass"], n)
    absorb = rng.uniform(*gc["absorbed"], n)      # incl. reflection
    n_mu = rng.uniform(*gc["mu_exponent"], n)     # mu ~ T^-n
    w = 1e-3 * np.exp(rng.uniform(np.log(bw["lo"]), np.log(bw["hi"]), n))
    res = {"what": d["text"]["10_what"]}
    for name, s in sc.items():
        i_peak = 2 * fm.power(7, s["ith"]) * light(name)[0] / (np.pi * w**2)
        dT = i_peak * np.pi * a**2 * absorb / (4 * k_glass * a)
        res[name] = {"dT_membrane_K": c.percentiles(dT), "dln_sigma": c.percentiles(-n_mu * dT / T0)}
    ext = json.load(open(c.OUT / "measured_extra.json"))["controls"]
    lin = {}
    for u in (2.5e-6, 5e-6):
        for dev, rows in ext.items():
            y = {int(k): abs(complex(*v)) for k, v in rows.items()}
            for ks in ((4, 5, 6, 7), (3, 4, 5, 6, 7)):
                ok = []
                for ith in fm.ITH_FINE:
                    P = np.array([fm.power(k, float(ith)) for k in ks])
                    yy = np.array([y[k] for k in ks])
                    sc_ = np.sum(P * yy) / np.sum(P**2)
                    if np.all(np.abs(sc_ * P - yy) <= fm.tol(u, yy)):
                        ok.append(float(ith))
                lin[f"{dev}|s{ks[0]}-7|u{u * 1e6:g}uS"] = ok
    res["post_hoc_linearity_thresholds"] = lin
    c.register("10_gaas_controls", res, code=[__file__, c.HERE / "common.py"])
    print(json.dumps(res, indent=0)[:2000])


# ------------------------------------------------------------------------------------------------ 12
def evidence():
    """Prior Monte Carlo evidence on a common data set: X ratios s5/s3 and s7/s3, D1, Ka ratios s5/s3 and s7/s3. Also the X ratios 4-7 alone."""
    c.require_private_repo()
    e = fm.setup()
    rng = np.random.default_rng(c.SEED + 12)
    n = 40_000
    d = c.device()["deep"]
    meas, tl, lit = e["meas"], e["tl"], e["lit"]
    cr = contact.r_th(contact.sample(n, seed=c.SEED + 120))
    ck = contact.r_th(contact.sample(n, seed=c.SEED + 121))
    th = dict(ith=rng.uniform(*d["evidence_ith_prior_A"], n), fac=rng.choice(e["fac_all"], n), R=cr["R"], C=cr["C"], Rk=ck["R"],
              act=rng.random(n) < 0.5, ea=rng.uniform(0, 0.40, n), pw=rng.uniform(0, 4, n), epw=rng.random(n) < 0.5, rho=rng.uniform(*rho_spec(), n))
    i_after = e["i_after"]
    gauss = lambda r, s: -0.5 * (r / s) ** 2 - np.log(np.sqrt(2 * np.pi) * s)
    cgauss = lambda r, s: -0.5 * (np.abs(r) / s) ** 2 - np.log(2 * np.pi * s**2)
    P = {k: fm.power(k, th["ith"]) for k in range(3, 8)}
    Tx = fm.temps(tl["X"], th["R"], th["C"], th["ith"], th["fac"])
    Tk = fm.temps(tl["Ka"], th["Rk"], th["C"], th["ith"], th["fac"])
    tau = lambda T: np.where(th["act"], fm.tau_law("Ea", th["ea"])(T), fm.tau_law("pow", th["pw"])(T))
    out = {}
    for mu_name in ("EPW", "Jacoboni"):
        mt = e["mats"][mu_name]
        sel = th["epw"] if mu_name == "EPW" else ~th["epw"]
        g = lambda T: mt["mu_sum"](T0 + T) * tau(T)
        def ratio(T, day, k):
            return fm.tc_ratios(P, {j: g(T[lit[day][j]]) for j in (3, k)}, (k,))[k]
        ll = {"X57": 0, "X4567": 0}
        for k in (4, 5, 6, 7):
            lk = gauss(ratio(Tx, "X", k) - meas["X"]["ratio"][k], np.hypot(meas["X"]["u_ratio"][k], 0.05 * meas["X"]["ratio"][k]))
            ll["X4567"] = ll["X4567"] + lk
            if k in (5, 7):
                ll["X57"] = ll["X57"] + lk
        lka = sum(gauss(ratio(Tk, "Ka", k) - meas["Ka"]["ratio"][k], np.hypot(meas["Ka"]["u_ratio"][k], 0.05 * meas["Ka"]["ratio"][k])) for k in (5, 7))
        sig = sigma_spec(mt)
        lo, hi = rho_spec()
        wgt = (np.log(th["rho"]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        rise = Tx[i_after]
        d1 = (1 - wgt) * handle_shift(e, mt, "X", rise, sig[lo]) + wgt * handle_shift(e, mt, "X", rise, sig[hi])
        sd = np.hypot(meas["X"]["u_after"], fm.U_REL * np.abs(d1))
        ld1 = cgauss(d1 - meas["X"]["after"], sd)
        common = ll["X57"] + ld1 + lka
        pass_x = fm.ok_ratio({k: ratio(Tx, "X", k) for k in (4, 5, 6, 7)}, meas["X"], (4, 5, 6, 7))
        pass_d1 = np.abs(d1 - meas["X"]["after"]) <= fm.K_TOL * sd
        ka_lin = th["ith"] <= max(float(x) for x in prereg("07c_one_laser")["Ka_linear_s4-7_thresholds"]) + 0.025
        lse = lambda v: float(np.log(np.mean(np.exp(v[sel] - np.max(v[sel])))) + np.max(v[sel]))
        out[f"H_TC|{mu_name}"] = {"log10_Z_common": lse(common) / np.log(10), "log10_Z_X4567_D1": lse(ll["X4567"] + ld1) / np.log(10),
                                  "log10_maxL_common": float(np.max(common[sel]) / np.log(10)), "n": int(sel.sum()),
                                  "P_prior_heating_reproduces_X_with_D1": float(np.mean((pass_x & pass_d1)[sel])),
                                  "P_prior_and_Ka_linear_07c": float(np.mean((pass_x & pass_d1 & ka_lin)[sel]))}
    # carriers: the registered nominal full-wave lit runs (04) at both days' centre frequencies, with their registered u_input; no D1 shift
    fw = prereg("04_fullwave")
    split, u_in = fw["split"], fw["u_input_lit_s7_rel"]
    lc = 0.0
    for day, f in fm.FREQ.items():
        dy = {k: complex(*split[f"ps_lit_r10_w10_s{k}"][f]["Y_mS"]) - complex(*split["ps_dark_r10"][f]["Y_mS"]) for k in (3, 5, 7)}
        for k in (5, 7):
            s_ = abs(dy[k]) / abs(dy[3])
            lc += gauss(s_ - meas[day]["ratio"][k], np.hypot(np.hypot(meas[day]["u_ratio"][k], 0.05 * meas[day]["ratio"][k]), u_in * s_))
    lc += cgauss(meas["X"]["after"], meas["X"]["u_after"])
    out["H_C_nominal"] = {"log10_Z_common": float(lc / np.log(10)), "note": "a single prior-predictive point with its registered u_input; weak: it carries no input sampling"}
    res = {"what": d["text"]["12_what"], "models": out, "notes": d["text"]["12_notes"]}
    c.register("12_evidence", res, code=[__file__, fm.__file__, c.HERE / "contact.py", c.HERE / "common.py"])
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ 11
def red_rows():
    """Dark C at Ka with the probe on the pads (the private repository's adopted probe model, deep.json) at the handle resistivities around the spec,
    SOL-corrected like the measurement."""
    c.require_private_repo()
    pr, txt = c.device()["deep"]["probe"], c.device()["deep"]["text"]
    probe = pr["tag"]
    sim, rdir = c.WF / "sim", c.WF / "results"
    base = json.load(open(rdir / f"{probe}sipads_u{pr['base_rho']}/meta.json"))["args"]
    for rho in pr["rhos"]:
        tag = f"{probe}sipads_u{rho}"
        if not (rdir / tag / "S11.s1p").exists():
            args = sum(([f"--{k.replace('_', '-')}", str(v)] for k, v in {**base, "rho_handle": float(rho)}.items() if k not in ("dry", "tag")), [])
            log = c.OUT / f"{tag}.log"
            rc = c.run_solver([c.HOME / ".venvs/openems/bin/python", sim / "run_probe.py", *c.replay(args, tag)], sim, log, lock="yield")
            if rc != 0:
                sys.exit(f"{tag}: run_probe failed, rc {rc}")
            c.record_stop(args, tag, log)
    sys.path.insert(0, str(sim))
    import virtual_cal as vc
    f, g_open = vc.read(probe + "open")
    _, g_iss = vc.read(probe + "open_iss")
    _, g_short = vc.read(probe + "short")
    _, g_load = vc.read(probe + "load")
    one, gs, gl = np.ones_like(f), -np.ones_like(f), np.zeros_like(f)
    cals = {"air11": vc.error_terms(f, [g_open, g_short, g_load], [vc.gamma_c(f, c.device()["raw"]["c_open_F"]), gs, gl]),
            "iss": vc.error_terms(f, [g_iss, g_short, g_load], [one, gs, gl])}
    rows = {}
    for rho in [pr["base_rho"]] + pr["rhos"]:
        ft, gt = vc.read(f"{probe}sipads_u{rho}")
        gt = np.interp(f, ft, gt.real) + 1j * np.interp(f, ft, gt.imag)
        rows[str(rho)] = {cv: dict(zip(("X", "Ka"), ({"G_mS": a, "C_fF": b} for a, b in vc.summary(f, vc.correct(gt, e))))) for cv, e in cals.items()}
    lr = np.log([float(r) for r in rows])
    ck = np.array([rows[r]["air11"]["Ka"]["C_fF"] for r in rows])
    order = np.argsort(lr)
    spec = [float(np.interp(np.log(x), lr[order], ck[order])) for x in rho_spec()]
    res = {"what": txt["11_what"], "by_rho": rows, "C_Ka_fF_spec_rho_14_22": spec, "u_num_probe_geometry_fF": pr["u_num_fF"],
           "D2_warming_part": txt["11_D2"], "seen_before": txt["11_seen_before"]}
    c.register("11_red_rows", res, code=[__file__, c.HERE / "common.py"])
    print(json.dumps({r: v["air11"] for r, v in rows.items()}, indent=0), spec)


# ------------------------------------------------------------------------------------------------ 13
def periodic(u_on, tau, tgrid, period):
    """Periodic steady state of tau x' = -x + u for a 50 % square wave of height u_on."""
    a = np.exp(-period / (2 * tau))
    x0 = u_on * a / (1 + a)
    x1 = u_on / (1 + a)
    on = tgrid < period / 2
    with np.errstate(over="ignore"):                     # the branch np.where discards can overflow for tau << period
        return np.where(on, u_on + (x0 - u_on) * np.exp(-tgrid / tau), x1 * np.exp(-(tgrid - period / 2) / tau))


def chopped(s, g, tau_c):
    """Fundamental of the response to a 50 % chopped setting 7, divided by the CW response times the chopping wave's own fundamental: 1 for a fast, linear response."""
    p7 = s["fac"] * fm.power(7, s["ith"])
    out = {}
    for f in CHOP_HZ:
        period = 1.0 / f
        t = np.linspace(0, period, 8192, endpoint=False)
        T = periodic(p7 * s["R"], s["R"] * s["C"], t, period)
        nc = periodic(1.0, tau_c, t, period)
        y = nc * g(T)
        sq = (t < period / 2).astype(float)
        ph = np.exp(-2j * np.pi * t / period)
        h = (np.mean(y * ph) / g(np.array(p7 * s["R"]))) / np.mean(sq * ph)
        out[f"{f:g}"] = [float(abs(h)), float(np.degrees(np.angle(h)))]
    return out


def chopped_set(s, g):
    return {f"tau_c_{tc * 1e6:g}us": chopped(s, g, tc) for tc in (1e-6, 1e-5, 1e-4)}


def protocol():
    c.require_private_repo()
    e = fm.setup()
    mt = e["mats"]["EPW"]
    sc = scenarios(e)
    ith_c, fb = corner_row()
    res = {"what": "13: predictions for the measurements that would settle the question, registered before any of them exists", "scenarios": sc}
    res["laser_P_W_at_setting"] = {name: {str(k): float(fm.power(k, s["ith"])) for k in range(1, 9)} for name, s in sc.items()}
    res["laser_overturn"] = (f"P(setting) at the sample, extrapolated to zero above threshold, crossing at >= {ith_c - 0.05:.2f} A (07c: Ka is linear only up to "
                             f"{max(float(x) for x in prereg('07c_one_laser')['Ka_linear_s4-7_thresholds']):.2f} A, heating needs >= {ith_c:.1f} A)")
    # chip temperature: lumped, and Elmer for the two scenarios (spatial range: coolest and hottest point of the chip)
    con = prereg("08_contact")
    tl = e["tl"]["X"]
    names = [n for n, _, _ in tl]
    res["dT_K_at_sweep_lumped"] = {name: dict(zip(names, map(float, fm.temps(tl, np.array(s["R"]), np.array(s["C"]), s["ith"], s["fac"])))) for name, s in sc.items()}
    dev = c.device()
    chip, t = dev["stack"]["chip_m"], dev["deep"]["handle"]["t_m_nominal"]
    area, cases = chip**2, {}
    for name, s in sc.items():
        train, w = light(name)
        frac = float(erf(np.sqrt(2) * 0.5 * chip / w) ** 2)
        # 06's nominal case with the handle, the beam and the gas gap that gives R_th through the handle and the gap (contact.py's conductivities)
        cases[f"deep_{name}"] = {**dev["thermal_cases"]["nominal"], "t": t, "chip": chip, "gap": max((s["R"] - t / (contact.K_SI * area)) * contact.K_AIR * area, 1e-7),
                                 "w": w, "law": f"amps_Ith{s['ith']:.1f}", "train": train, "afrac": s["fac"] / (train * frac), "T0": T0}
    for cname, cs in cases.items():
        te.run_case(cname, cs, tl, modes=("steady",))     # the time dependence is the lumped model's, which 06 checks against Elmer transients
    el = {}
    for cname, cs in cases.items():
        sd = np.atleast_2d(np.loadtxt(c.OUT / "thermal" / cname / "steady/scalars.dat"))[-1]
        el[cname] = {"inputs": cs, "steady_s7_dT_K_min_mean_max": [float(sd[1] - T0), float(sd[2] - T0), float(sd[0] - T0)]}
    res["elmer"] = el
    res["elmer_note"] = "steady solves at setting 7 for the spatial range; transients were stopped (about 30 s per step) and dT(t) is the lumped model above"
    res["temperature_overturn"] = {"steady_dT7_K_at_least": fb["dT7_K_at_R_min"], "note": f"07b: the smallest rise at setting 7 with which heating reproduces the X day at {ith_c:.1f} A"}
    res["contact_tau_th_s"] = con["tau_th_s"]
    res["chopped_dY_relative"] = {name: chopped_set(s, g_of(mt, s["law"])) for name, s in sc.items()}
    c.register("13_protocol_predictions", res, code=[__file__, fm.__file__, c.HERE / "thermal_elmer.py", c.HERE / "common.py"])
    print(json.dumps({k: res[k] for k in ("laser_P_W_at_setting", "temperature_overturn")}, indent=0)[:1500])


CONTACTS = {"thermal paste under the chip": 0.5, "vacuum on": 2.0, "as measured (the corner's own R_th)": None, "1 mm glass slide under the chip": 30.0}   # K/W (ASM)


def corner_law(e, mt, s):
    """The activated lifetime laws (within 07b's E_a <= 0.40 eV cap) with which heating reproduces the X day, with D1, at the corner's own R_th and C;
    returns the middle one of them and the list."""
    fam, lim = fm.CAPS["Ea<=0.40"]
    meas, lit = e["meas"], e["lit"]["X"]
    T = fm.temps(e["tl"]["X"], np.array(s["R"]), np.array(s["C"]), s["ith"], s["fac"])
    d1 = any(fm.ok_complex(np.nan_to_num(fm.handle_shift(e["ysig"], mt, "X", T[e["i_after"]], sig), nan=np.inf), meas["X"]["after"], meas["X"]["u_after"])
             for sig in mt["sigma"].values())
    P = {k: fm.power(k, s["ith"]) for k in range(3, 8)}
    eas = [float(x) for f, x, tau in fm.g_laws() if f == fam and x <= lim + 1e-9 and d1
           and fm.ok_ratio(fm.tc_ratios(P, {k: mt["mu_sum"](T0 + T[lit[k]]) * tau(T[lit[k]]) for k in range(3, 8)}), meas["X"], range(4, 8))]
    return (fam, eas[len(eas) // 2]), eas


def contact_change():
    """13b (after the independent review). The corner as registered in 13 used the cap E_a = 0.40 eV; at its R_th only the law(s) found here reproduce the
    X day, so the corner is restated with them. Predictions: the X-day ratios when only the thermal contact changes (laser law, beam and timeline fixed, so
    it tests heating without the 2021 laser law; carriers alone predict no change), the rise at setting 7, and the chopped-laser response."""
    c.require_private_repo()
    e = fm.setup()
    mt = e["mats"]["EPW"]
    r13 = prereg("13_protocol_predictions")
    s = dict(r13["scenarios"]["corner"])
    capped = g_of(mt, s["law"])
    s["law"], eas = corner_law(e, mt, s)
    g = g_of(mt, s["law"])
    tl, lit = e["tl"]["X"], e["lit"]["X"]
    P = {k: fm.power(k, s["ith"]) for k in (3, 5, 7)}
    ratios = lambda T, g: fm.tc_ratios(P, {k: g(T[lit[k]]) for k in (3, 5, 7)}, (5, 7))
    rows = {}
    for name, R in CONTACTS.items():
        R = s["R"] if R is None else R
        T = fm.temps(tl, np.array(R), np.array(s["C"]), s["ith"], s["fac"])
        rows[name] = {"R_th_K_W": R, "dT7_K": float(T[lit[7]]), **{f"s{k}/s3": float(v) for k, v in ratios(T, g).items()}}
    T_own = fm.temps(tl, np.array(s["R"]), np.array(s["C"]), s["ith"], s["fac"])
    res = {"what": "13b: the corner restated with the lifetime law that reproduces the X day at its R_th, and its predictions: X-day |dY| ratios when only the "
                   "chip-to-chuck contact changes (carriers alone: unchanged), and the chopped-laser response (registered before any such measurement)",
           "corner": s, "E_a_passing_at_R_th_eV": eas,
           "why": f"13's corner used E_a = 0.40 eV (the cap), which gives s7/s3 {float(ratios(T_own, capped)[7]):.1f} at the corner's R_th, outside the 00d band around the "
                  f"measured {e['meas']['X']['ratio'][7]:.1f}",
           "rows": rows, "chopped_dY_relative_corner": chopped_set(s, g)}
    c.register("13b_contact_change", res, code=[__file__, fm.__file__, c.HERE / "common.py"])
    print(json.dumps(rows, indent=1))
    amended = {**r13, "scenarios": {**r13["scenarios"], "corner": s}, "chopped_dY_relative": {**r13["chopped_dY_relative"], "corner": res["chopped_dY_relative_corner"]}}
    write_protocol(amended, res)


def write_protocol(r, b):
    """out/PROTOCOL.md from 13 with 13b's restated corner overlaid (r), and 13b itself (b); the only writer of the file. The tables are built here, the
    prose (equipment, dates, the steps) is the private templates/PROTOCOL.md."""
    lp, chop = r["laser_P_W_at_setting"], {name: r["chopped_dY_relative"][name]["tau_c_10us"] for name in ("nominal", "corner")}
    cell = lambda a: f"{a[0]:.3f} at {a[1]:+.1f} deg"
    rows = {"contact_rows": [f"| {name} | {v['R_th_K_W']:.1f} | {v['s5/s3']:.2f} | {v['s7/s3']:.2f} | {v['dT7_K']:.1f} |" for name, v in b["rows"].items()],
            "laser_rows": [f"| {k} | {lp['nominal'][str(k)]:.3f} | {lp['corner'][str(k)]:.3f} |" for k in range(1, 9)],
            "elmer_rows": ["| {} | {:.2f} / {:.2f} / {:.2f} |".format(cname[5:], *v["steady_s7_dT_K_min_mean_max"]) for cname, v in r["elmer"].items()],
            "chopped_rows": [f"| {f:g} | {cell(chop['nominal'][f'{f:g}'])} | {cell(chop['corner'][f'{f:g}'])} |" for f in CHOP_HZ]}
    text = c.template("PROTOCOL.md").format(r=r, n=r["scenarios"]["nominal"], k=r["scenarios"]["corner"], chop10_nominal=chop["nominal"]["10"][0],
                                            chop10_corner=chop["corner"]["10"][0], **{name: "\n".join(v) for name, v in rows.items()})
    (c.OUT / "PROTOCOL.md").write_text(text)


if __name__ == "__main__":
    {"fingerprints": fingerprints, "gaas": gaas, "evidence": evidence, "redrows": red_rows, "protocol": protocol, "contact": contact_change}[sys.argv[1]]()

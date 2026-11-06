"""Si conductivity from first principles at the actual carrier density: thermally broadened Drude weight, EPW scattering, sigma(omega, T).

Si at these densities is non-degenerate, so no Fermi surface exists at the device's carrier density. The analogue of the Fermi-surface Drude
weight is the thermal average D(T) = (2/V) sum_nk (-df/dE) v_x^2 on the thermally occupied band edge. It is computed two ways:
  A. from the Wannier Hamiltonian (si_hr.dat, Hellmann-Feynman velocities, dense grids around the band edges): D, DOS, conductivity mass;
  B. from EPW's own fine grid (IBTEvel_sup.fmt + inv_tau.fmt), with the state-resolved SERTA rates, which also gives
     sigma(omega) = e^2 D-weighted sum of tau/(1 - i omega tau), checked against EPW's printed mobility and with Kramers-Kronig.
PBE's gap is not used for n_i(T): Eg(T) is Thurmond's fit, labelled ASM, with the AI densities of states. No spin-orbit in this Hamiltonian.

  materials_si.py        registers prereg/02_materials.json once the EPW series exists
"""
import re
from pathlib import Path

import numpy as np

import common as c

BASE = c.HOME / "qe-test/Si_EPW"
RY = 13.605693122994            # eV
A0 = 0.529177210903             # Angstrom
KB = 8.617333262e-5             # eV/K
HB2_M = 7.619964231             # hbar^2/m_e in eV Angstrom^2: converts <(hbar v)^2>/kT into a mass ratio
ALAT = 10.2631 * A0             # Angstrom, from scf.out
OMEGA = ALAT**3 / 4             # cell volume, Angstrom^3
TEMPS = np.array([300.0, 350.0, 400.0, 450.0, 500.0])


def read_hr(path):
    """R vectors (crystal), degeneracies and H(R) [eV] from a Wannier90 _hr.dat."""
    with open(path) as f:
        f.readline()
        nw, nr = int(f.readline()), int(f.readline())
        deg = []
        while len(deg) < nr:
            deg += [int(x) for x in f.readline().split()]
        data = np.loadtxt(f)
    r = data[::nw * nw, :3].astype(int)
    h = (data[:, 5] + 1j * data[:, 6]).reshape(nr, nw, nw).transpose(0, 2, 1)     # the file loops over n slowly, then m
    return r, np.array(deg), h


class Bands:
    """E(k) and velocity for Cartesian k (1/Angstrom). fcc lattice vectors a/2 (-1,0,1),(0,1,1),(-1,1,0), QE ibrav=2."""
    def __init__(self, hr):
        self.r, self.deg, self.h = read_hr(hr)
        a = ALAT / 2 * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], float)
        self.Rc = self.r @ a

    def eval(self, k):
        k = np.atleast_2d(k)
        ph = np.exp(1j * k @ self.Rc.T) / self.deg
        hk = np.einsum("kr,rij->kij", ph, self.h, optimize=True)
        e, u = np.linalg.eigh(hk)
        v = np.zeros((*e.shape, 3))
        for ax in range(3):
            dh = np.einsum("kr,rij->kij", 1j * ph * self.Rc[:, ax], self.h, optimize=True)
            v[..., ax] = np.einsum("kin,kij,kjn->kn", u.conj(), dh, u, optimize=True).real
        return e, v                                         # eV, eV Angstrom (= hbar v)


def wedge(hi, n, valley=None):
    """The octant at Gamma with weight x8 (holes, side hi); or one valley's quarter box along +x, with weight x4 x6 valleys (electrons)."""
    if valley is None:
        g = (np.arange(n) + 0.5) * hi / n
        kx, ky, kz = np.meshgrid(g, g, g, indexing="ij")
        pts = np.c_[kx.ravel(), ky.ravel(), kz.ravel()]
        return pts, np.full(len(pts), (hi / n) ** 3 * 8.0)
    k0, half_l, half_t, kx = valley
    dl = (kx - (k0 - half_l)) / n                           # up to the zone boundary at X: beyond it the states belong to the next valley
    gl = k0 - half_l + (np.arange(n) + 0.5) * dl
    gt = (np.arange(n // 2) + 0.5) * half_t / (n // 2)
    kx, ky, kz = np.meshgrid(gl, gt, gt, indexing="ij")
    pts = np.c_[kx.ravel(), ky.ravel(), kz.ravel()]
    return pts, np.full(len(pts), dl * (half_t / (n // 2)) ** 2 * 4.0 * 6.0)   # 4 quadrants, 6 valleys


def _edge(bands, pts, lo, hi):
    """Energies and (hbar v)^2/3 of bands lo..hi-1, in batches."""
    es, vs = [], []
    for i in range(0, len(pts), 2000):
        e, v = bands.eval(pts[i:i + 2000])
        es.append(e[:, lo:hi])
        vs.append((v[:, lo:hi] ** 2).sum(-1) / 3)
    return np.concatenate(es), np.concatenate(vs)


def band_tables(bands):
    """Band-edge tables (E, v^2/3, weight per volume) for holes and electrons, cached."""
    cache = c.OUT / "bands_edge.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d}
    bz = (2 * np.pi) ** 3
    pts, w = wedge(0.45, 44)                                # holes: octant around Gamma, 3 top valence bands
    eh, vh = _edge(bands, pts, 1, 4)
    # electrons: the conduction minimum along Gamma-X
    g = np.linspace(0.5, 1.0, 101) * 2 * np.pi / ALAT
    e, _ = bands.eval(np.c_[g, 0 * g, 0 * g])
    k0 = g[np.argmin(e[:, 4])]
    pts_e, w_e = wedge(0.5, 44, valley=(k0, 0.45, 0.28, 2 * np.pi / ALAT))
    ee, ve = _edge(bands, pts_e, 4, 6)
    out = dict(eh=eh, vh=vh, wh=w / bz, ee=ee, ve=ve, we=w_e / bz, k0=np.array(k0))
    np.savez(cache, **out)
    return out


def boltz_weights(table, T):
    """Non-degenerate densities and Drude weights per cubic cm per spin-degenerate pair, with Boltzmann statistics."""
    kt = KB * T
    vbm, cbm = table["eh"].max(), table["ee"].min()
    fh = np.exp((table["eh"] - vbm) / kt)
    fe = np.exp(-(table["ee"] - cbm) / kt)
    wh = table["wh"][:, None] * 2.0
    we = table["we"][:, None] * 2.0
    # weights: integrals d^3k/(2 pi)^3 per Angstrom^3 -> cm^-3
    to_cm3 = 1e24
    Nv, Dv = (wh * fh).sum() * to_cm3, (wh * fh * table["vh"]).sum() * to_cm3 / kt
    Nc, Dc = (we * fe).sum() * to_cm3, (we * fe * table["ve"]).sum() * to_cm3 / kt
    return dict(Nv=Nv, Nc=Nc, Dv=Dv, Dc=Dc, m_cond_h=Nv / Dv * HB2_M, m_cond_e=Nc / Dc * HB2_M, vbm=vbm, cbm=cbm)


def thurmond_gap(T):
    return 1.1695 - 4.73e-4 * T**2 / (T + 636.0)


def epw_mob(path):
    """{T: (SERTA, IBTE)} drift mobility, trace/3, from one EPW output: the first and last mobility table for each temperature."""
    t = Path(path).read_text(errors="ignore")
    num = r"\s+(\S+)"
    row = re.compile(r"^\s+(\d+\.\d{3})\s+\S+\s+\S+\s+\S+" + num + r"\s+\S+\s+\S+\s*\n\s+\S+\s+\S+" + num + r"\s+\S+\s*\n\s+\S+\s+\S+\s+\S+" + num + r"\s*$", re.M)
    rows = {}
    for piece in re.split(r"Drift (?:Elec|Hole) mobility", t)[1:]:
        for m in row.finditer(piece):
            rows.setdefault(float(m[1]), []).append((float(m[2]) + float(m[3]) + float(m[4])) / 3)
    return {T: (v[0], v[-1]) for T, v in rows.items()}


def epw_states(d):
    """Fine-grid states of one EPW transport directory, carrier's bands only: E [eV], weight, (hbar v)^2/3 [eV^2 A^2], and per temperature the Fermi level [eV] and rate [1/s]."""
    d = Path(d)
    lines = open(d / "IBTEvel_sup.fmt").read().splitlines()
    nh, ne = map(int, lines[1].split())
    carrier = "h" if nh else "e"
    evbm, ecbm = [float(x) * RY for x in lines[3].split()]
    i, efs = 5, []
    while not lines[i].startswith("#"):
        p = lines[i].split()
        efs.append(float(p[1 if carrier == "h" else 2]) * RY)
        i += 1
    dat = np.array([ln.split() for ln in lines[i + 1:] if ln.strip()], float)
    e, wk = dat[:, 5] * RY, dat[:, 6]
    v2 = (dat[:, 2:5] ** 2).sum(1) / 3 * (RY * A0) ** 2
    rates = np.loadtxt(d / ("inv_tau.fmt" if carrier == "h" else "inv_taucb.fmt"), comments="#")
    nt = len(efs)
    rate = (rates[:, 4] * RY / 6.582119569e-16).reshape(nt, -1)           # 1/s; the file's Ry value is hbar/tau
    mid = 0.5 * (evbm + ecbm)
    keep = e < mid if carrier == "h" else e > mid                           # a wide window also holds the other carrier's band
    return dict(carrier=carrier, e=e[keep], w=wk[keep], v2=v2[keep], efs=efs, rate=rate[:, keep])


def epw_mobility_omega(st, itemp, T, omega):
    """Complex mobility (cm^2/Vs) and Drude-weight mass: mu(w) = (1/hbar^2) Sum w (-df/dE) (hbar v)^2/3 tau/(1 - i w tau) / Sum w f_carrier. States with zero rate get tau = 0, as in EPW."""
    r = st["rate"][itemp]
    tau = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
    kt, ef = KB * T, st["efs"][itemp]
    f = 1.0 / (np.exp(np.clip((st["e"] - ef) / kt, -500, 500)) + 1.0)
    occ = 1 - f if st["carrier"] == "h" else f
    dfde = f * (1 - f) / kt
    n = (st["w"] * occ).sum()
    hb = 6.582119569e-16
    mu = np.array([((st["w"] * dfde * st["v2"] * tau / (1 - 1j * w * tau)).sum()) / hb**2 / n * 1e-16 for w in np.atleast_1d(omega)])
    D = (st["w"] * dfde * st["v2"]).sum() / n                 # (hbar v)^2/3 thermal average per carrier / kT: hbar^2/m_c
    return mu, HB2_M / D, float(np.sum(st["w"] * dfde * st["v2"] * tau) / np.sum(st["w"] * dfde * st["v2"]))


def kk_residual(st, itemp, T):
    """Rms of Im[mu(w)] from the Kramers-Kronig transform of Re[mu(w)], against Im from the state sum, over the band that carries the weight."""
    w = np.logspace(9.5, 15.5, 1201)
    mu, _, _ = epw_mobility_omega(st, itemp, T, w)
    im_kk = np.empty(w.size)
    lw = np.log(w)
    for i, wi in enumerate(w):
        # subtract the pole: P Int_0^inf dw / (w^2 - wi^2) = 0, so Re(w) - Re(wi) leaves a smooth integrand
        den = w**2 - wi**2
        den[i] = 1.0
        integrand = (mu.real - mu.real[i]) * w / den
        integrand[i] = 0.5 * (integrand[i - 1] + integrand[i + 1]) if 0 < i < w.size - 1 else 0.0
        im_kk[i] = -(2 * wi / np.pi) * np.trapezoid(integrand, lw)
    k = (w > 1e11) & (w < 1e14)
    return float(np.sqrt(np.mean((im_kk[k] - mu.imag[k]) ** 2)) / np.sqrt(np.mean(mu.imag[k] ** 2)))


def sigma_table(tt, mu_h, mu_e, ni, rho, carrier="p"):
    """sigma(T), S/m, of a uniformly doped layer whose 300 K resistivity is rho (ohm cm): N from rho and the AI mobility, then charge neutrality at each T."""
    i300 = int(np.argmin(abs(tt - 300.0)))
    mu_maj = mu_h if carrier == "p" else mu_e
    N = 1.0 / (1.602176634e-19 * mu_maj[i300] * rho)
    maj = 0.5 * (N + np.sqrt(N**2 + 4 * ni**2))
    mino = ni**2 / maj
    sig = 1.602176634e-19 * (maj * mu_maj + mino * (mu_e if carrier == "p" else mu_h)) * 100.0
    return N, sig


def main():
    c.require_private_repo()
    bands = Bands(BASE / "work/si_hr.dat")
    tb = band_tables(bands)
    wan = {}
    ext_T = np.arange(290.0, 701.0, 5.0)
    nc, nv, ni = [], [], []
    for T in ext_T:
        w = boltz_weights(tb, T)
        nc.append(w["Nc"]); nv.append(w["Nv"])
        ni.append(np.sqrt(w["Nv"] * w["Nc"]) * np.exp(-thurmond_gap(T) / (2 * KB * T)))
        if T in TEMPS:
            wan[f"{T:g}"] = {"m_cond_h": w["m_cond_h"], "m_cond_e": w["m_cond_e"]}
    ni = np.array(ni)

    # controls: the fast build against unmodified EPW, printed digits at 300 and 500 K
    control = {}
    for carrier in "he":
        up, tr = epw_mob(BASE / f"mob/psC_{carrier}24_up/epw.out"), epw_mob(BASE / f"mob/psC_{carrier}24_tr/epw.out")
        control[carrier] = {f"{T:g}": {"unmodified": up[T], "fast": tr[T], "rel_diff_serta": abs(up[T][0] / tr[T][0] - 1), "rel_diff_ibte": abs(up[T][1] / tr[T][1] - 1)} for T in up}

    series, drude = {}, {}
    w_rf = 2 * np.pi * np.array([10e9, 31.5e9])
    kk = {}
    for carrier, d in (("h", BASE / "mob/psT_h48"), ("e", BASE / "mob/psT_e48")):
        mob = epw_mob(d / "epw.out")
        series[carrier] = {f"{T:g}": {"serta": mob[T][0], "ibte": mob[T][1]} for T in TEMPS}
        st = epw_states(d)
        rows = {}
        for i, T in enumerate(TEMPS):
            mu, mc, tau = epw_mobility_omega(st, i, T, np.r_[0.0, w_rf])
            rows[f"{T:g}"] = {"mu_state_sum": mu[0].real, "m_cond": mc, "tau_tr_fs": tau * 1e15, "omega_tau_10GHz": w_rf[0] * tau, "omega_tau_31p5GHz": w_rf[1] * tau,
                              "re_mu_ratio_31p5GHz": mu[2].real / mu[0].real}
        drude[carrier] = rows
        kk[carrier] = kk_residual(st, 0, 300.0)

    tt = ext_T
    mu_ib = {x: np.array([series[x][f"{T:g}"]["ibte"] for T in TEMPS]) for x in "he"}
    slopes = {x: float(np.polyfit(np.log(TEMPS), np.log(mu_ib[x]), 1)[0]) for x in "he"}
    mu_t = {x: np.exp(np.interp(np.log(tt), np.log(TEMPS), np.log(mu_ib[x]), right=np.nan)) for x in "he"}
    for x in "he":                                  # beyond 500 K: the fitted power law, flagged in the file
        far = tt > TEMPS[-1]
        mu_t[x][far] = mu_ib[x][-1] * (tt[far] / TEMPS[-1]) ** slopes[x]
        near = tt < TEMPS[0]
        mu_t[x][near] = mu_ib[x][0] * (tt[near] / TEMPS[0]) ** slopes[x]

    layers = {}
    for rho in c.device()["rhos_ohm_cm"]:
        N, sig = sigma_table(tt, mu_t["h"], mu_t["e"], ni, rho)
        i300 = int(np.argmin(abs(tt - 300.0)))
        turn = tt[int(np.argmin(sig))]
        layers[f"{rho:g}"] = {"N_cm3": N, "sigma_S_m": sig.tolist(), "T_min_sigma_K": float(turn), "dlnsigma_dT_300K": float(np.gradient(np.log(sig), tt)[i300])}
    T_equal_dark = {rho: float(tt[np.argmax((np.array(v["sigma_S_m"]) > v["sigma_S_m"][2]) & (tt > 320))]) for rho, v in layers.items()}

    res = {"what": "Si conductivity from the AI Drude weight and EPW scattering; n_i uses AI DOS and Thurmond's gap",
           "T_K": tt.tolist(), "mu_h_ibte": mu_t["h"].tolist(), "mu_e_ibte": mu_t["e"].tolist(), "n_i_cm3": ni.tolist(), "Nc_cm3": nc, "Nv_cm3": nv,
           "mobility_table": series, "power_law_exponents": slopes, "drude_epw_route": drude, "drude_wannier_route": wan, "control_fast_vs_unmodified_24cube": control,
           "kk_rms_rel": kk, "layers_vs_rho": layers, "T_where_sigma_recovers_its_300K_value": T_equal_dark,
           "notes": ["No spin-orbit in the Hamiltonian: hole masses and mu_h carry this.", "48^3 fine grid; holes +17 % to 120^3 in PLAN item 23 (with quadrupoles), electrons about 0.",
                     "Beyond 500 K mobility is the fitted power law, not an EPW result."]}
    c.register("02_materials", res, code=[__file__, c.HERE / "common.py"])
    for x in "he":
        print(x, "IBTE mu(T)", mu_ib[x].round(1), "exponent", round(slopes[x], 2), "m_c EPW", round(drude[x]["300"]["m_cond"], 3), "m_c Wannier", round(wan["300"]["m_cond_h" if x == "h" else "m_cond_e"], 3),
              "omega tau 10/31.5", round(drude[x]["300"]["omega_tau_10GHz"], 4), round(drude[x]["300"]["omega_tau_31p5GHz"], 4), "KK", round(kk[x], 4))
    print("control", {k: {t: (round(v["rel_diff_serta"], 8), round(v["rel_diff_ibte"], 8)) for t, v in cv.items()} for k, cv in control.items()})
    print({k: (round(v["N_cm3"] / 1e15, 3), round(v["dlnsigma_dT_300K"], 5)) for k, v in layers.items()}, T_equal_dark)


if __name__ == "__main__":
    main()

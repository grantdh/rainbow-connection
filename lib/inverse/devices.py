"""Catalog of 2D silicon-photonic inverse-design problems (Si core in SiO2 cladding, Ez polarization).

Each builder returns a Device: background permittivity with the access waveguides, the design window,
the ports, and the transmission targets the optimizer drives toward. All lengths in um.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from fdfd import Grid, Port

EPS_SI = 3.48**2
EPS_SIO2 = 1.444**2


@dataclass(frozen=True)
class Term:
    """Drive T(wavelength; src port/mode -> mon port/mode) toward `target` with weight `weight`."""
    wavelength: float
    src: str
    src_mode: int
    mon: str
    mon_mode: int
    target: float
    weight: float = 1.0

    @property
    def label(self) -> str:
        return f"{self.src}{self.src_mode}->{self.mon}{self.mon_mode} @{self.wavelength * 1e3:.0f}nm"


@dataclass
class Device:
    kind: str
    grid: Grid
    eps_bg: np.ndarray
    design: tuple[slice, slice]
    ports: dict[str, Port]
    terms: list[Term]
    eps_min: float = EPS_SIO2
    eps_max: float = EPS_SI
    params: dict = field(default_factory=dict)

    @property
    def design_shape(self) -> tuple[int, int]:
        sx, sy = self.design
        return sx.stop - sx.start, sy.stop - sy.start

    def eps(self, rp: np.ndarray) -> np.ndarray:
        """Design-window permittivity of a density in [0, 1]."""
        return self.eps_min + rp * (self.eps_max - self.eps_min)

    def eps_full(self, rp: np.ndarray) -> np.ndarray:
        e = self.eps_bg.copy()
        e[self.design] = self.eps(rp)
        return e


DEFAULTS = {
    "bend": dict(wavelength=1.55, width=0.5, design=3.2, bandwidth=0.0),
    "splitter": dict(wavelength=1.55, width=0.5, design=3.2, ratio=0.5, separation=1.6, bandwidth=0.0),
    "mode_converter": dict(wavelength=1.55, width=0.5, width_out=1.0, design=3.2, bandwidth=0.0),
    "demux": dict(wavelength=1.55, wavelength2=1.31, width=0.5, design=3.2, separation=1.6, crosstalk=0.5,
                  bandwidth=0.02),
}


def build(kind: str, dx: float = 0.04, pml: float = 1.0, margin: float = 0.8, **overrides) -> Device:
    p = {**DEFAULTS[kind], **overrides}
    L = p["design"]
    npml = int(round(pml / dx))
    nd = int(round(L / dx))
    nm = int(round(margin / dx))
    n = nd + 2 * (nm + npml)
    grid = Grid(n, n, dx, npml)
    c = n // 2
    d0, d1 = c - nd // 2, c - nd // 2 + nd
    design = (slice(d0, d1), slice(d0, d1))
    eps = np.full((n, n), EPS_SIO2)
    hw = lambda w: max(1, int(round(w / dx / 2)))  # noqa: E731  half-width in cells
    sep = lambda s: int(round(s / dx / 2))         # noqa: E731
    src_i = npml + 4
    out_i = n - npml - 5
    win = int(round(1.0 / dx))  # half-span of a port line

    def port(name, axis, index, centre):
        return Port(name, axis, index, max(npml, centre - win), min(n - npml, centre + win))

    def guide_x(x0, x1, yc, w):  # horizontal guide
        eps[x0:x1, yc - hw(w):yc + hw(w)] = EPS_SI

    def guide_y(y0, y1, xc, w):  # vertical guide
        eps[xc - hw(w):xc + hw(w), y0:y1] = EPS_SI

    w = p["width"]
    ports = {"in": port("in", "x", src_i, c)}
    guide_x(0, d0, c, w)
    lam = p["wavelength"]
    if kind == "bend":
        guide_y(0, d0, c, w)
        ports["out"] = port("out", "y", src_i, c)  # bottom edge, light leaves toward -y
        terms = [Term(lam, "in", 0, "out", 0, 1.0)]
    elif kind == "mode_converter":
        guide_x(d1, n, c, p["width_out"])
        ports["out"] = port("out", "x", out_i, c)
        terms = [Term(lam, "in", 0, "out", 1, 1.0)]
    elif kind in ("splitter", "demux"):  # two output arms at +-separation/2
        s = sep(p["separation"])
        guide_x(d1, n, c + s, w)
        guide_x(d1, n, c - s, w)
        ports["top"] = port("top", "x", out_i, c + s)
        ports["bot"] = port("bot", "x", out_i, c - s)
        if kind == "splitter":
            r = float(p["ratio"])
            terms = [Term(lam, "in", 0, "top", 0, r), Term(lam, "in", 0, "bot", 0, 1.0 - r)]
        else:
            lam2, xt = p["wavelength2"], float(p["crosstalk"])
            terms = [Term(lam, "in", 0, "top", 0, 1.0), Term(lam2, "in", 0, "bot", 0, 1.0),
                     Term(lam, "in", 0, "bot", 0, 0.0, xt), Term(lam2, "in", 0, "top", 0, 0.0, xt)]
    else:
        raise ValueError(kind)
    bw = float(p.get("bandwidth", 0.0))
    if bw > 0:
        # Hold each target over lambda +- bw, not at one point: the 40 nm grid's numerical dispersion shifts narrowband
        # responses ~18 nm bluer on finer grids (README, MEEP check), so a single-point optimum detunes.
        terms = [replace(t, wavelength=round(t.wavelength + o, 4), weight=t.weight / 3)
                 for t in terms for o in (-bw, 0.0, bw)]
    return Device(kind, grid, eps, design, ports, terms, params=p)


def straight() -> tuple[Device, np.ndarray]:
    """Baseline: a straight 500 nm guide through the window, TE0 -> TE0 at 1550 nm (expect T ~ 1, R ~ 0)."""
    dev = build("mode_converter", width_out=0.5)
    dev.terms = [Term(1.55, "in", 0, "out", 0, 1.0)]
    rp = np.zeros(dev.design_shape)
    c, hw = rp.shape[1] // 2, int(round(0.25 / dev.grid.dx))
    rp[:, c - hw:c + hw] = 1
    return dev, rp

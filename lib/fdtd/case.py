"""Load, validate, hash and rasterise MUPPET FDTD case files (schema muppet-fdtd-case/0).

The contract is docs/FDTD_BACKEND_SPEC.md. Units are SI. Grid indices follow the spec's Yee convention:
nodes at integer indices 0..N, and on a 3-D grid

    Ex(i,j,k) at (i+1/2, j, k)    Hx(i,j,k) at (i, j+1/2, k+1/2)
    Ey(i,j,k) at (i, j+1/2, k)    Hy(i,j,k) at (i+1/2, j, k+1/2)
    Ez(i,j,k) at (i, j, k+1/2)    Hz(i,j,k) at (i+1/2, j+1/2, k)

in units of dx, with the domain [0, N dx] on each axis. A 2-D case (dimensions = 2) is TM-z: only Ez, Hx and Hy,
with Ez(i,j) at (i, j), Hx(i,j) at (i, j+1/2), Hy(i,j) at (i+1/2, j), and index arrays of width 2.

CLI:  python3 lib/fdtd/case.py CASE.json [...]   validates each case and prints its derived quantities.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA = "muppet-fdtd-case/0"
E_COMPONENTS = ("Ex", "Ey", "Ez")
H_COMPONENTS = ("Hx", "Hy", "Hz")
FACES = ("x-", "x+", "y-", "y+", "z-", "z+")
FACES_2D = ("x-", "x+", "y-", "y+")
TM_E, TM_H = ("Ez",), ("Hx", "Hy")

# Offset of each component from its integer index, in cells (the staggering above).
STAGGER = {
    "Ex": (0.5, 0.0, 0.0), "Ey": (0.0, 0.5, 0.0), "Ez": (0.0, 0.0, 0.5),
    "Hx": (0.0, 0.5, 0.5), "Hy": (0.5, 0.0, 0.5), "Hz": (0.5, 0.5, 0.0),
}
STAGGER_2D = {"Ez": (0.0, 0.0), "Hx": (0.0, 0.5), "Hy": (0.5, 0.0)}

# pec_cells conventions: which three E samples a "cell" (i,j,k) owns, as offsets to spec indices.
# e_minus_half is the Swift programs' layout (LoopFDTD dipole_s11.swift, and the FDTD.metal applyPEC mask):
# their updateE/updateH stencils put Ex(i) at x = i - 1/2, Ey(j) at y = j - 1/2, Ez(k) at z = k - 1/2,
# so their Ex(i,j,k) is the spec's Ex(i-1,j,k), and so on.
PEC_CELL_CONVENTIONS = {
    "e_plus_half": {"Ex": (0, 0, 0), "Ey": (0, 0, 0), "Ez": (0, 0, 0)},
    "e_minus_half": {"Ex": (-1, 0, 0), "Ey": (0, -1, 0), "Ez": (0, 0, -1)},
}
# In 2-D TM a cell owns only its Ez(i, j), whatever the 3-D layout (there is no z staggering).
PEC_CELL_CONVENTIONS_2D = {"e_plus_half": {"Ez": (0, 0)}, "e_minus_half": {"Ez": (0, 0)}}


def stagger(component: str, dim: int) -> tuple:
    table = STAGGER if dim == 3 else STAGGER_2D
    if component not in table:
        raise CaseError(f"component {component} does not exist in a {dim}-D case")
    return table[component]


def pec_conventions(dim: int) -> dict:
    return PEC_CELL_CONVENTIONS if dim == 3 else PEC_CELL_CONVENTIONS_2D


class CaseError(ValueError):
    pass


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise CaseError(f"{where}: missing '{key}'")
    return d[key]


def index_range(component: str, n: list[int]) -> list[tuple[int, int]]:
    """Inclusive index bounds of one field component on an n-cell grid (2-D or 3-D, from len(n))."""
    off = stagger(component, len(n))
    return [(0, n[a] - 1) if off[a] else (0, n[a]) for a in range(len(n))]


def position(component: str, idx: np.ndarray) -> np.ndarray:
    """Positions (units of dx) of samples idx (M, d) of a component; d = 2 or 3 from the array width."""
    arr = np.asarray(idx, dtype=float)
    return arr + np.asarray(stagger(component, arr.shape[-1]))


def box_indices(ranges) -> np.ndarray:
    """All integer points of an inclusive box [(lo, hi), ...] as an (M, d) int64 array, first axis slowest."""
    grids = np.meshgrid(*[np.arange(lo, hi + 1) for lo, hi in ranges], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1).astype(np.int64)


def _load_indices(spec, base: Path, where: str, dim: int = 3) -> np.ndarray:
    """An index set: inline [[i,j,k], ...], an inclusive box {"lo": [...], "hi": [...]}, or an .npy file."""
    if isinstance(spec, str):
        arr = np.load(base / spec)
    elif isinstance(spec, dict):
        lo, hi = _req(spec, "lo", where), _req(spec, "hi", where)
        if any(h < l for l, h in zip(lo, hi)):
            raise CaseError(f"{where}: box hi < lo")
        arr = box_indices(list(zip(lo, hi)))
    else:
        arr = np.asarray(spec)
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size % dim or (arr.ndim == 2 and arr.shape[1] != dim):
        raise CaseError(f"{where}: indices must have {dim} entries each in a {dim}-D case")
    arr = arr.reshape(-1, dim)
    if len(arr) == 0:
        raise CaseError(f"{where}: empty index set")
    return arr


def _check_in_range(component: str, idx: np.ndarray, n: list[int], where: str) -> None:
    for a, (lo, hi) in enumerate(index_range(component, n)):
        bad = (idx[:, a] < lo) | (idx[:, a] > hi)
        if bad.any():
            raise CaseError(f"{where}: {component} index {idx[bad][0].tolist()} outside "
                            f"{index_range(component, n)} on a {n} grid")


class Case:
    """A validated case. `raw` is the JSON as loaded; everything else is derived from it."""

    def __init__(self, raw: dict, path: Path | None = None):
        self.raw = raw
        self.path = path
        self.base = path.parent if path else Path.cwd()
        self.files: dict[str, str] = {}        # referenced .npy -> sha256, folded into the case hash
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "Case":
        path = Path(path)
        return cls(json.loads(path.read_text()), path)

    # ---- validation -------------------------------------------------------------------------------

    def _validate(self) -> None:
        r = self.raw
        if r.get("schema") != SCHEMA:
            raise CaseError(f"schema must be {SCHEMA!r}, got {r.get('schema')!r}")
        _req(r, "name", "case")
        self.dim = r.get("dimensions", 3)
        if self.dim not in (2, 3):
            raise CaseError("dimensions must be 2 (TM-z: Ez, Hx, Hy) or 3")
        self.e_comps, self.h_comps = (E_COMPONENTS, H_COMPONENTS) if self.dim == 3 else (TM_E, TM_H)
        g = _req(r, "grid", "case")
        self.n = [int(v) for v in _req(g, "n", "grid")]
        self.dx = float(_req(g, "dx_m", "grid"))
        if len(self.n) != self.dim or min(self.n) < 2 or self.dx <= 0:
            raise CaseError(f"grid: n must be {self.dim} integers >= 2 and dx_m > 0")
        t = _req(r, "time", "case")
        self.courant = float(_req(t, "courant", "time"))
        self.c = float(_req(t, "c_m_per_s", "time"))
        self.steps = int(_req(t, "steps", "time"))
        if not 0 < self.courant <= 1 / math.sqrt(self.dim):
            raise CaseError(f"time: courant {self.courant} outside (0, 1/sqrt({self.dim})] ({self.dim}-D stability)")
        if r.get("precision", "any") not in ("fp32", "fp64", "any"):
            raise CaseError("precision must be fp32, fp64 or any")
        b = _req(r, "boundaries", "case")
        faces = FACES if self.dim == 3 else FACES_2D
        if sorted(b) != sorted(faces):
            raise CaseError(f"boundaries: need exactly the faces {faces}")
        self.boundary = {face: kind if isinstance(kind, str) else kind.get("type") for face, kind in b.items()}
        for face, k in self.boundary.items():
            if k not in ("pec", "pml"):
                raise CaseError(f"boundaries.{face}: '{k}' is not pec or pml (periodic is reserved)")
        mats = _req(r, "materials", "case")
        for name, m in mats.items():
            _req(m, "eps_r", f"materials.{name}")
        if r.get("background", "vacuum") not in mats:
            raise CaseError("background must name a material")
        self.background = mats[r.get("background", "vacuum")]

        self.pec: dict[str, list[np.ndarray]] = {c: [] for c in self.e_comps}
        self.material_boxes: list[tuple[str, list, list]] = []
        for i, reg in enumerate(r.get("regions", [])):
            self._region(reg, f"regions[{i}]", mats)
        self.pec_samples = {c: np.unique(np.concatenate(v), axis=0) if v else np.zeros((0, self.dim), np.int64)
                            for c, v in self.pec.items()}

        # Index sets of sources (parallel to raw["sources"]) and probes (by name), loaded once here.
        self.source_samples = [self._source(s, f"sources[{i}]") for i, s in enumerate(r.get("sources", []))]
        self.probe_samples: dict[str, np.ndarray] = {}
        self.probe_component: dict[str, str] = {}
        for i, p in enumerate(r.get("probes", [])):
            where = f"probes[{i}]"
            name = _req(p, "name", where)
            if name in self.probe_samples:
                raise CaseError(f"{where}: duplicate probe name {name!r}")
            self.probe_samples[name] = self._probe(p, where)
            self.probe_component[name] = p["component"]

    def _region(self, reg: dict, where: str, mats: dict) -> None:
        kind = _req(reg, "kind", where)
        if kind == "box":
            if reg.get("material") not in mats:
                raise CaseError(f"{where}: material {reg.get('material')!r} is not defined")
            self.material_boxes.append((reg["material"], _req(reg, "lo", where), _req(reg, "hi", where)))
        elif kind == "pec_samples":
            comp = _req(reg, "component", where)
            if comp not in self.e_comps:
                raise CaseError(f"{where}: component must be one of {self.e_comps}")
            idx = self._indices(reg, where)
            _check_in_range(comp, idx, self.n, where)
            self.pec[comp].append(idx)
        elif kind == "pec_cells":
            conv = _req(reg, "convention", where)
            if conv not in PEC_CELL_CONVENTIONS:
                raise CaseError(f"{where}: convention must be one of {list(PEC_CELL_CONVENTIONS)}")
            cells = self._indices(reg, where)
            for comp, shift in pec_conventions(self.dim)[conv].items():
                idx = cells + np.asarray(shift)
                _check_in_range(comp, idx, self.n, f"{where} ({conv} -> {comp})")
                self.pec[comp].append(idx)
        else:
            raise CaseError(f"{where}: unknown region kind {kind!r}")

    def _indices(self, d: dict, where: str) -> np.ndarray:
        spec = _req(d, "indices", where)
        if isinstance(spec, str):
            self.files[spec] = hashlib.sha256((self.base / spec).read_bytes()).hexdigest()
        return _load_indices(spec, self.base, where, self.dim)

    def _source(self, s: dict, where: str) -> np.ndarray:
        kind = _req(s, "kind", where)
        if kind not in ("soft_e", "hard_e"):
            raise CaseError(f"{where}: kind {kind!r} not in v0 (soft_e, hard_e; current and plane_wave reserved)")
        comp = _req(s, "component", where)
        if comp not in self.e_comps:
            raise CaseError(f"{where}: component must be one of {self.e_comps}")
        idx = self._indices(s, where)
        _check_in_range(comp, idx, self.n, where)
        _req(s, "amplitude_V_per_m", where)
        w = _req(s, "waveform", where)
        if w.get("type") != "gaussian_modulated":
            raise CaseError(f"{where}: waveform type must be gaussian_modulated in v0")
        for key in ("f0_Hz", "t0_s", "width_s"):
            _req(w, key, f"{where}.waveform")
        return idx

    def _probe(self, p: dict, where: str) -> np.ndarray:
        kind = _req(p, "kind", where)
        if kind not in ("e_samples", "h_samples"):
            raise CaseError(f"{where}: kind {kind!r} not in v0 (e_samples, h_samples; dft_plane reserved)")
        comp = _req(p, "component", where)
        allowed = self.e_comps if kind == "e_samples" else self.h_comps
        if comp not in allowed:
            raise CaseError(f"{where}: component must be one of {allowed}")
        if int(p.get("every", 1)) < 1:
            raise CaseError(f"{where}: every must be >= 1")
        idx = self._indices(p, where)
        _check_in_range(comp, idx, self.n, where)
        return idx

    # ---- derived quantities -----------------------------------------------------------------------

    @property
    def sha256(self) -> str:
        """Identity of the case: canonical JSON plus the bytes of every referenced index file."""
        canon = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256(canon.encode())
        for name in sorted(self.files):
            h.update(f"\0{name}\0{self.files[name]}".encode())
        return h.hexdigest()

    def dt(self, c: float | None = None) -> float:
        """Time step for wave speed c (the case's c by default; pass an engine's own c to get its dt)."""
        return self.courant * self.dx / (c if c is not None else self.c)

    @property
    def box_m(self) -> list[float]:
        return [n * self.dx for n in self.n]

    def is_vacuum(self) -> bool:
        """Vacuum everywhere: no material boxes and a lossless, non-dispersive eps_r = mu_r = 1 background."""
        bg = self.background
        return (not self.material_boxes and bg.get("eps_r") == 1.0 and not bg.get("sigma_S_per_m", 0.0)
                and not bg.get("lorentz") and bg.get("mu_r", 1.0) == 1.0)

    def wall_samples(self, axis: int, plane: int, lattice: int | None = None) -> dict[str, np.ndarray]:
        """Spec indices of every E component tangential to the wall `axis = plane` (plane 0 or n[axis]); with
        `lattice`, only a lattice x lattice subset per component (a cheap spot check)."""
        out = {}
        for comp in self.e_comps:
            if self.dim == 3 and comp == E_COMPONENTS[axis]:
                continue                                     # normal to this wall: free on a PEC wall
            ranges = index_range(comp, self.n)
            if lattice:
                ranges = [(0, 0) if a != axis else r for a, r in enumerate(ranges)]
            ranges[axis] = (plane, plane)
            idx = box_indices(ranges)
            if lattice:
                for a in range(self.dim):
                    if a != axis:                            # spread the points over the other axes
                        pts = np.linspace(0, self.n[a] - 1, lattice).round().astype(np.int64)
                        idx = np.repeat(idx, len(pts), axis=0); idx[:, a] = np.tile(pts, len(idx) // len(pts))
            out[comp] = idx
        return out

    def pec_count(self, convention: str | None = None) -> int:
        """Conductor size: total PEC E samples, or (with a convention) the number of distinct cells."""
        if convention is None:
            return int(sum(len(v) for v in self.pec_samples.values()))
        shift = pec_conventions(self.dim)[convention]
        cells = [self.pec_samples[c] - np.asarray(shift[c]) for c in self.e_comps if len(self.pec_samples[c])]
        return int(len(np.unique(np.concatenate(cells), axis=0))) if cells else 0

    def sources_on_pec(self) -> int:
        """Source samples that are also PEC samples. The spec says PEC wins (held at zero). The Swift programs add
        the soft source after zeroing their conductor, so there the drive survives on those samples for a step."""
        pec = {c: {tuple(r) for r in v} for c, v in self.pec_samples.items()}
        return sum(tuple(r) in pec[s["component"]]
                   for s, idx in zip(self.raw.get("sources", []), self.source_samples) for r in idx)

    def summary(self) -> dict:
        return {
            "name": self.raw["name"], "sha256": self.sha256, "grid": self.n, "dx_m": self.dx,
            "box_m": self.box_m, "dt_s": self.dt(), "steps": self.steps,
            "pec_samples": {c: len(v) for c, v in self.pec_samples.items()},
            "sources": len(self.raw.get("sources", [])), "source_samples_on_pec": self.sources_on_pec(),
            "probes": {k: {self.probe_component[k]: len(v)} for k, v in self.probe_samples.items()},
        }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for p in argv:
        try:
            print(json.dumps(Case.load(p).summary(), indent=1))
        except (CaseError, OSError, json.JSONDecodeError) as e:
            print(f"[FAIL] {p}: {e}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Source patches to LoopFDTD's dipole_s11.swift: the one definition, for the lib/fdtd backend and the examples.

guard_fix() replaces the program's single E boundary guard
    if (i == 0 || i >= Nx-1 || j == 0 || j >= Ny-1 || k == 0 || k >= Nz-1) return;
with per-component guards. As shipped, that single guard also clamps the normal E on the far half-planes, so the
box is not a Yee PEC cavity of any size (examples/yee-dispersion, the LoopFDTD finding).
Pure `re`, so it loads under any interpreter (yee-dispersion runs under meep_arm).
"""
from __future__ import annotations

import re

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
GUARD_FIX_LABEL = "guard_fix: per-component E boundary guard"    # the entry in a record's engine.patches


def guard_fix(text: str) -> tuple[str, int]:
    """Apply the guard fix to the program text; returns (text, number of replacements). The replacement is
    inserted literally (a function replacement, so no backslash or group processing)."""
    return GUARD_FIX_OLD.subn(lambda mo: GUARD_FIX_NEW, text)

# Quantum Geometric Tensor Module

Computes the quantum metric and Berry curvature from QE wavefunctions,
then constructs the complete optical conductivity σ(ω) following:

- Chen (interband): Re[σ(ω)] = quantum metric × joint DOS kernel
- Queiroz (Drude weight): D_geometric from quantum metric at Fermi surface
- Kramers-Kronig: Im[σ] from Re[σ], completing ε(ω)

Input:  QE NSCF wavefunctions (via Wannier90 or direct)
Output: Full ε(ω) from 0 to UV, JSON format compatible with MEEP converter

This replaces the Yambo BSE + empirical Drude approach with a unified
geometric calculation. Zero empirical parameters.

Key references:
- Chen: quantum metric → optical properties of semiconductors
- Queiroz & Verma, PNAS 2025: tQGT and instantaneous response
- Queiroz, Nature Reviews Physics 2026: quantum geometry perspective
- Ghosh et al., Science Advances 2024: probing QGT via optical conductivity
- Marzari & Vanderbilt 1997: Wannier spread = integrated quantum metric

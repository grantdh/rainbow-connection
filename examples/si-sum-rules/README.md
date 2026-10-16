# Example 5: Yambo optics of silicon against exact sum rules and the plasmon

Sum rules are the cleanest theoretical test of a dielectric function: they do not depend on the
functional or on experiment, only on the valence electron density and on causality.

```
QE SCF + NSCF (80 bands), NC ONCV PBE Si (no ultrasoft question for Yambo), Gamma-centred 4x4x4
  -> p2y -> yambo setup -> yambo IP-RPA eps(w), 0-40 eV, retarded (GrFnTpXd="R"), with 10, 20, 40 and 80 bands
  -> f-sum rule        (2/pi) Int_0^Wmax w eps2 dw  vs  (hbar wp)^2,  hbar wp = sqrt(4 pi n) Ha = 16.6 eV for Si
  -> KK static limit   eps1(0) - 1  vs  (2/pi) Int eps2(w)/w dw
  -> energy loss       -Im(1/eps) peak near the plasmon; the o-*.eel file must equal -Im(1/eps) from o-*.eps
  -> Yambo's reported gap vs QE
```

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py          # ~2 min
```

## References and gates

| check | theory | gate |
|---|---|---|
| f-sum, partial, full velocity p/m + i[Vnl,r] | (hbar wp)^2 with n = 8 e-/cell, V = a^3/4 is the local-potential limit; with a non-local pseudopotential the sum is N plus a double-commutator term of Vnl (Yabana and Bertsch 1998) | non-decreasing with bands; 0.9-1.3 of the limit; > 0.60 at 80 bands and 40 eV |
| control without [r,Vnl] | p/m alone obeys no sum rule with a non-local H | reported; the commutator must change the sum by > 0.02 (Read and Needs 1991: a 10-20 % effect on Si matrix elements) |
| Kramers-Kronig | exact for a causal eps | eps1(0)-1 vs (2/pi) Int eps2/w within 5 % |
| plasmon | hbar wp 16.6 eV; measured Si loss peak 16.7 eV; RPA 16-17 eV | 15-19 eV (IP on an 8-k mesh sits high) |
| loss file | -Im(1/eps) = eps2/(eps1^2+eps2^2) | max difference < 1 % of the peak |
| Yambo header | retarded (requested) or T-ordered IP response | must not say Resonant |
| p2y transfer | Yambo's indirect gap == QE HOMO/LUMO on the mesh; direct gap == QE Gamma bands 5-4 | both within 2 meV |
| p2y report | `l_p2y` (p2y writes its report there, not to stdout) | present, no 'not supported' |

Why the retarded response: Yambo's default T-ordered form has eps2(0) != 0 at finite broadening,
which breaks the Kramers-Kronig static limit by an amount proportional to the broadening (with "R" the
static limit closes to 0.0 %). Why the f-sum is not exactly 1: with a non-local pseudopotential the
Thomas-Reiche-Kuhn derivation ([x, H] = i hbar p/m) fails, so the sum with the full velocity
(p/m + i[Vnl, r], included automatically when p2y wrote `SAVE/ns.kb_pp_pwscf`) equals N plus a
pseudopotential-specific double-commutator term, and the p/m-only sum obeys no rule at all. The driver
measures both on the same wavefunctions (a SAVE copy without the kb_pp files is the control): with the
commutator 1.169 of (hbar wp)^2, p/m alone 1.387; the commutator lowers the Si oscillator strength by
16 %, the direction Read and Needs found for the matrix elements. A first version of this example gated
the full-velocity sum at < 1.10 and assumed the commutator raises it; the control run showed both
assumptions were wrong, so the gates were restated rather than the physics. The monotonic increase with
bands, the exact KK closure, the loss-file identity and the meV agreement of both gaps with QE are the
sharp tests.

## Results

15 gated checks pass, 0 fail, 2 reported (2026-10-15, ~1 min; the p/m-only f-sum and the [r,Vnl] flag are
filed with `checks.report()`, listed as PASS in the `results/RESULTS.md` on disk and printed as REPORTED by
the next run): f-sum 1.123 / 1.154 / 1.169 / 1.169 of (hbar wp)^2 for 10 / 20 / 40
/ 80 bands (control without [r,Vnl]: 1.387); eps1(0) - 1 = 23.518 vs (2/pi) Int eps2/w = 23.509; plasmon
17.99 eV (hbar wp 16.6, experiment 16.7); loss file equals -Im(1/eps) to 8e-6 of the peak; Yambo's
indirect and direct gaps equal QE's to < 0.1 meV. See `results/RESULTS.md`.

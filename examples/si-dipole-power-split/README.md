# Si dipole on SOI: where the feed power goes, and whether heating explains its photoresponse

One antenna, one question. The device is a Si meander dipole on SOI, measured on a probe station in two bands under a CW
laser. The question: how does the power at the feed divide between radiation and heat along the antenna, and can heating
explain a superlinear turn-on seen on one measurement day and not on the other?

The method is the point of this example. It starts from raw sweeps, labels every input, commits each prediction to git before
comparing it, and leaves failures red. Its device data (raw `.s1p` files, geometry, stack values, run flags, predictions and results) are
unpublished, so they stay in a private repository and are read by path (`device.json`, `prereg/inputs.json` and the write-ups' prose in `templates/` there).
This directory holds only code.

```bash
export NM_REPRO=~/Dev/Claude/nm-antenna-reproduction     # the private repository
~/.venvs/nmphys/bin/python run.py                        # every stage, skipping what is already computed
~/.venvs/nmphys/bin/python run.py --stage validate       # only the comparison
```

Without the private repository the scripts stop and say so.

## The chain

```
raw .s1p + calibration readbacks    ->  Y(f) under three explicit standard definitions, u_D           recompute_y.py
si_hr.dat + EPW (symmetric NSCF)    ->  Drude weight, mu(T), sigma(omega, T), n_i(T)                  epw_series.py, materials_si.py
stack + drawing                     ->  openEMS: P_rad (flux boxes), Re(J*.E)/2 by region, q(x)       fullwave_openems.py, fullwave_queue.py
                                        Palace: the same dark device with finite elements             fem_palace.py
                                        reciprocity and its Kirchhoff form                            reciprocity_openems.py
laser + chuck + sweep timeline      ->  closed form, then Elmer: T(r, t), kappa(T)                    thermal_closed.py, thermal_elmer.py
dark fields + sigma(T) + T(t)       ->  what heating predicts in the data                             heating.py
everything committed                ->  ASME V&V 20 table, identities, verdict                        validate.py
```

The MUPPET Studio workflow `examples/08-si-dipole-power-split.json` runs the same scripts through one node per stage.

## The deeper question: what heating would need (`run.py --stage deep`)

The first answer was a "no" conditional on two unrecorded inputs. The `deep` stage replaces the conditions with numbers. Rules are fixed in the private
`prereg/00d-00g` before each prediction, and every prediction (`07-13`) is committed before it is compared.

```
flip map: H_T and H_TC on the recorded timelines, R_th x C, mu(T) EPW or Jacoboni,        flip_map.py map | threshold | linearity
          laser laws; then along the laser threshold; then one laser for both days
the chip resting on the chuck: gas-gap contact physics -> R_th distribution                contact.py
time fingerprints, the GaAs controls, model evidence, the probe on the pads (a red row),   deep.py fingerprints | gaas | evidence | redrows | protocol
and the predictions for the measurements that would settle it (PROTOCOL.md)
the map, the comparisons, a figure                                                         validate_deep.py
```

- **Scan, don't sample, to find a flip point.** Sampling the priors only explores where they put weight. The flip map scans each unrecorded input on a grid and
  reports where the verdict changes; the priors come back only for the evidence.
- **Look for the hinge before refining.** Here the chuck contact was not the hinge: the setting-to-power law of the laser was. A heating-free test (the
  response at the lowest settings) bounds it, and the second measurement day, taken with the same laser, constrains it again.
- **Boundary-condition-free fingerprints have a reach.** A dark sweep shortly after illumination excludes slow heating whatever the mount, but heating with a
  time constant of a second or two leaves no trace in sweeps taken tens of seconds apart. The protocol asks for the measurements that can see it.

## The rules it enforces

- **Labels.** Every input in the private `prereg/inputs.json` is AI (ab initio), MS (measured on the samples, not RF), DER (derived from
  those) or ASM (assumed or from the literature, with a range). The file also lists the values that were once tuned against RF, and those are not used.
- **Predictions first.** `common.register` writes `prereg/NN_*.json`. `validate.py` takes S from those committed files only, refuses to run if any of them is uncommitted, dirty or made from
  other inputs, and prints each one's commit next to its comparison. A rerun that reproduces a committed file to 1 part in 10^6 leaves it untouched; one that differs dirties it, and validate stops.
- **RF data are never fitted.** Nothing computed from a measured S-parameter enters an input. One test (heating coupled to carriers) anchors on one measured setting so that it
  gives the hypothesis its best chance, and says so.
- **Numerics stop at the input uncertainty.** One mesh step, with every material face on a mesh line in both meshes, and no further once u_num is well below the input spread.
- **Machine etiquette.** Every solver runs under `nice`, waits while `~/Dev/.timing-lock` exists and is frozen if it reappears. `epw.x` takes the machine lock's shared mode, because a frozen
  `epw.x` would hold up a benchmark waiting for it to exit. The other solvers take no lock, start when no benchmark holds the machine and freeze while one does.

## The one-port limit

A one-port S11 fixes only the total Re(Y). No analysis of these data can split it into radiation and heat, so that split stays a model result until a Wheeler cap, a gain
measurement or a thermal measurement exists.

## What a run gives

`$NM_REPRO/workflows/nm-meander-dipole/power-split/out/RESULTS.md`: the verdict under the rule fixed in advance, the V&V 20 table, every identity with its gate, and the power split per
state. The results are kept in the private repository because they contain the measured values.

## Points worth carrying to other devices

- **Si at these densities is non-degenerate.** There is no Fermi surface to integrate on, so the Drude weight is a thermal average over the band edge. `materials_si.py` computes it from the
  Wannier Hamiltonian and from EPW's fine grid. omega tau is small at these frequencies, so the dynamic part is a check rather than a correction.
- **Sum heat on the Yee edges, not at cell centres.** J E times an edge's whole dual volume is that edge's G V^2. Cell-centred dumps average across the oxide/handle face and the pad edges and overcounted the
  heat by 5 %, most of it spuriously in the membrane. The raw edge sum closes the energy balance to under 1 %.
- **openEMS reruns stop where the machine's load lets them.** Its end criterion is tested on a wall-clock interval, so a run that stops on it stops at a
  load-dependent step: two runs of the same input stopped at 46158 and 43549 steps and differed by about 1e-4 in Y. That is far below u_num, but enough to fail a
  bit-level reproduction check. Each run that stopped on the criterion has its step stored in `out/openems_stop_steps.json` (with a hash of its arguments), and a rerun
  replays exactly that many steps, which reproduces it bit for bit. Runs that reach `--max-ts` and Palace (to 1e-12) reproduce without help.
- **Palace's domain energy gives Joule heat directly:** P = 2 (sigma/eps) E_elec for a uniform lossy domain, which closes the balance without a separate field integral. Its phasors are RMS-normalised, so its
  power flux Re{E x H*} has no factor 1/2. Missing that made the FEM radiated fraction look 1.8x the FDTD one.

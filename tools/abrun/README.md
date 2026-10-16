# abrun: CPU/GPU A/B runner for Quantum ESPRESSO on this M2 Max

`abrun` runs the same pw.x input on the apple-bottom-free reference build and on a frozen apple-bottom
build in `AB_MODE=cpu`, `auto` or `gpu`, under the house machine etiquette, and records energies, QE
wall times and the library's own dispatch counters. The point is energy/force parity and dispatch
visibility, not speed. What has been measured (apple-bottom
`benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/tables.md`, apple-bottom 1.4.0 release c5a1fa6 on the
33fbaa8c8944 freeze, 2026-10-15): `AB_MODE=gpu` at the library floor (`ab_gpu`, `AB_MIN_GPU_DIM` 32) took
1.14-1.40x the `cpu` wall on the three si64 shapes at 1 rank (A 1.295, B 1.402, C 1.141) and 1.06-1.21x at
8 ranks (np8nk4 A 1.055, B 1.214; np8 C 1.163), and `auto` routed 0 GEMMs to the GPU in every run. No wall
has been measured for `AB_MIN_GPU_DIM=0` (`ab_gpu_all`); its si64 walls in the examples are untimed
observations. Wall times are observations unless a run shows `timing_grade=bench`; abrun never computes a
speedup.

Package: `tools/abrun/` (Python 3.13, stdlib + numpy via `examples/common/exlib.py`, which it imports by
path). Launcher: `bin/abrun`. Run artefacts: `tools/abrun/runs/<run id>/` (gitignored); the example drivers
point `run_dir` into their own `results/`. Nothing is written under `~/Dev/Claude/apple-bottom`,
`~/Dev/Claude/muppet-studio` or `~/qe-test`; the only out-of-tree writes are `examples/common/solver-runs.log`
(plus machlock's own `machine.log`, written by machlock itself) and, in `--fake-pw` mode only,
`$TMPDIR/abrun-fake-<uid>/` (fake runs and their own log; never the shared one).

This README covers the core (catalog, plan, runner, parsers, compare, CLI) and the HTTP server, SSE and
browser UI (`bin/abrun serve`, section "Server, API and UI").

## Two meanings of "CPU only" (and auto as a third)

| label | what runs where |
|---|---|
| **reference** (CPU only, meaning A) | `/usr/local/bin/pw.x`: QE 7.4.1, OpenBLAS for all BLAS/LAPACK, Homebrew Open MPI, `nm` shows 0 `_ab_` symbols. |
| **ab_cpu** (CPU only, meaning B) | an apple-bottom-linked pw.x (the frozen QE develop 7.5 builds, or the `qe741-static` relink) with `AB_MODE=cpu`: dgemm/zgemm go through libapplebottom to Accelerate/AMX; everything else OpenBLAS. The static relink exports no `AB_*` counters (`strings` finds none), so its verdict is always amber "unverifiable". |
| **ab_auto** | same binary, `AB_MODE=auto`: the calibrated dispatch table. On dylibs with `ab_build_info` (33fb, 3ae1, b8cd, 72ce, 5fdc) every QE shape stays on Accelerate (0 GPU calls, by design); pre-1.4.0 builds (e56d, 7bbb) used fixed FLOP thresholds and send Davidson ZGEMMs to the GPU. |
| **ab_gpu** | `AB_MODE=gpu` at the library floor (`AB_MIN_GPU_DIM` default 32): what apple-bottom would really do. 0 GPU calls on a small cell is a legitimate result. This is the mode the 2026-10-15 si64 campaign timed (1.14-1.40x the cpu wall at 1 rank). |
| **ab_gpu_all** | `AB_MODE=gpu AB_MIN_GPU_DIM=0`: every GEMM on Metal. The parity/dispatch-evidence gate of the house examples; no bench-grade wall exists for it. |

Three pair classes (`pair` on every compare row). **same_binary**: same pw.x sha256 *and* same
libapplebottom sha256, so only `AB_MODE` differs: the one honest BLAS A/B, tolerance 1e-8 Ry. **cross_build**:
different pw.x (reference 7.4.1 OpenBLAS vs any apple-bottom build, or two apple-bottom builds with different
pw.x such as 33fb vs b8cd): tolerance 1e-5 Ry, "not a BLAS A/B"; the note names both builds from the records'
catalogued metadata (`different pw.x (e70ae3 vs 2f0f02): QE 7.4.1 (OpenBLAS, no apple-bottom) vs QE develop 7.5
-O3 (OMP=1 only) (apple-bottom 1.4.0 c5a1fa6 GEMMs + OpenBLAS): 1e-5 Ry, not a BLAS A/B`), falling back to a
generic sentence only when a record has no binary metadata. **same_pw_other_dylib**: four freezes (33fb, 3ae1,
72ce, 5fdc) share one `-O3` pw.x (2f0f029a) with different dylibs; a pair across them is labelled `same pw.x
2f0f02, different libapplebottom 33fbaa vs 5fdc1e (... vs ...): library A/B, not the one-binary mode A/B, 1e-8
Ry` and is never listed under the same-binary heading.

## Mode table: exact environment and command

Every mode: `cwd = runs/<id>/`, `input.in` is the preset with `outdir -> './tmp'` and `pseudo_dir ->
'./pseudo'` (UPFs symlinked after a sha256 pin check and `exlib.check_upf`), stdin `/dev/null`, stdout
`out.txt`, stderr `err.txt` (all apple-bottom reports are on stderr). The child environment is
`os.environ` minus every `AB_*` and `DYLD_*` key plus the mode's pairs; every exported pair is also
passed as `mpirun -x NAME=value`. Full argv = `wrapper_argv + shim_argv + solver_argv`:

```
[lock wrapper, bench/build only] /opt/homebrew/bin/python3 ~/qe-test/xp/tools/machlock.py <bench|build> --
[always]  <interpreter> tools/abrun/shim.py --run-dir <cwd> --nice 10 --
[solver]  /opt/homebrew/bin/mpirun -np N -x OMP_NUM_THREADS=<omp> -x OPENBLAS_NUM_THREADS=<omp> -x VECLIB_MAXIMUM_THREADS=<omp> <mode -x list> <abs pw.x> -in input.in [-nk K]
```

| mode | `AB_MODE` | `AB_MIN_GPU_DIM` | `AB_REPORT` | `AB_PROFILE` + `AB_PROFILE_JSON` | `DYLD_PRINT_LIBRARIES` | `OMP_NUM_THREADS` |
|---|---|---|---|---|---|---|
| reference | not set (binary must be apple-bottom-free) | not set | not set | not set | `1` when `verify_dylib` (parse must find no libapplebottom line) | `<omp>` (default 1) |
| ab_cpu | `cpu` | only on explicit override | `1` if the dylib has it | `1` + `<cwd>/ab_profile.json` when `ranks == 1` | `1` when `verify_dylib` (default) | `1` (forced; `allow_omp_gt1` only on non-`omp1_only` builds) |
| ab_auto | `auto` | only on explicit override | `1` if the dylib has it | same | same | `1` |
| ab_gpu | `gpu` | only when set (`> 0`); otherwise the library default 32 | `1` if the dylib has it | same | same | `1` |
| ab_gpu_all | `gpu` | `0` (fixed) | `1` if the dylib has it | same | same | `1` |

`AB_TRACE=<cwd>/ab_trace` is exported only when `trace=true` and the dylib lists `AB_TRACE` (3ae1, 72ce,
5fdc). `AB_MODE` is only ever the literal `cpu`, `auto` or `gpu`; `AB_REPORT`/`AB_PROFILE` are only ever `1`;
`AB_TEST_FAIL_STAGE` can never be set. At `ranks > 1`, `AB_PROFILE_JSON` is auto-off (no pid suffix: the last
rank wins) and the per-pid `AB_REPORT` lines in `err.txt` are summed instead (`force_profile_json=true`
exports it anyway with a warning). Example `ab_gpu_all` solver argv:

```
/opt/homebrew/bin/mpirun -np 1 -x OMP_NUM_THREADS=1 -x OPENBLAS_NUM_THREADS=1 -x VECLIB_MAXIMUM_THREADS=1 \
  -x AB_MODE=gpu -x AB_MIN_GPU_DIM=0 -x AB_REPORT=1 -x AB_PROFILE=1 -x AB_PROFILE_JSON=<cwd>/ab_profile.json \
  -x DYLD_PRINT_LIBRARIES=1 /Users/grantheileman/qe-test/ab-frozen/33fbaa8c8944/pw.x -in input.in
```

### GPU verdict (from counters only)

`gpu_verified` never looks at the requested mode. Evidence is the profile JSON (1 rank; v1 and v2 schemas
normalised), else the summed `AB_REPORT` lines, else none (`gpu_verified = null`, amber "unverifiable").
reference: n/a. Every ab mode first: `calls == 0` (counters exist but no dgemm/zgemm reached libapplebottom:
the library's own "No BLAS-ABI calls recorded." or a profile JSON with an empty `routines` list) is `null`/amber
"no GEMM call reached apple-bottom (calls == 0)" for ab_cpu, ab_auto and ab_gpu and `false`/red for
ab_gpu_all; nothing passes on zero calls. Then: ab_cpu: true (grey) iff `gpu_calls == 0` and `fallbacks == 0`,
else false/red. ab_auto: any fallback is false/red; `gpu_calls == 0` is true/grey; `gpu_calls > 0` is false/amber
on a dylib with `ab_build_info` (calibrated table) and `null`/amber (informational) on pre-1.4.0 dylibs. ab_gpu:
false/red iff `gpu_failed + other > 0` (or any fallback without a breakdown); 0 GPU calls reads "no GEMM above
the floor" (true/grey); otherwise true, amber when `too_large > 0`, else green. ab_gpu_all: false/red when
`no_device`, `gpu_failed` or `other > 0` (`gpu_failed > 0` is a bug to report) or when GEMMs stayed on the CPU
for any other reason; true/green iff `gpu_calls == calls` and `fallbacks == 0`; true/amber when the only
fallbacks are `too_large` (`gpu_calls + too_large == calls`, operand too large for Metal). `build_matches_dylib`
compares the `AB_REPORT`/JSON build string with the dylib's `strings`-derived `ab_build_info`; `lib_ok` compares
the dyld-reported libapplebottom path with the catalogued dylib.

## Binaries

`bin/abrun binaries [--all] [--refresh]` catalogs, read-only (`shasum`, `otool`, `nm`, `strings`):
`/usr/local/bin/pw.x` (id `ref`), every `~/qe-test/ab-frozen/<hash>/pw.x` (id = hash; `SHA256SUMS`
verified at catalog time and again at every launch; `PROVENANCE.txt` shown raw), the static relink
`~/qe-test/q-e-qe-7.4.1/bin/pw.x` (`qe741-static`, never citable), the `~/qe-test/builds/*/bin/pw.x` trees
(`builds:<name>`, hidden unless `--all`, need `allow_dev`, never citable or timed; `mpi-gpu-ab` loads the
LIVE `apple-bottom/build` dylib), and informationally the ph.x and Yambo binaries with their real linkage
(only `yambo-build/builds/dp-mpi/yambo` is apple-bottom-free; `$SF_YAMBO_BIN/yambo` links the live build).
`pw.x-xp` and `pw.x-async` are private research builds: never in `binaries[]` (not even with `--all` /
`?all=1`), their paths in `hidden_private` and their records in `private[]` so `binaries.get()` can refuse
them by name; labelled from their own strings only (the directory's `PROVENANCE.txt` describes the sibling
pw.x and is not read for them); never launchable.

Advisory class (override table for the seven known freezes, keyword heuristics for unknown ones):
33fbaa8c8944 citable (1.4.0 release c5a1fa6, clean, -O3 pw.x, OMP=1 only); b8cd26bdaee1 citable (1.4.0
candidate, -O0 pw.x); 3ae144a9cadd experiments (adds AB_TRACE; parity and trace runs only); 7bbb86408ee5
pre-1.4.0; e56d1b46c753 parity-only; 72cef8fb565a and 5fdc1e839797 experimental. Only class `citable` and
the reference can be timed. abrun never uses exlib's default `e56d1b46c753`; the default is 33fbaa8c8944.

## Presets

`si2` (2-atom FCC Si, `examples/si-thin-film` SI_SCF template, ~5 s; no expected energy: the first reference
run becomes the pinned reference), `C` (Si64 160 bands Gamma, expected -2989.94060060 Ry), `B` (Si64 160
bands 2x2x2, -2990.44093535 Ry), `A` (Si64 500 bands 2x2x2, -2990.44276149 Ry; expected values from
apple-bottom `benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/tables.md`, 1 rank, conv_thr 1e-6), and
`custom` (any absolute `.in`). The si64 inputs are read from `~/qe-test/xp/inputs` (`MUPPET_SI64_INPUTS`
overrides); their sha256 and the UPF sha256 are pinned in `presets.py`; a leftover `OUTDIR`/`PSEUDO_DIR`
placeholder or `/private/tmp` path is an error. Copying the two PAW UPFs into `shared/pseudo/` is an owner
decision (the search list already looks there).

## Etiquette

1. `~/Dev/.timing-lock` is polled every 5 s before every launch in every lock mode; nothing skips it (tests
   monkeypatch `abrun.runner.TIMING_LOCK` in-process only). Because the machlock wait that follows can last
   minutes, the shim re-checks the lock **inside** the machine lock: if it has appeared, the shim exits 75
   without writing `machine.json` or starting the solver, machlock releases, and the runner waits for the
   lock again and relaunches (`etiquette.timing_lock_relaunches`). It never polls inside the lock: the
   timing-lock holder may itself be waiting for machlock. In guard mode `MUPPET_GUARD_MAX_WAIT_S` discharges
   foreign solvers only, never the timing lock, and `MUPPET_GUARD_OVERRIDE` skips the foreign-solver wait, never
   the lock. A `machine.json` that still shows `timing_lock_present` makes the grade `untimed`.
2. Default lock `share` for untimed runs (MUPPET_SHARE semantics, PROTOCOL section 9): no machlock, the
   foreign-solver list and load1 recorded at launch. `auto` (the default request) resolves to **bench** when
   `timed`, to **build** when another process holds machlock bench at dequeue time (so the run waits instead
   of becoming a foreign pw.x inside someone's timing window), else share. An **explicit `share` while a bench
   is held also waits as build** (`etiquette.lock_share_promoted_to_build`, a `[guard]` line says so): machlock
   only checks for foreign pw.x before a bench starts, so a share run launched later would be invisible to the
   bench holder. `allow_share_during_bench=true` (`--allow-share-during-bench`, or the field in a `POST
   /api/runs` / `ab_set` / `rerun` body) launches anyway and says so in the guard line. `build`/`bench` wrap
   the command in `/opt/homebrew/bin/python3 ~/qe-test/xp/tools/machlock.py <kind> --`. `MUPPET_NO_MACHLOCK=1`
   turns build/bench into the guard path (`exlib.wait_for_idle` semantics, 5 s cancellable polls).
3. Niceness: `shim.py` calls `os.nice(10)` inside the lock, single-threaded, then `execv`s mpirun;
   `/usr/bin/nice` is never used (SIP strips `DYLD_*`). `caffeinate -ims -w <pid>` is a sibling of timed runs.
   A missing mpirun is one shim line and exit 127, not a traceback.
4. `machine.json` is written by the shim at lock acquisition (load1, power, timing-lock state, foreign
   solvers, pgid); the runner samples load1 and foreign solvers every 20 s (`etiquette.contention[]`) with
   the run's **own process tree excluded**: the launched pid plus its descendants from a ppid walk, refreshed
   each sample (PRRTE puts every rank in its own process group, so a pgid filter would count the run's own
   ranks as foreign and grade every mpirun run longer than 20 s "contended"). The tree is persisted in
   `run.json` as `process {pid, pgid, descendants[], sampled, wrapper, exited}`. `gpu_idle` is always `unverified`.
5. After every run **that started a process** one line in exlib's format is appended to
   `examples/common/solver-runs.log`:
   `<time> rc=<n> wall=<s>s load1=<f> idle_at_start=<bool> lock=machlock|share|guard label=abrun <preset> <mode> <binid> cwd=<run dir>`.
   `wall` (= `wall_runner_s`) is counted from lock acquisition (the shim's `machine.json` timestamp; from the
   launch when there is no wrapper) to exit, like exlib's subprocess wall: queue, timing-lock, guard and
   machlock waits are not in it (they are in `wall_total_s` and `etiquette`). A run refused or cancelled
   before any process started writes no line. `--fake-pw` runs never write the shared log (see the server section).
6. Frozen binaries by absolute path from a scratch cwd (`LC_RPATH @executable_path`); a cwd under
   `~/qe-test/ab-frozen` is refused, and so is the runs root itself or any of its ancestors.
7. Cancel is descendant-aware. The runner snapshots the descendants of the launched pid (ppid walk) and
   SIGINTs the **solver tree directly**: every descendant when machlock is the root (the wrapper itself is
   never signalled: its `subprocess.call` turns a KeyboardInterrupt into `p.kill()` on prterun before mpirun
   can abort the ranks, which sit in their own process groups and would survive), the root's group plus the
   descendants on the direct path, the group alone while machlock is still waiting. prterun aborts its ranks,
   machlock logs its release and exits. SIGTERM follows at 10 s and SIGKILL at 20 s over every descendant seen
   at any point, then a sweep of the snapshot (only pids that are still ours: a child of the tree, an orphan
   re-parented to launchd, or a solver by name). The same sweep runs after every exit, so a launcher that dies
   before its ranks never leaves them behind. A cancel that lands between the last check and `Popen` is picked
   up by the poll loop. The run directory is kept.
8. Timing grade: `bench` only when `timed=true` took machlock bench and `machine.json` shows load1 < 1.5, AC
   power, no timing lock, no foreign solvers, and no contention event occurred, on a citable binary;
   everything else is `untimed` with the reason. Wall ratios are only comparable between bench-grade runs of
   the same pw.x: `wall_ratio_vs_ref` is `null` otherwise (the raw quotient, if both walls exist, is
   `wall_quotient_untimed`, never a ratio).
9. `tmp/` is purged after parsing (si64 500-band wavefunctions are GB-scale) unless `purge_tmp=false`.

## CLI

```
bin/abrun binaries [--all] [--refresh] [--json]
bin/abrun plan --preset si2 --binary 33fbaa8c8944 --mode ab_gpu_all [--ranks N --nk K --omp 1 --lock auto|share|build|bench --timed ...] [--json]
bin/abrun run  --preset si2 --binary 33fbaa8c8944 --mode ab_gpu_all [--run-dir DIR] [--label ...]
bin/abrun compare <id|run dir>... [--ref <id>] [--md|--json|--csv]
bin/abrun serve [--port 8766] [--open] [--fake-pw]
```

`run` is the headless twin of `POST /api/runs`: it runs in-process through `runner.run_sync` and prints the
run id and summary. Exit status: 0 done, 1 failed, 2 invalid plan.

## Server, API and UI

`bin/abrun serve` starts one process on `http://127.0.0.1:8766` (`--port`, or `ABRUN_PORT`; muppet-studio
keeps 8765): `http.server.ThreadingHTTPServer` bound to the loopback interface, **no auth**, one worker
thread that drains the persisted queue (`runs/queue.json`), a machine-status thread every 5 s, and one
Server-Sent-Events writer per open EventSource. `Ctrl-C`, `SIGTERM` or `SIGHUP` (a closed terminal) cancels
the running job (descendant-aware, see Etiquette 7; waits for it to reach a terminal state), persists the
queue and exits. `--open` opens the UI in the default browser.

**Safety.** Binding to loopback is not origin isolation: any web page the user has open can send a
preflight-free POST to 127.0.0.1, and a DNS-rebound hostname could read. So every request must carry
`Host: 127.0.0.1:<port>` or `localhost:<port>` (403 otherwise), `POST`/`DELETE` must be
`Content-Type: application/json` (415) with `Origin` absent (curl, http.client) or same-origin (403), `OPTIONS`
is 403 with no CORS headers, bodies above 1 MiB are 413, idle sockets time out after 60 s. `DELETE
/api/runs/{id}` never names the runs root or its parent (`.`/`..` have no route; `delete_run` requires a direct
subdirectory), and `new_run(run_dir=...)` refuses the runs root and its ancestors. Custom inputs: the
`&CONTROL` group is parsed like QE does (one-line, comma- and blank-separated namelists, comments, quotes) and
rewritten one entry per line with `outdir='./tmp'`, `pseudo_dir='./pseudo'`, `wfcdir` removed and `prefix`
refused when it contains `/` or `..`; an unparseable group is an error, never a pass-through. An
`ATOMIC_SPECIES` name with a path separator or `.`/`..` is refused before anything is linked, and `prepare()`
only ever unlinks a direct child of the run's `pseudo/`.

**Restart.** On start-up queued ids are re-enqueued; a job that was running at the previous exit is marked
`failed` (`server restarted`) when its process tree is gone, but when `run.json`'s `process` record still
names live pids (the solver runs in its own session and survives a `kill -9` of the server) the worker
**holds**: `GET /api/queue` shows `state: held`, `paused: true` and a `paused_reason`, nothing is dequeued
until the orphan has exited (polled every 5 s; `POST /api/runs/{id}/cancel` on it stops the tree), and the
record then finishes as `failed` (or `cancelled`). `POST /api/queue/resume` cannot lift a hold.

`--fake-pw` registers `tests/fake_pw.py` as binaries `fake` (apple-bottom, dynamic, every AB_* feature) and
`reference-fake`, points `launch.MACHLOCK` at `/nonexistent` so fake runs never take the real machlock, and
redirects the runs directory and the shared `solver-runs.log` to `$TMPDIR/abrun-fake-<uid>/` so fake runs
never land beside real ones (unless `ABRUN_RUNS_DIR` already moved them); the real `~/Dev/.timing-lock` is
still honoured (nothing bypasses it). Environment: `ABRUN_PORT`, `ABRUN_RUNS_DIR` (run directories; default
`tools/abrun/runs`), `ABRUN_HTTP_LOG=1` (log every request), `MUPPET_NO_MACHLOCK` / `MUPPET_GUARD_OVERRIDE` /
`MUPPET_GUARD_MAX_WAIT_S` / `MUPPET_SI64_INPUTS` as in the core.

Every `/api/*` response is JSON with `Cache-Control: no-store`; errors are
`{"error": "<message>", "violations": ["..."]}` with 400 (invalid), 403 (Host/Origin), 404 (unknown
id/route), 405 (method), 409 (state conflict: duplicate plan, cancel/delete on the wrong state), 413 (body),
415 (Content-Type).

| route | what |
|---|---|
| `GET /`, `GET /static/{file}` | the single-page UI (`static/index.html`, `app.js`, `style.css`) |
| `GET /api/health` | `{ok, version, interpreter, exlib, runs_dir, port, started, queue{running,queued,paused}, fake_pw, reference_binary_id, timing_lock, run_log}` |
| `GET /api/machine` | `{updated, load1, timing_lock{path,present,text}, machlock{path,available,bench_held,log_tail[3]}, foreign_solvers[{pid,cmd}], power AC/Battery/unknown, disk_free_gb, disk_warn, timed_possible, timed_blockers[], worker{state,current,queued,paused}}` |
| `GET /api/binaries[?all=1][&refresh=1]` | `{generated, binaries[Binary], other_solvers[Binary], hidden_private[], hidden_dev[], reference_binary_id, default_binary_id}`; dev builds only with `all=1` |
| `GET /api/presets`, `GET /api/presets/{id}/input[?input_path=]` | the preset cards; the rewritten `input.in` as text/plain |
| `GET /api/modes` | the five modes with label, one-sentence `short`, `requires`, `env`, `removes` |
| `POST /api/plan` | body = Request; 200 `{plan, violations: [], run_id_would_be}` or 400 `{error: "invalid plan", violations, plan}`; nothing on disk |
| `POST /api/runs` | body = Request (+ `allow_share_during_bench`); 201 `{id, state: "queued", position, plan, run_dir}`; 400 on violations; 409 if an identical plan (same preset/input/binary/mode/ranks/nk/omp/min_gpu_dim/flags/lock/timed) is queued or running |
| `POST /api/ab_set` | `{preset, binary_id, ranks, nk, omp, lock, timed, legs: ["reference","ab_cpu","ab_auto","ab_gpu","ab_gpu_all"], label, ...}` -> 201 `{ids, ref, pinned}`: every leg validated first, then queued in order; the reference leg uses `reference_binary_id` and is pinned for that input sha256; `timed` forces lock bench on all legs |
| `GET /api/runs[?state=&preset=&mode=&binary=&limit=200]` | `{runs: [RunSummary + position + run_dir], total}`, newest first; `pinned_reference` comes from `runs/pins.json` |
| `GET /api/runs/{id}` | the full `run.json` (request, plan, binary snapshot, etiquette, result, ab, files) |
| `GET /api/runs/{id}/log?since=0&stream=both\|out\|err\|guard&max=2000` | `{next, state, state_why, state_since, lines[[seq,stream,text]], progress{iteration,max,scf_accuracy_Ry}, suppressed_profile_lines, event_seq, live}`; for a run no longer in memory the lines are rebuilt from `out.txt`/`err.txt` with the same dyld/AB_PROFILE filtering |
| `GET /api/runs/{id}/events[?since=N]`, `GET /api/events[?since=N]` | `text/event-stream`; frames `id: <seq>` + `data: {"type","data"}`; `Last-Event-ID` or `?since=` replays what was missed; `: ping` every 10 s; the per-run stream closes after `muppet.done` (a disk-only run gets one synthetic `muppet.done`) |
| `GET /api/runs/{id}/files/{name}` | raw file; whitelist `input.in out.txt err.txt ab_profile.json machine.json summary.json run.json ab_trace.<pid>` |
| `POST /api/runs/{id}/cancel` | `{state: "cancelled"}` for a queued run (dequeued), `{state: "cancelling"}` for the running one (also for a run waiting on a lock, or an orphan held after a restart); 409 when terminal |
| `POST /api/runs/{id}/rerun` | body `{mode, binary_id?, lock?, timed?, label?, ...}` -> 201 like `POST /api/runs` with `derived_from` set (same preset/input path, re-prepared) |
| `POST` / `DELETE /api/runs/{id}/pin` | `{input_sha256, reference}` / `{unpinned}` (`runs/pins.json`) |
| `DELETE /api/runs/{id}` | `{deleted: true}`; 409 unless terminal |
| `GET /api/queue`, `POST /api/queue/pause\|resume`, `POST /api/queue/reorder {ids}` | `{running, queued[], paused, paused_reason, held, state: busy\|held\|idle}` / `{paused}` / `{queued}` |
| `GET /api/compare?ids=a,b,c[&ref=a][&energy_tol=1e-8]` | `{ref, rows[], groups[], sections{same_binary, cross_build}, markdown}` (compare.py): rows ordered ref, same_binary, same_pw_other_dylib, cross_build; each row carries `pair`, `same_pw_sha`, `same_dylib_sha`, `same_binary`, `tolerance_note` (built from both records' binary metadata), `binary_desc`, `wall_ratio_vs_ref` (`null` unless comparable) and `wall_quotient_untimed`; `sections.same_binary` holds only the one-binary A/B rows, `sections.cross_build` every other row |
| `GET /api/export?ids=..&ref=..&format=md\|csv\|json` | the compare table as text/markdown, text/csv or JSON |

Event types and payloads (the studio's names plus `abrun.state`): `status {queue_remaining}`,
`execution_start {run_id}`, `executing {run_id, node: "pw"}`, `abrun.state {run_id, state, why, since}`,
`muppet.log {run_id, seq, stream: out|err|guard, text}`, `progress {run_id, value, max, scf_accuracy_Ry}`,
`executed {run_id, summary}`, `execution_error {run_id, message, tail[<=25]}`, `execution_interrupted {run_id}`,
`muppet.done {run_id, status: success|error|interrupted, seconds}`. Per-call `AB_PROFILE` lines are streamed
for the first 50, then counted (`+N AB_PROFILE per-call lines suppressed, see err.txt` marker); dyld lines
that do not name libapplebottom are dropped from the stream (all of it is in `err.txt`).

### curl cookbook

```bash
bin/abrun serve --port 8766 &                              # add --fake-pw to exercise everything without a solver
curl -s localhost:8766/api/health
curl -s localhost:8766/api/machine
curl -s localhost:8766/api/binaries | python3 -m json.tool | grep -E '"(id|label|class|ab_build_info)"'
curl -s -X POST localhost:8766/api/plan -H 'content-type: application/json' \
     -d '{"preset":"si2","binary_id":"33fbaa8c8944","mode":"ab_gpu_all","ranks":1}'
curl -s -X POST localhost:8766/api/runs -H 'content-type: application/json' \
     -d '{"preset":"si2","binary_id":"33fbaa8c8944","mode":"ab_cpu","lock":"auto"}'
curl -s -X POST localhost:8766/api/ab_set -H 'content-type: application/json' \
     -d '{"preset":"si2","binary_id":"33fbaa8c8944","ranks":1,"legs":["reference","ab_cpu","ab_gpu_all"]}'
curl -sN localhost:8766/api/events                         # watch until the muppet.done frames
curl -sN 'localhost:8766/api/runs/<id>/events?since=0'     # one run, replay from the start, closes at muppet.done
curl -s 'localhost:8766/api/runs/<id>/log?since=0&stream=err' | head
curl -s localhost:8766/api/runs/<gpu id> | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["result"]["total_energy_Ry"], r["ab"]["totals"], r["ab"]["gpu_verified"], r["ab"]["lib_loaded"])'
curl -s 'localhost:8766/api/compare?ids=<a>,<b>,<c>&ref=<a>' | python3 -c 'import json,sys; print(json.load(sys.stdin)["markdown"])'
curl -s 'localhost:8766/api/export?ids=<a>,<b>&ref=<a>&format=csv'
J='-H content-type:application/json'                       # every POST/DELETE needs it (415 otherwise; see Safety)
curl -s -X POST $J localhost:8766/api/runs/<id>/cancel      # queued or running
curl -s -X POST $J localhost:8766/api/runs/<id>/rerun -d '{"mode":"ab_auto"}'
curl -s -X POST $J localhost:8766/api/runs/<id>/pin; curl -s -X DELETE $J localhost:8766/api/runs/<id>/pin
curl -s -X POST $J localhost:8766/api/queue/pause; curl -s localhost:8766/api/queue; curl -s -X POST $J localhost:8766/api/queue/resume
curl -s -X DELETE $J localhost:8766/api/runs/<id>           # terminal runs only
kill -INT %1                                               # graceful (so are TERM and HUP): cancels the running job, persists the queue
```

### The page

One page, three columns (Launch 340 px | Runs + Log | Compare 420 px) that stack under 1100 px; dark mode
follows `prefers-color-scheme`; no external assets; the footer sentence is always visible: *Wall times are
observations unless a run shows grade=bench. Measured with apple-bottom 1.4.0 on this M2 Max (campaign
2026-10-15, si64 shapes A/B/C): gpu mode at the library floor took 1.14-1.40x the cpu wall at 1 rank and
1.06-1.21x at 8 ranks, and auto routed 0 GEMMs to the GPU (apple-bottom
benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/tables.md). No wall was measured for AB_MIN_GPU_DIM=0; the
point of this tool is energy/force parity and dispatch visibility.* (The per-shape ratios are in the footer's
tooltip.)

- **Status bar** (polls `/api/machine` every 5 s): load1 chip (green < 1.5, amber < 5, red), a red
  `TIMING LOCK PRESENT: every run will wait` chip with the lock's text, the machlock chip (`bench held by
  another session`), foreign-solver count with the list on hover, power, free disk (warning under 20 GB),
  worker state with Pause/Resume, `timed runs possible: yes/no` with the blockers on hover, a FAKE chip in
  fake mode, and the EventSource state. When a bench is held or the timing lock exists the Run button stays
  enabled but reads `Run (will wait: ...)`.
- **Launch**: preset select (+ path for `custom`) with the preset card (nat, nbnd, k-points, pseudos with
  the sha tick, expected E with its source, campaign walls tagged *observation*) and `Show rewritten input`;
  binary select in optgroups (Reference / Frozen: citable / Frozen: experiments... / Static relink / Dev
  builds, hidden unless `Show dev builds`; private builds never appear) with the binary card (path, QE build, links, `nm _ab_` counts,
  resolved dylib + build string, feature chips, pw.x and dylib sha with the SHA256SUMS tick, class badge
  and reason, the first 6 PROVENANCE lines with expand, OMP=1-only / LIVE warnings); the five mode radios
  (incompatible ones disabled with the violation as tooltip); Advanced (ranks, -nk, OMP locked to 1 for
  apple-bottom binaries unless `allow OMP>1` on a non-omp1_only build, AB_MIN_GPU_DIM for ab_gpu only,
  AB_PROFILE+JSON with the ranks>1 note, AB_REPORT, AB_TRACE only when the dylib lists it, verify dylib,
  lock radio, timed (disabled with the blockers listed), purge tmp, label); the plan preview (POST
  `/api/plan` on every change: cwd, wrapper/shim/solver argv one token per line, env set/removed, lock
  resolution, notes, violations in red, `Copy shell line`); Run (disabled while violations exist), `Queue
  A/B set` with the leg checkboxes (default reference + ab_cpu + ab_gpu_all), Clear.
- **Runs + Log**: the runs table (newest first; preset/mode/binary/state filters): state chip, preset,
  binary short, mode badge, E_tot (8 decimals), dE vs the preset's expected energy, iters, QE wall with the
  grade tag (`untimed` struck through, `bench` solid), GPU/GEMM calls (green when `gpu_verified`, grey
  `0/N` for cpu/auto, amber `unverifiable`, red on hard fallbacks), lock, label (with the pin star), and the
  actions Cancel / Rerun on (other modes, same binary) / Pin / Delete / Copy dir. Campaign pseudo-rows show
  the recorded si64 energies and walls with their source. Below it the live log: EventSource with
  `Last-Event-ID` replay and a polling fallback (`/log?since=`), capped at 2000 lines, stream filter
  (all / out / err / AB_* only / guard), autoscroll, the progress strip (`iteration #N / electron_maxstep,
  estimated scf accuracy`), the guard state with elapsed time (`waiting_timing_lock since ...`), the
  `+N AB_PROFILE lines suppressed` marker, a result strip when done (E_tot, dE vs expected, iters, QE wall
  and grade, runner wall, lib_loaded tick/cross, build string, the gpu_verified sentence), Cancel and the
  out.txt / err.txt / ab_profile.json / summary.json downloads.
- **Compare**: a checkbox per terminal run and a reference radio (default: the pinned run, else the newest
  reference-mode run); the table from `/api/compare` with the one-binary rows under *BLAS A/B: same pw.x and
  same libapplebottom sha256* and every other row under *Not the one-binary A/B* (cross-build, or the same
  pw.x with another libapplebottom; the tol tooltip names both builds): E, dE vs ref, tolerance with
  pass/fail colour and the note as tooltip, iters, QE wall + grade, wall ratio (a number only when both runs
  are bench-grade on one pw.x; `-` otherwise, the reason being `wall_note` in the JSON), GEMM/GPU calls, GPU %,
  fallbacks with the breakdown on hover, lib build, and max |dF| / max |dsigma| when both runs carry
  forces/stress; `Copy markdown` (the exact text the examples embed in RESULTS.md: the table plus a legend
  naming each binary's catalogued build and the mode/tolerance key) and `Download compare.json`; three inline SVG plots: SCF
  convergence log10|E_i - E_final| per run, QE wall bars hatched when untimed, and the GEMM dimension
  histogram (the profiler's bins 1-32 ... 8K+) grouped per run.

Browser checklist (`bin/abrun serve --fake-pw --open`): status bar updates; pick `fake` + ab_gpu_all; the
preview changes on every edit; Run; the log streams and the progress strip moves; Cancel a sleeping fake
(`ABRUN_FAKE_SLEEP=60` in the server's environment); Queue A/B set; pin the reference; the compare table lists
the same-binary rows first with struck-through untimed ratios; Copy markdown; reload the page (runs persist);
dark mode renders.

## Python API (what the examples and the server build on)

```python
sys.path.insert(0, "<repo>/tools"); from abrun import runner, launch, compare, binaries, presets, parse
rec = runner.run_sync(launch.Request(preset="si2", binary_id="33fbaa8c8944", mode="ab_gpu_all"), run_dir=results/"ab_gpu")
# rec == run.json: id, state, request, plan, binary, etiquette, result, ab, files ...
table = compare.compare([rec_ref, rec_cpu, rec_gpu], ref_id=rec_ref["id"])["markdown"]
```

- `binaries.inventory(refresh=False, include_private=False)` (`{binaries, other_solvers, private, hidden_private, hidden_dev}`), `binaries.get(id)`, `binaries.verify_at_launch(b)`
- `presets.prepare(preset, cwd, input_path=None, write=True)`, `presets.list_presets()`, `presets.rewrite_input(text)`
- `launch.Request`, `launch.Plan`, `launch.plan(req, binary, cwd, lock_effective=None, environ=None, bench_held=False)`,
  `launch.validate(req, binary, cwd)`, `launch.resolve_lock(lock, timed, bench_held)`, `launch.child_env(env_set)`, `launch.modes_list()`
- `runner.new_run(req, run_dir=None, allow_share_during_bench=None)`, `runner.execute(run, log=None)`, `runner.cancel(run)`,
  `runner.run_sync(req, run_dir=None, log=print, allow_share_during_bench=None)`, `runner.plan_only(req)`,
  `runner.Worker(...)` (`submit/pause/resume/reorder/dequeue_cancel/status/persist/restore`; `hold` after a restart),
  `runner.REGISTRY`, `runner.load_records()`, `runner.load_record(id)`, `runner.Pins()`, `runner.delete_run(id)`, `runner.global_bus`,
  `runner.ps_table()`, `runner.descendants(pid)`, `runner.foreign_solvers(own_pids)`, `runner.live_pids(process, run_dir)`
- `parse.parse_out(text)`, `parse.parse_err(text)`, `parse.parse_profile_json(path)`, `parse.parse_trace_headers(dir)`,
  `parse.verify_gpu(mode, evidence_source, totals, has_build_info)`, `parse.ab_record(...)`
- `compare.compare(records, ref_id=None, energy_tol=None, pins=None)`, `compare.export(result, fmt)`, `compare.pair_kind(rec, ref)`,
  `compare.tolerance_note(rec, ref, kind)`, `compare.binary_short(rec)`, `compare.build_desc(rec)`
- `events.RingBuffer`, `events.EventBus`, `events.sse_stream(bus, since, stop, heartbeat_s, close_on_terminal)`

## Tests

```
~/miniconda3_arm64/envs/meep_arm/bin/python -m unittest discover -s tools/abrun/tests -t tools
```

No solver, no real lock, no real mpirun or machlock: `test_parse` pins the fixtures, `test_plan` is the
golden argv/env and violation matrix, `test_presets` covers the `&CONTROL` rewrite and the pseudopotential
name gate, `test_binaries` makes read-only assertions against the real inventory (skipped where a path is
absent), `test_runner` drives the runner end to end on `tests/fake_pw.py` with the timing lock, run log and
runs directory redirected to a temp directory, and (`TreeCase`) on the machlock + mpirun process tree with
`tests/fake_machlock.py` (runs the job with `subprocess.call` like the real wrapper) and
`tests/fake_mpirun.py` (ranks in their own process groups, abort ~0.5 s after SIGINT): cancel without
orphaning ranks that ignore SIGINT, the group-SIGINT control that does orphan them (and the exit sweep that
catches them), SIGKILL escalation against a deaf launcher, the direct-mpirun path, own ranks never foreign,
a cancel landing before `Popen`, the shim's exit 75 and relaunch, guard max-wait vs the timing lock, share
during a held bench, `persist()` under four threads, `restore()` holding on a live orphan, and the runs-root
refusals. `test_events` covers the ring buffer, bus and SSE framing, and `test_api` starts
`server.Server(port=0, fake_pw=True)` in-process and drives every route with `http.client`:
health/static/catalog, plan, an ab_gpu_all run read through its SSE stream (both profile schemas),
log/files, compare/export, rerun/pin/delete, cancel of a sleeping fake (no survivor in the pgid), the
timing-lock wait on a temp lock, the failure tail, duplicates and error codes, pause/A-B set/reorder, the
Host/Origin/Content-Type/OPTIONS/body-cap refusals, `DELETE /api/runs/.`, global replay with
`Last-Event-ID`, the persisted-queue restart (`server restarted`), and the `--fake-pw` path redirect.

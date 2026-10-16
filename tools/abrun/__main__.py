"""abrun command line: binaries | plan | run | compare | serve.

    bin/abrun binaries [--all] [--refresh] [--json]
    bin/abrun plan --preset si2 --binary 33fbaa8c8944 --mode ab_gpu_all [run flags...] [--json]
    bin/abrun run  --preset C --binary 33fbaa8c8944 --mode ab_gpu_all [--ranks N --nk K --omp 1
                   --lock auto|share|build|bench --timed --label ... --run-dir DIR --no-profile-json ...]
    bin/abrun compare <id|run dir>... [--ref <id>] [--md|--json|--csv]
    bin/abrun serve [--port 8766] [--open] [--fake-pw]   (HTTP API + SSE + browser UI on 127.0.0.1; ABRUN_PORT)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import DEFAULT_FROZEN, INTERPRETER, RUNS_DIR, VERSION
from . import binaries as binaries_mod
from . import compare as compare_mod
from . import launch, runner


def _add_request_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="si2", help="si2 | C | B | A | custom")
    p.add_argument("--input", dest="input_path", default=None, help="absolute .in path for --preset custom")
    p.add_argument("--binary", dest="binary_id", default=DEFAULT_FROZEN, help="binary id (ref, <frozen hash>, qe741-static, builds:<name>)")
    p.add_argument("--mode", default="ab_gpu_all", choices=list(launch.MODES))
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--nk", type=int, default=1)
    p.add_argument("--omp", type=int, default=1)
    p.add_argument("--min-gpu-dim", dest="min_gpu_dim", type=int, default=None)
    p.add_argument("--no-profile-json", dest="profile_json", action="store_false")
    p.add_argument("--no-report", dest="report", action="store_false")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--no-verify-dylib", dest="verify_dylib", action="store_false")
    p.add_argument("--lock", default="auto", choices=list(launch.LOCKS))
    p.add_argument("--timed", action="store_true")
    p.add_argument("--allow-omp-gt1", dest="allow_omp_gt1", action="store_true")
    p.add_argument("--allow-dev", dest="allow_dev", action="store_true")
    p.add_argument("--force-profile-json", dest="force_profile_json", action="store_true")
    p.add_argument("--keep-tmp", dest="purge_tmp", action="store_false")
    p.add_argument("--allow-share-during-bench", dest="allow_share_during_bench", action="store_true",
                   help="let an explicit --lock share launch while another session holds machlock bench "
                        "(default: it waits as build; PROTOCOL 9: timing windows are exclusive)")
    p.add_argument("--label", default="")


def _request_from(args: argparse.Namespace) -> launch.Request:
    keys = launch.Request.__dataclass_fields__
    return launch.Request.from_dict({k: v for k, v in vars(args).items() if k in keys})


def cmd_binaries(args) -> int:
    inv = binaries_mod.inventory(refresh=args.refresh, include_private=args.all)
    if args.json:
        out = {"generated": inv["generated"], "binaries": [b.to_dict() for b in inv["binaries"]],
               "other_solvers": [b.to_dict() for b in inv["other_solvers"]], "hidden_private": inv["hidden_private"],
               "hidden_dev": inv["hidden_dev"]}
        print(json.dumps(out, indent=1))
        return 0
    print(f"abrun {VERSION} binaries (generated {inv['generated']}; cache {binaries_mod.CACHE_PATH})")
    for b in inv["binaries"]:
        flag = "" if b.launchable else f"  [not launchable: {b.launch_reason}]"
        print(f"  {b.id:<22} {b.label}{flag}")
        print(f"  {'':<22} path {b.path}")
        print(f"  {'':<22} sha256 {b.sha256[:12] if b.sha256 else '?'}{' (SHA256SUMS ok)' if b.sha_verified else ''}"
              f"  dylib {Path(b.dylib_path).name + ' ' + (b.dylib_sha256 or '?')[:12] if b.dylib_path else '-'}"
              f"  schema {b.profile_schema}  features {len(b.features)}  citable {b.citable}")
    if inv["other_solvers"]:
        print("other solvers (informational, not launchable):")
        for b in inv["other_solvers"]:
            print(f"  {b.id:<22} {b.label}")
    if inv["hidden_private"]:
        print("hidden private builds:", ", ".join(inv["hidden_private"]))
    if inv["hidden_dev"]:
        print("hidden dev builds (use --all):", ", ".join(inv["hidden_dev"]))
    return 0


def cmd_plan(args) -> int:
    req = _request_from(args)
    res = runner.plan_only(req)
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        p = res["plan"]
        print(f"run id would be: {res['run_id_would_be']}")
        if p:
            print(f"cwd: {p['cwd']}")
            print(f"lock: requested {p['lock_requested']} -> would resolve to {p['lock_would_resolve_to']}; launcher {p['launcher']}; "
                  f"timing grade possible: {p['timing_grade_possible']}")
            for name in ("wrapper_argv", "shim_argv", "solver_argv"):
                print(f"{name}:")
                for tok in p[name]:
                    print(f"    {tok}")
            print("env set:")
            for k, v in p["env_set"].items():
                print(f"    {k}={v}")
            print(f"env removed: {p['env_removed']}")
            for n in p["notes"]:
                print(f"note: {n}")
            print(f"shell: {p['shell']}")
    for v in res["violations"]:
        print(f"VIOLATION: {v}")
    return 1 if res["violations"] else 0


def cmd_run(args) -> int:
    req = _request_from(args)
    try:
        rec = runner.run_sync(req, run_dir=args.run_dir, log=print,
                              allow_share_during_bench=bool(getattr(args, "allow_share_during_bench", False)))
    except runner.PlanError as exc:
        for v in exc.violations:
            print(f"VIOLATION: {v}")
        return 2
    s = runner.REGISTRY.get(rec["id"])
    summ = s.summary() if s else {}
    print(json.dumps({k: summ.get(k) for k in ("id", "state", "total_energy_Ry", "scf_iterations", "wall_qe_s", "timing_grade",
                                                  "gemm_calls", "gpu_calls", "fallbacks", "gpu_verified", "lib_ok")}, indent=1))
    print(f"run dir: {rec['run_dir']}")
    return 0 if rec["state"] == "done" else 1


def cmd_compare(args) -> int:
    recs = []
    for x in args.ids:
        rec = runner.load_record(x)
        if rec is None:
            print(f"no run record for {x}", file=sys.stderr)
            return 2
        recs.append(rec)
    res = compare_mod.compare(recs, args.ref, args.energy_tol, runner.Pins().load())
    fmt = "json" if args.json else ("csv" if args.csv else "md")
    print(compare_mod.export(res, fmt))
    return 0


def cmd_serve(args) -> int:
    """HTTP server + single-page UI on 127.0.0.1:<port> (server.py); Ctrl-C cancels the running job and persists the queue."""
    from . import server
    return server.main(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="abrun", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"abrun {VERSION} ({INTERPRETER}; runs in {RUNS_DIR})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("binaries", help="catalog of pw.x binaries (read-only probes, cached)")
    p.add_argument("--all", action="store_true", help="include dev builds and name the private research builds")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_binaries)
    p = sub.add_parser("plan", help="show the argv/env a request would launch with (nothing is started)")
    _add_request_flags(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run", help="run one job in-process (headless twin of POST /api/runs)")
    _add_request_flags(p)
    p.add_argument("--run-dir", default=None, help="run directory (default tools/abrun/runs/<id>)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("compare", help="compare run records")
    p.add_argument("ids", nargs="+", help="run ids (under runs/) or run directories")
    p.add_argument("--ref", default=None)
    p.add_argument("--energy-tol", dest="energy_tol", type=float, default=None)
    p.add_argument("--md", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--csv", action="store_true")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("serve", help="HTTP API + SSE + browser UI on 127.0.0.1 (default port 8766 or ABRUN_PORT)")
    p.add_argument("--port", type=int, default=int(os.environ.get("ABRUN_PORT", "8766") or 8766))
    p.add_argument("--open", action="store_true", help="open the UI in the default browser")
    p.add_argument("--fake-pw", action="store_true", help="register tests/fake_pw.py as binaries 'fake' and 'reference-fake' (no solver, no machlock)")
    p.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

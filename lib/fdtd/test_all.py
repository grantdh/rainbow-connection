"""Run every lib/fdtd conformance suite at once: the case-file tests, then the backend launches of the MEEP,
LoopFDTD and MEEPhistopheles suites in one pool (--jobs N at a time, ring-downs first), then each suite's gates.

Run: MUPPET_SHARE=1 python3 lib/fdtd/test_all.py --jobs 4 [--skip-ringdown]

Concurrency changes only the wall times: every record carries `concurrent_jobs` (the effective number, 1 under the
shared machine lock), the reported wall times say when they were measured alongside other runs, and no gate reads
a wall time. Without MUPPET_SHARE=1, machlock serialises the launches whatever N is. The as-shipped LoopFDTD C1
ring-down gate is red by design (boundary-guard defect), so a full run exits 1 with exactly that FAIL.
"""
from __future__ import annotations

import sys
import time

from testlib import exlib, jobs_arg, print_failures, run_jobs, share_note, unique_jobs
import test_case
import test_loopfdtd_backend
import test_meep_backend
import test_meephistopheles_backend

SUITES = [("MEEP 3-D", test_meep_backend), ("LoopFDTD", test_loopfdtd_backend),
          ("MEEPhistopheles + MEEP 2-D", test_meephistopheles_backend)]


def main(argv: list[str]) -> int:
    n = jobs_arg(argv)
    print("=== case files")
    test_case.main()

    jobs = unique_jobs([j for _, suite in SUITES for j in suite.jobs(argv)])
    print(f"\n=== {len(jobs)} backend launches, {n} at a time")
    note = share_note(n)
    if note:
        print(note)
    launches = exlib.Checks()
    tic = time.time()
    results = run_jobs(launches, jobs, n)
    wall = time.time() - tic

    blocks = [("case files", test_case.checks), ("launches", launches)]
    for title, suite in SUITES:
        print(f"\n=== {title}")
        checks = exlib.Checks()
        suite.gate(checks, results, argv)
        blocks.append((title, checks))

    print(f"\n=== summary (launches took {wall:.0f} s with --jobs {n}; observation)")
    for title, checks in blocks:
        print(f"  {title}: {checks.summary()}")
        print_failures(checks, "    ")
    return 0 if all(checks.all_pass for _, checks in blocks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

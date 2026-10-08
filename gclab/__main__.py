"""Command line front end: bench, fuzz and heapmap."""
from __future__ import annotations

import argparse
import random
import sys
from typing import List, Optional

from . import COLLECTORS
from .heap import OutOfMemory, Stats
from .mutator import Lockstep, VerificationError, fuzz
from .workloads import WORKLOADS

ROOTS = 16


def _run(name: str, workload: str, heap: int, scale: int, seed: int):
    """Run one workload. Stats are snapshotted before the final verifying collection."""
    gc = COLLECTORS[name](heap, num_roots=ROOTS)
    lock = Lockstep(gc)
    status = "ok"
    try:
        WORKLOADS[workload](lock, random.Random(seed), scale)
        lock.verify()
        frag = gc.fragmentation()
        stats = Stats(**gc.stats.as_dict())
        gc.collect()
        if name == "refcount":
            lock.verify()
            leaked = sum(1 for _ in gc.objects()) - len(lock.model.reachable())
            status = f"leak {leaked}" if leaked else "ok"
        else:
            lock.verify(complete=True)
    except OutOfMemory:
        status, frag, stats = "OOM", gc.fragmentation(), gc.stats
    return stats, status, frag


def cmd_bench(args: argparse.Namespace) -> int:
    workloads = list(WORKLOADS) if args.workload == "all" else [args.workload]
    collectors = list(COLLECTORS) if args.collector == "all" else [args.collector]
    header = f"{'collector':<14}{'status':<10}{'GCs':>6}{'minor':>7}{'GC work':>10}{'max pause':>11}{'copied':>9}{'barrier':>9}{'frag':>7}"
    for w in workloads:
        print(f"\n== {w}: heap {args.heap} words, scale {args.scale}, seed {args.seed}")
        print(header)
        print("=" * len(header))
        for c in collectors:
            s, status, frag = _run(c, w, args.heap, args.scale, args.seed)
            print(f"{c:<14}{status:<10}{s.collections:>6}{s.minor_collections:>7}{s.work():>10}"
                  f"{s.max_pause:>11}{s.words_copied:>9}{s.barrier_hits:>9}{frag:>7.2f}")
    return 0


def cmd_fuzz(args: argparse.Namespace) -> int:
    collectors = list(COLLECTORS) if args.collector == "all" else [args.collector]
    failures = 0
    for c in collectors:
        cls = COLLECTORS[c]
        ok = 0
        for seed in range(args.seeds):
            try:
                fuzz(lambda: cls(args.heap, num_roots=8), seed, steps=args.steps,
                     complete=(c != "refcount"))
                ok += 1
            except (VerificationError, OutOfMemory) as e:
                failures += 1
                print(f"  {c} seed {seed}: {type(e).__name__}: {e}")
        print(f"{c:<14}{ok}/{args.seeds} seeds verified")
    return 1 if failures else 0


def cmd_heapmap(args: argparse.Namespace) -> int:
    for c in ("marksweep", "markcompact"):
        gc = COLLECTORS[c](args.heap, num_roots=ROOTS)
        lock = Lockstep(gc)
        WORKLOADS[args.workload](lock, random.Random(args.seed), args.scale)
        gc.collect()
        print(f"{c:<13} |{gc.occupancy_map(args.width)}|  frag {gc.fragmentation():.2f}")
    print("legend: '#' >85% live, '+' >50%, ':' some, '.' free")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gclab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="compare collectors on synthetic workloads")
    b.add_argument("--workload", default="all", choices=["all", *WORKLOADS])
    b.add_argument("--collector", default="all", choices=["all", *COLLECTORS])
    b.add_argument("--heap", type=int, default=4000)
    b.add_argument("--scale", type=int, default=10)
    b.add_argument("--seed", type=int, default=1)
    b.set_defaults(fn=cmd_bench)

    f = sub.add_parser("fuzz", help="random programs checked against the reference model")
    f.add_argument("--collector", default="all", choices=["all", *COLLECTORS])
    f.add_argument("--seeds", type=int, default=50)
    f.add_argument("--steps", type=int, default=600)
    f.add_argument("--heap", type=int, default=1200)
    f.set_defaults(fn=cmd_fuzz)

    h = sub.add_parser("heapmap", help="show fragmentation: marksweep vs markcompact")
    h.add_argument("--workload", default="fragment", choices=list(WORKLOADS))
    h.add_argument("--heap", type=int, default=4000)
    h.add_argument("--scale", type=int, default=2)
    h.add_argument("--seed", type=int, default=1)
    h.add_argument("--width", type=int, default=64)
    h.set_defaults(fn=cmd_heapmap)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

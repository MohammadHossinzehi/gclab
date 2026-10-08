"""Synthetic mutator workloads, each shaped to stress a different collector trait.

Every workload takes a Lockstep driver, so a benchmark run is also a full
correctness check against the reference model.
"""
from __future__ import annotations

import random
from typing import Callable, Dict

from .mutator import Lockstep

Workload = Callable[[Lockstep, random.Random, int], None]


def _tree(lock: Lockstep, r: int, depth: int) -> None:
    lock.new(r, 2)
    if depth > 0:
        _tree(lock, r + 1, depth - 1)
        lock.store(r, 0, r + 1)
        _tree(lock, r + 1, depth - 1)
        lock.store(r, 1, r + 1)
        lock.clear(r + 1)


def churn(lock: Lockstep, rng: random.Random, scale: int) -> None:
    """Almost everything dies young; a slow trickle survives in a list at r0."""
    for i in range(scale * 400):
        lock.new(1 + rng.randrange(4), rng.randint(0, 3))
        if i % 50 == 0:
            lock.new(6, 2)
            lock.store(6, 0, 0)
            lock.mov(0, 6)
            lock.clear(6)


def trees(lock: Lockstep, rng: random.Random, scale: int) -> None:
    """GCBench in miniature: one long lived tree plus many short lived ones."""
    _tree(lock, 0, 7)
    for _ in range(scale * 6):
        _tree(lock, 1, rng.randint(3, 6))
        lock.clear(1)


def cache(lock: Lockstep, rng: random.Random, scale: int) -> None:
    """A long lived table constantly repointed at fresh objects (old to young stores)."""
    lock.new(0, 32)
    for i in range(scale * 400):
        lock.new(1, 2)
        lock.seti(2, i)
        lock.store(1, 0, 2)
        if rng.random() < 0.25:
            lock.store(0, rng.randrange(32), 1)
        lock.clear(1)


def cycles(lock: Lockstep, rng: random.Random, scale: int) -> None:
    """Rings of 2 to 8 objects that become garbage as a whole. Reference counting leaks them."""
    for _ in range(scale * 60):
        n = rng.randint(2, 8)
        lock.new(1, 2)
        lock.mov(2, 1)
        for _ in range(n - 1):
            lock.new(3, 2)
            lock.store(2, 0, 3)
            lock.mov(2, 3)
        lock.store(2, 0, 1)
        lock.clear(1)
        lock.clear(2)
        lock.clear(3)


def fragment(lock: Lockstep, rng: random.Random, scale: int) -> None:
    """Interleave survivors with garbage of varied sizes, then ask for big blocks."""
    for rnd in range(scale * 3):
        lock.clear(0)
        for i in range(120):
            lock.new(1, rng.choice((0, 1, 6, 12)))
            if i % 3 == 0 and lock.model.objs[lock.model.roots[1][1]]:  # type: ignore[index]
                lock.store(1, 0, 0)
                lock.mov(0, 1)
        for _ in range(8):
            lock.new(2, 40)
        lock.clear(1)
        lock.clear(2)


WORKLOADS: Dict[str, Workload] = {
    "churn": churn,
    "trees": trees,
    "cache": cache,
    "cycles": cycles,
    "fragment": fragment,
}

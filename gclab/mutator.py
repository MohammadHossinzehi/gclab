"""The mutator, an address free reference model, and the heap verifier.

The Mutator is a tiny register machine. Its registers are the collector's
root set, and it can only touch the heap through six instructions. The
Model runs the very same instructions on plain Python dicts keyed by object
id, with no addresses and no collector. verify() then checks that the real
heap, after any amount of moving, sweeping and compacting, is isomorphic to
the model's reachable graph. That one check catches dangling pointers,
missed root updates, lost objects, corrupted fields and duplicated copies.
"""
from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple, Union

from .heap import HEADER, INFO, NULL, OID, Collector, addr_of, is_num, is_ptr, num, num_of, ptr

Value = Union[None, Tuple[str, int]]


class VerificationError(AssertionError):
    pass


class Mutator:
    def __init__(self, gc: Collector) -> None:
        self.gc = gc
        self.next_oid = 1
        self.ops = 0

    def new(self, r: int, nfields: int) -> int:
        oid = self.next_oid
        self.next_oid += 1
        a = self.gc.alloc(nfields, oid)
        self.gc.set_root(r, ptr(a))
        self.ops += 1
        return oid

    def load(self, dst: int, src: int, i: int) -> None:
        self.gc.set_root(dst, self.gc.read(addr_of(self.gc.roots[src]), i))
        self.ops += 1

    def store(self, dst: int, i: int, src: int) -> None:
        self.gc.write(addr_of(self.gc.roots[dst]), i, self.gc.roots[src])
        self.ops += 1

    def seti(self, r: int, n: int) -> None:
        self.gc.set_root(r, num(n))
        self.ops += 1

    def mov(self, dst: int, src: int) -> None:
        self.gc.set_root(dst, self.gc.roots[src])
        self.ops += 1

    def clear(self, r: int) -> None:
        self.gc.set_root(r, NULL)
        self.ops += 1


class Model:
    """Reference semantics: objects are lists, references are object ids."""

    def __init__(self, num_roots: int) -> None:
        self.roots: List[Value] = [None] * num_roots
        self.objs: Dict[int, List[Value]] = {}

    def new(self, r: int, nfields: int, oid: int) -> None:
        self.objs[oid] = [None] * nfields
        self.roots[r] = ("r", oid)

    def load(self, dst: int, src: int, i: int) -> None:
        self.roots[dst] = self.objs[self.roots[src][1]][i]  # type: ignore[index]

    def store(self, dst: int, i: int, src: int) -> None:
        self.objs[self.roots[dst][1]][i] = self.roots[src]  # type: ignore[index]

    def seti(self, r: int, n: int) -> None:
        self.roots[r] = ("i", n)

    def mov(self, dst: int, src: int) -> None:
        self.roots[dst] = self.roots[src]

    def clear(self, r: int) -> None:
        self.roots[r] = None

    def reachable(self) -> Set[int]:
        seen: Set[int] = set()
        stack = [v[1] for v in self.roots if v is not None and v[0] == "r"]
        while stack:
            o = stack.pop()
            if o in seen:
                continue
            seen.add(o)
            stack.extend(v[1] for v in self.objs[o] if v is not None and v[0] == "r")
        return seen


def verify(gc: Collector, model: Model, complete: bool = False) -> None:
    """Assert the heap matches the model. With complete=True also assert no garbage survived."""
    m = gc.mem
    addr_to_oid: Dict[int, int] = {}
    oid_to_addr: Dict[int, int] = {}
    for a in gc.objects():
        oid = m[a + OID]
        if oid in oid_to_addr:
            raise VerificationError(f"object {oid} exists twice (at {oid_to_addr[oid]} and {a})")
        addr_to_oid[a] = oid
        oid_to_addr[oid] = a

    def decode(w: int, where: str) -> Value:
        if w == NULL:
            return None
        if is_num(w):
            return ("i", num_of(w))
        if is_ptr(w):
            a = addr_of(w)
            if a not in addr_to_oid:
                raise VerificationError(f"dangling pointer to {a} in {where}")
            return ("r", addr_to_oid[a])
        raise VerificationError(f"malformed word {w} in {where}")

    for i, w in enumerate(gc.roots):
        got = decode(w, f"root {i}")
        if got != model.roots[i]:
            raise VerificationError(f"root {i}: heap has {got}, model has {model.roots[i]}")

    live = model.reachable()
    for oid in live:
        if oid not in oid_to_addr:
            raise VerificationError(f"reachable object {oid} was freed")
        a = oid_to_addr[oid]
        expect = model.objs[oid]
        nf = m[a + INFO] >> 8
        if nf != len(expect):
            raise VerificationError(f"object {oid} has {nf} fields, expected {len(expect)}")
        for i in range(nf):
            got = decode(m[a + HEADER + i], f"object {oid} field {i}")
            if got != expect[i]:
                raise VerificationError(f"object {oid} field {i}: heap {got}, model {expect[i]}")

    if complete:
        extra = set(oid_to_addr) - live
        if extra:
            raise VerificationError(f"{len(extra)} unreachable objects survived a full collection")


class Lockstep:
    """Drive a Mutator and a Model with identical instructions."""

    def __init__(self, gc: Collector) -> None:
        self.gc = gc
        self.mut = Mutator(gc)
        self.model = Model(len(gc.roots))

    def new(self, r: int, nfields: int) -> int:
        oid = self.mut.new(r, nfields)
        self.model.new(r, nfields, oid)
        return oid

    def __getattr__(self, op: str):
        if op not in ("load", "store", "seti", "mov", "clear"):
            raise AttributeError(op)

        def run(*args: int) -> None:
            getattr(self.mut, op)(*args)
            getattr(self.model, op)(*args)
        return run

    def verify(self, complete: bool = False) -> None:
        verify(self.gc, self.model, complete)


def random_program(lock: Lockstep, rng: random.Random, steps: int,
                   max_fields: int = 4, verify_every: int = 0) -> None:
    """Run `steps` random but always valid instructions in lockstep."""
    n = len(lock.gc.roots)
    model = lock.model
    for step in range(1, steps + 1):
        refs = [i for i, v in enumerate(model.roots) if v is not None and v[0] == "r"]
        roll = rng.random()
        if roll < 0.30 or not refs:
            lock.new(rng.randrange(n), rng.randint(0, max_fields))
        elif roll < 0.55:
            dst = rng.choice(refs)
            nf = len(model.objs[model.roots[dst][1]])  # type: ignore[index]
            if nf:
                lock.store(dst, rng.randrange(nf), rng.randrange(n))
        elif roll < 0.75:
            src = rng.choice(refs)
            nf = len(model.objs[model.roots[src][1]])  # type: ignore[index]
            if nf:
                lock.load(rng.randrange(n), src, rng.randrange(nf))
        elif roll < 0.85:
            lock.mov(rng.randrange(n), rng.randrange(n))
        elif roll < 0.93:
            lock.clear(rng.randrange(n))
        else:
            lock.seti(rng.randrange(n), rng.randint(-1000, 1000))
        if verify_every and step % verify_every == 0:
            lock.verify()


def fuzz(make_gc, seed: int, steps: int = 400, verify_every: int = 20,
         complete: bool = True) -> Lockstep:
    rng = random.Random(seed)
    lock = Lockstep(make_gc())
    random_program(lock, rng, steps, verify_every=verify_every)
    lock.verify()
    lock.gc.collect()
    lock.verify(complete=complete)
    return lock

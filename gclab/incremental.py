"""Incremental tri colour mark sweep with a Dijkstra insertion barrier."""
from __future__ import annotations

from typing import Iterator, List

from .heap import HEADER, INFO, MARK, Collector, FreeListSpace, OutOfMemory


class IncrementalMarkSweep(Collector):
    """Interleave marking with the mutator in small, bounded slices.

    Colours: white = unmarked, grey = marked and still on the grey stack,
    black = marked and scanned. The invariant that makes incremental marking
    safe is "no black object points to a white object". The mutator can
    break it by storing a white pointer into a black object and then
    deleting the last other path to it; the Dijkstra barrier repairs that by
    shading the stored target grey on every pointer store while marking.

    Registers are not barriered (that would be far too expensive in a real
    VM), so mark termination rescans the roots once in a short stop the
    world step, then drains whatever turned grey. New objects are allocated
    black during a cycle, which is why they survive to the next one even if
    they die immediately ("floating garbage").

    Set barrier=False to watch the verifier catch the lost object bug.
    """

    name = "incremental"

    def __init__(self, heap_words: int, num_roots: int = 8, slice_words: int = 48,
                 trigger: float = 0.3, barrier: bool = True) -> None:
        super().__init__(heap_words, num_roots)
        self.space = FreeListSpace(self.mem, 0, heap_words)
        self.slice_words = slice_words
        self.trigger = trigger
        self.barrier = barrier
        self.marking = False
        self.grey: List[int] = []
        self.cycles = 0

    # tri colour machinery
    def shade(self, a: int) -> None:
        m = self.mem
        if not m[a + INFO] & MARK:
            m[a + INFO] |= MARK
            self.grey.append(a)
            if len(self.grey) > self.stats.max_mark_stack:
                self.stats.max_mark_stack = len(self.grey)

    def start_cycle(self) -> None:
        before = self.stats.work()
        self.marking = True
        for a in self.root_addresses():
            self.shade(a)
        self.stats.words_traced += len(self.roots)
        self.stats.record_pause(self.stats.work() - before)

    def step(self, budget: int) -> bool:
        """Scan grey objects until `budget` words are traced. True when grey is empty."""
        m, st = self.mem, self.stats
        while self.grey and budget > 0:
            a = self.grey.pop()
            words = HEADER + (m[a + INFO] >> 8)
            st.words_traced += words
            budget -= words
            for slot in self.pointer_slots(a):
                self.shade(m[slot] >> 2)
        return not self.grey

    def finish_cycle(self) -> None:
        st = self.stats
        before = st.work()
        for a in self.root_addresses():
            self.shade(a)
        self.step(float("inf"))  # type: ignore[arg-type]
        freed, visited = self.space.sweep()
        st.objects_freed += freed
        st.words_swept += visited
        st.collections += 1
        self.marking = False
        self.cycles += 1
        st.record_pause(st.work() - before)

    # mutator interface
    def alloc(self, nfields: int, oid: int) -> int:
        st = self.stats
        if not self.marking and self.space.free_words() < self.trigger * len(self.mem):
            self.start_cycle()
        elif self.marking:
            before = st.work()
            done = self.step(self.slice_words)
            st.record_pause(st.work() - before)
            if done:
                self.finish_cycle()

        words = HEADER + nfields
        got = self.space.allocate(words)
        if got is None:
            if self.marking:
                self.finish_cycle()
                got = self.space.allocate(words)
            if got is None:
                self.collect()
                got = self.space.allocate(words)
            if got is None:
                raise OutOfMemory(f"{self.name}: no block of {words} words")
        a, size = got
        self.init_object(a, size, nfields, oid)
        if self.marking:
            self.mem[a + INFO] |= MARK  # allocate black
        return a

    def write(self, obj: int, i: int, value: int) -> None:
        super().write(obj, i, value)
        if self.marking and self.barrier and value & 3 == 1:
            self.stats.barrier_hits += 1
            self.shade(value >> 2)

    def collect(self) -> None:
        if self.marking:
            self.finish_cycle()
        self.start_cycle()
        self.finish_cycle()

    def objects(self) -> Iterator[int]:
        return self.space.objects()

    def free_words(self) -> int:
        return self.space.free_words()

    def fragmentation(self) -> float:
        return self.space.fragmentation()

"""Naive reference counting, the baseline every tracing collector is compared to."""
from __future__ import annotations

from typing import Iterator

from .heap import AUX, HEADER, INFO, Collector, FreeListSpace, OutOfMemory


class RefCounting(Collector):
    """Count incoming references in the AUX header word; free at zero.

    Garbage is reclaimed the instant it is created and pause time is
    bounded by the size of the structure being freed, but every pointer
    store and register move pays an increment and a decrement, and cyclic
    garbage is never reclaimed. collect() only coalesces free blocks; it
    cannot find cycles. That leak is deliberate and covered by the tests.
    """

    name = "refcount"

    def __init__(self, heap_words: int, num_roots: int = 8) -> None:
        super().__init__(heap_words, num_roots)
        self.space = FreeListSpace(self.mem, 0, heap_words)
        self.rc_ops = 0

    def alloc(self, nfields: int, oid: int) -> int:
        words = HEADER + nfields
        got = self.space.allocate(words)
        if got is None:
            self.collect()
            got = self.space.allocate(words)
            if got is None:
                raise OutOfMemory(f"{self.name}: no block of {words} words (cycles leak under RC)")
        a, size = got
        self.init_object(a, size, nfields, oid)
        return a

    def _inc(self, w: int) -> None:
        if w & 3 == 1:
            self.rc_ops += 1
            self.mem[(w >> 2) + AUX] += 1

    def _dec(self, w: int) -> None:
        if w & 3 != 1:
            return
        m, st = self.mem, self.stats
        self.rc_ops += 1
        a = w >> 2
        m[a + AUX] -= 1
        if m[a + AUX] > 0:
            return
        before = st.work()
        doomed = [a]
        while doomed:
            o = doomed.pop()
            st.words_traced += HEADER + (m[o + INFO] >> 8)
            for slot in self.pointer_slots(o):
                t = m[slot] >> 2
                self.rc_ops += 1
                m[t + AUX] -= 1
                if m[t + AUX] == 0:
                    doomed.append(t)
            self.space.release(o)
            st.objects_freed += 1
        st.record_pause(st.work() - before)

    def set_root(self, i: int, value: int) -> None:
        self._inc(value)
        old = self.roots[i]
        self.roots[i] = value
        self._dec(old)

    def write(self, obj: int, i: int, value: int) -> None:
        self._check_field(obj, i)
        slot = obj + HEADER + i
        self._inc(value)
        old = self.mem[slot]
        self.mem[slot] = value
        self._dec(old)

    def collect(self) -> None:
        before = self.stats.work()
        self.stats.words_swept += self.space.coalesce()
        self.stats.collections += 1
        self.stats.record_pause(self.stats.work() - before)

    def objects(self) -> Iterator[int]:
        return self.space.objects()

    def free_words(self) -> int:
        return self.space.free_words()

    def fragmentation(self) -> float:
        return self.space.fragmentation()

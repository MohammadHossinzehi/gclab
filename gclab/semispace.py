"""Cheney's semispace copying collector (1970)."""
from __future__ import annotations

from typing import Iterator

from .heap import AUX, FORWARDED, HEADER, INFO, SIZE, Collector, OutOfMemory, ptr


class SemiSpace(Collector):
    """Split the heap in two and copy survivors from one half to the other.

    Cheney's trick is that to space itself is the breadth first queue: a
    scan pointer chases the free pointer, and every object between them is
    copied but not yet scanned. No mark stack, no recursion, and the cost
    is proportional to live data only; garbage is never touched at all.
    A forwarding address left in the old copy (FORWARDED flag + AUX word)
    makes sure shared objects and cycles are copied exactly once.
    """

    name = "semispace"
    moving = True

    def __init__(self, heap_words: int, num_roots: int = 8) -> None:
        super().__init__(heap_words, num_roots)
        self.half = heap_words // 2
        self.lo = 0
        self.top = 0
        self._free = 0

    def capacity(self) -> int:
        return self.half

    def alloc(self, nfields: int, oid: int) -> int:
        words = HEADER + nfields
        if self.top + words > self.lo + self.half:
            self.collect()
            if self.top + words > self.lo + self.half:
                raise OutOfMemory(f"{self.name}: {words} words requested, {self.free_words()} free")
        a = self.top
        self.top += words
        self.init_object(a, words, nfields, oid)
        return a

    def _forward(self, w: int) -> int:
        m = self.mem
        a = w >> 2
        if m[a + INFO] & FORWARDED:
            return ptr(m[a + AUX])
        size = m[a + SIZE]
        new = self._free
        m[new:new + size] = m[a:a + size]
        self._free += size
        self.stats.words_copied += size
        m[a + INFO] |= FORWARDED
        m[a + AUX] = new
        return ptr(new)

    def collect(self) -> None:
        m, st = self.mem, self.stats
        before = st.work()
        old_lo, old_top = self.lo, self.top
        self.lo = self.half if self.lo == 0 else 0
        self._free = scan = self.lo

        for i, r in enumerate(self.roots):
            if r & 3 == 1:
                self.roots[i] = self._forward(r)
        while scan < self._free:
            st.words_traced += HEADER + (m[scan + INFO] >> 8)
            for slot in self.pointer_slots(scan):
                m[slot] = self._forward(m[slot])
            scan += m[scan + SIZE]

        freed = sum(1 for a in self.walk(old_lo, old_top) if not m[a + INFO] & FORWARDED)
        self.top = self._free
        st.objects_freed += freed
        st.collections += 1
        st.record_pause(st.work() - before)

    def objects(self) -> Iterator[int]:
        return self.walk(self.lo, self.top)

    def free_words(self) -> int:
        return self.lo + self.half - self.top

    def fragmentation(self) -> float:
        return 0.0

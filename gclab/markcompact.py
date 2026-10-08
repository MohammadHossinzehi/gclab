"""Lisp 2 sliding mark compact with bump allocation."""
from __future__ import annotations

from typing import Iterator

from .heap import AUX, HEADER, INFO, MARK, SIZE, Collector, OutOfMemory, ptr


class MarkCompact(Collector):
    """Mark, then slide every survivor down to the bottom of the heap.

    The classic Lisp 2 algorithm needs three linear passes after marking:

    1. compute each survivor's new address and stash it in the AUX word,
    2. rewrite every root and every pointer field through those addresses,
    3. move the objects (always downwards, so a left to right copy is safe).

    The payoff is zero fragmentation, allocation by pointer bump, and
    preserved allocation order (good locality). The price is touching the
    heap three times per collection.
    """

    name = "markcompact"
    moving = True

    def __init__(self, heap_words: int, num_roots: int = 8) -> None:
        super().__init__(heap_words, num_roots)
        self.top = 0

    def alloc(self, nfields: int, oid: int) -> int:
        words = HEADER + nfields
        if self.top + words > len(self.mem):
            self.collect()
            if self.top + words > len(self.mem):
                raise OutOfMemory(f"{self.name}: {words} words requested, {self.free_words()} free")
        a = self.top
        self.top += words
        self.init_object(a, words, nfields, oid)
        return a

    def collect(self) -> None:
        m, st = self.mem, self.stats
        before = st.work()
        self.mark_from(self.root_addresses())

        # Pass 1: forwarding addresses.
        free = 0
        for a in self.walk(0, self.top):
            st.words_swept += HEADER
            if m[a + INFO] & MARK:
                m[a + AUX] = free
                free += m[a + SIZE]

        # Pass 2: update references.
        for i, r in enumerate(self.roots):
            if r & 3 == 1:
                self.roots[i] = ptr(m[(r >> 2) + AUX])
        for a in self.walk(0, self.top):
            if m[a + INFO] & MARK:
                st.words_traced += HEADER + (m[a + INFO] >> 8)
                for slot in self.pointer_slots(a):
                    m[slot] = ptr(m[(m[slot] >> 2) + AUX])

        # Pass 3: slide.
        a = 0
        freed = 0
        while a < self.top:
            size = m[a + SIZE]
            info = m[a + INFO]
            if info & MARK:
                dst = m[a + AUX]
                if dst != a:
                    m[dst:dst + size] = m[a:a + size]
                    st.words_copied += size
                m[dst + INFO] = info & ~MARK
                m[dst + AUX] = 0
            else:
                freed += 1
            a += size
        for k in range(free, self.top):
            m[k] = 0
        self.top = free

        st.objects_freed += freed
        st.collections += 1
        st.record_pause(st.work() - before)

    def objects(self) -> Iterator[int]:
        return self.walk(0, self.top)

    def free_words(self) -> int:
        return len(self.mem) - self.top

    def fragmentation(self) -> float:
        return 0.0

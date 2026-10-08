"""Two generation collector: copying nursery, mark sweep old space."""
from __future__ import annotations

from typing import Iterator, List, Optional, Set

from .heap import (
    AUX, FORWARDED, HEADER, INFO, MARK, SIZE,
    Collector, FreeListSpace, OutOfMemory, ptr,
)


class Generational(Collector):
    """Exploit the weak generational hypothesis: most objects die young.

    Layout: mem[0:nursery) is a bump allocated nursery, the rest is an old
    space managed by a free list. A minor collection traces only the
    nursery, starting from the roots plus the remembered set, and promotes
    every survivor straight into old space ("en masse" promotion, so there
    is no aging). A major collection is a full mark sweep.

    The remembered set is maintained by a write barrier: whenever the
    mutator stores a nursery pointer into an old object, that old object is
    recorded. Without it a minor collection would miss nursery objects that
    are only reachable from old space. The flip side is nepotism: a dead old
    object in the remembered set still keeps its young referents alive until
    the next major collection. The test suite checks both behaviours.
    """

    name = "generational"
    moving = True

    def __init__(self, heap_words: int, num_roots: int = 8,
                 nursery_words: Optional[int] = None) -> None:
        super().__init__(heap_words, num_roots)
        self.nursery_end = nursery_words or heap_words // 4
        if not HEADER * 4 <= self.nursery_end <= heap_words - HEADER * 4:
            raise ValueError("bad nursery size")
        self.top = 0
        self.old = FreeListSpace(self.mem, self.nursery_end, heap_words)
        self.remembered: Set[int] = set()
        self._work_list: List[int] = []

    def in_nursery(self, a: int) -> bool:
        return a < self.nursery_end

    # allocation
    def alloc(self, nfields: int, oid: int) -> int:
        words = HEADER + nfields
        if words > self.nursery_end // 4:
            return self._alloc_old(words, nfields, oid)
        if self.top + words > self.nursery_end:
            self.minor()
        a = self.top
        self.top += words
        self.init_object(a, words, nfields, oid)
        return a

    def _alloc_old(self, words: int, nfields: int, oid: int) -> int:
        """Pretenure large objects so they are never copied."""
        got = self.old.allocate(words)
        if got is None:
            self.collect()
            got = self.old.allocate(words)
            if got is None:
                raise OutOfMemory(f"{self.name}: old space cannot fit {words} words")
        a, size = got
        self.init_object(a, size, nfields, oid)
        return a

    # barrier
    def write(self, obj: int, i: int, value: int) -> None:
        super().write(obj, i, value)
        if value & 3 == 1 and obj >= self.nursery_end and (value >> 2) < self.nursery_end:
            self.stats.barrier_hits += 1
            self.remembered.add(obj)

    # minor collection
    def _promote(self, w: int) -> int:
        m = self.mem
        a = w >> 2
        if a >= self.nursery_end:
            return w
        if m[a + INFO] & FORWARDED:
            return ptr(m[a + AUX])
        nf = m[a + INFO] >> 8
        words = HEADER + nf
        got = self.old.allocate(words)
        if got is None:
            raise OutOfMemory(f"{self.name}: old space full during promotion")
        new, size = got
        m[new:new + words] = m[a:a + words]
        m[new + SIZE] = size
        m[new + INFO] &= ~MARK
        m[a + INFO] |= FORWARDED
        m[a + AUX] = new
        self.stats.words_copied += words
        self._work_list.append(new)
        return ptr(new)

    def minor(self) -> None:
        m, st = self.mem, self.stats
        if self.old.free_words() < self.top:
            self.major()
        before = st.work()
        self._work_list = []
        for i, r in enumerate(self.roots):
            if r & 3 == 1:
                self.roots[i] = self._promote(r)
        for o in self.remembered:
            st.words_traced += HEADER + (m[o + INFO] >> 8)
            for slot in self.pointer_slots(o):
                m[slot] = self._promote(m[slot])
        while self._work_list:
            o = self._work_list.pop()
            st.words_traced += HEADER + (m[o + INFO] >> 8)
            for slot in self.pointer_slots(o):
                m[slot] = self._promote(m[slot])
        freed = sum(1 for a in self.walk(0, self.top) if not m[a + INFO] & FORWARDED)
        st.objects_freed += freed
        self.top = 0
        self.remembered.clear()
        st.minor_collections += 1
        st.record_pause(st.work() - before)

    # major collection
    def major(self) -> None:
        m, st = self.mem, self.stats
        before = st.work()
        self.mark_from(self.root_addresses())
        dead: Set[int] = set()
        freed, visited = self.old.sweep(dead.add)
        for a in self.walk(0, self.top):
            m[a + INFO] &= ~MARK
        self.remembered -= dead
        st.objects_freed += freed
        st.words_swept += visited
        st.collections += 1
        st.record_pause(st.work() - before)

    def collect(self) -> None:
        # Major first so dead old objects stop pinning young ones, then
        # minor to empty the nursery: afterwards nothing unreachable remains.
        self.major()
        self.minor()

    def objects(self) -> Iterator[int]:
        yield from self.walk(0, self.top)
        yield from self.old.objects()

    def free_words(self) -> int:
        return (self.nursery_end - self.top) + self.old.free_words()

    def fragmentation(self) -> float:
        return self.old.fragmentation()

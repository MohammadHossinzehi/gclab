"""Stop the world mark sweep over a first fit free list (McCarthy, 1960)."""
from __future__ import annotations

from typing import Iterator

from .heap import HEADER, Collector, FreeListSpace, OutOfMemory


class MarkSweep(Collector):
    """Non moving tracing collector.

    Mark: depth first from the roots with an explicit stack, so deep lists
    cannot blow the Python stack. Sweep: one linear pass that frees white
    objects and coalesces neighbouring free blocks in the same pass.
    Objects never move, so the heap fragments over time; compare the
    occupancy map with MarkCompact on the same workload.
    """

    name = "marksweep"

    def __init__(self, heap_words: int, num_roots: int = 8) -> None:
        super().__init__(heap_words, num_roots)
        self.space = FreeListSpace(self.mem, 0, heap_words)

    def alloc(self, nfields: int, oid: int) -> int:
        words = HEADER + nfields
        got = self.space.allocate(words)
        if got is None:
            self.collect()
            got = self.space.allocate(words)
            if got is None:
                raise OutOfMemory(
                    f"{self.name}: no block of {words} words "
                    f"(free {self.space.free_words()}, largest {self.space.largest_free()})"
                )
        a, size = got
        self.init_object(a, size, nfields, oid)
        return a

    def collect(self) -> None:
        st = self.stats
        before = st.work()
        self.mark_from(self.root_addresses())
        freed, visited = self.space.sweep()
        st.objects_freed += freed
        st.words_swept += visited
        st.collections += 1
        st.record_pause(st.work() - before)

    def objects(self) -> Iterator[int]:
        return self.space.objects()

    def free_words(self) -> int:
        return self.space.free_words()

    def fragmentation(self) -> float:
        return self.space.fragmentation()

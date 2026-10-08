"""Word addressed simulated heap shared by every collector.

Memory is a flat Python list of integers ("words"). Every block, live or
free, starts with a 4 word header so the heap can always be parsed by a
linear walk:

    word 0  SIZE  total block length in words, header included
    word 1  INFO  flag bits (low byte) | number of fields << 8
    word 2  OID   stable object id, used only by the verifier
    word 3  AUX   forwarding address, reference count or scratch

Field values are tagged words: 0 is null, (addr << 2) | 1 is a pointer and
(n << 2) | 2 is a small integer. Tagging is what lets a collector tell a
pointer from plain data without type maps, the same trick many real
runtimes (OCaml, V8 Smis, Lisp machines) use.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, fields as dc_fields
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

HEADER = 4
SIZE, INFO, OID, AUX = 0, 1, 2, 3

MARK = 1
FREE = 2
FORWARDED = 4
FLAG_MASK = 0xFF

NULL = 0


class OutOfMemory(Exception):
    """Raised when an allocation cannot be satisfied even after collecting."""


class UseAfterFree(Exception):
    """Raised when the mutator touches a block the collector has reclaimed."""


# Tagged value helpers

def ptr(addr: int) -> int:
    return (addr << 2) | 1


def num(n: int) -> int:
    return (n << 2) | 2


def is_ptr(w: int) -> bool:
    return w & 3 == 1


def is_num(w: int) -> bool:
    return w & 3 == 2


def addr_of(w: int) -> int:
    if not is_ptr(w):
        raise TypeError(f"word {w!r} is not a pointer")
    return w >> 2


def num_of(w: int) -> int:
    if not is_num(w):
        raise TypeError(f"word {w!r} is not an integer")
    return w >> 2


# Statistics

@dataclass
class Stats:
    """Work counters. One unit of work is one heap word touched by the GC."""

    collections: int = 0
    minor_collections: int = 0
    objects_allocated: int = 0
    words_allocated: int = 0
    objects_freed: int = 0
    words_traced: int = 0
    words_copied: int = 0
    words_swept: int = 0
    barrier_hits: int = 0
    max_mark_stack: int = 0
    max_pause: int = 0
    total_pause: int = 0
    pauses: int = 0

    def work(self) -> int:
        return self.words_traced + self.words_copied + self.words_swept

    def record_pause(self, work: int) -> None:
        self.pauses += 1
        self.total_pause += work
        if work > self.max_pause:
            self.max_pause = work

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# Free list space

class FreeListSpace:
    """First fit free list allocator over mem[start:end).

    Free blocks keep the normal 4 word header (with the FREE flag) so the
    region stays walkable. The list of free block addresses is kept sorted
    by address, which makes first fit equal to address ordered first fit,
    a policy known to fragment less than LIFO ordering.
    """

    def __init__(self, mem: List[int], start: int, end: int) -> None:
        if end - start < HEADER:
            raise ValueError("space too small")
        self.mem = mem
        self.start = start
        self.end = end
        self.free: List[int] = []
        self.reset()

    def reset(self) -> None:
        self._make_free(self.start, self.end - self.start)
        self.free = [self.start]

    def _make_free(self, a: int, size: int) -> None:
        m = self.mem
        m[a + SIZE] = size
        m[a + INFO] = FREE
        m[a + OID] = 0
        m[a + AUX] = 0

    def contains(self, a: int) -> bool:
        return self.start <= a < self.end

    def allocate(self, words: int) -> Optional[Tuple[int, int]]:
        """Return (address, block_size) or None. Splits when the tail is usable."""
        m = self.mem
        for idx, a in enumerate(self.free):
            size = m[a + SIZE]
            if size >= words:
                rest = size - words
                if rest >= HEADER:
                    self._make_free(a + words, rest)
                    self.free[idx] = a + words
                    return a, words
                del self.free[idx]
                return a, size
        return None

    def release(self, a: int) -> None:
        self._make_free(a, self.mem[a + SIZE])
        bisect.insort(self.free, a)

    def blocks(self) -> Iterator[int]:
        a, m, end = self.start, self.mem, self.end
        while a < end:
            yield a
            a += m[a + SIZE]

    def objects(self) -> Iterator[int]:
        m = self.mem
        for a in self.blocks():
            if not m[a + INFO] & FREE:
                yield a

    def free_words(self) -> int:
        return sum(self.mem[a + SIZE] for a in self.free)

    def largest_free(self) -> int:
        return max((self.mem[a + SIZE] for a in self.free), default=0)

    def fragmentation(self) -> float:
        """External fragmentation: 1 - largest free block / total free words."""
        total = self.free_words()
        return 0.0 if total == 0 else 1.0 - self.largest_free() / total

    def coalesce(self) -> int:
        """Merge adjacent free blocks. Returns words visited."""
        m = self.mem
        new_free: List[int] = []
        run = None
        visited = 0
        a = self.start
        while a < self.end:
            size = m[a + SIZE]
            visited += HEADER
            if m[a + INFO] & FREE:
                if run is None:
                    run = a
            elif run is not None:
                self._make_free(run, a - run)
                new_free.append(run)
                run = None
            a += size
        if run is not None:
            self._make_free(run, self.end - run)
            new_free.append(run)
        self.free = new_free
        return visited

    def sweep(self, on_free: Optional[Callable[[int], None]] = None) -> Tuple[int, int]:
        """Free unmarked objects, unmark survivors, coalesce free runs.

        Returns (objects_freed, words_visited).
        """
        m = self.mem
        new_free: List[int] = []
        run = None
        freed = 0
        visited = 0
        a = self.start
        while a < self.end:
            size = m[a + SIZE]
            info = m[a + INFO]
            visited += HEADER
            if info & FREE or not info & MARK:
                if not info & FREE:
                    freed += 1
                    if on_free is not None:
                        on_free(a)
                if run is None:
                    run = a
            else:
                m[a + INFO] = info & ~MARK
                if run is not None:
                    self._make_free(run, a - run)
                    new_free.append(run)
                    run = None
            a += size
        if run is not None:
            self._make_free(run, self.end - run)
            new_free.append(run)
        self.free = new_free
        return freed, visited


# Collector base class

class Collector:
    """Interface every collector implements, plus shared tracing helpers.

    The mutator only ever talks to a collector through alloc, read, write
    and set_root. That narrow surface is where barriers live, and it is
    what lets a moving collector relocate objects behind the mutator's back:
    registers (roots) are owned by the collector and get rewritten.
    """

    name = "abstract"
    moving = False

    def __init__(self, heap_words: int, num_roots: int = 8) -> None:
        if heap_words < 64:
            raise ValueError("heap must be at least 64 words")
        self.mem: List[int] = [0] * heap_words
        self.roots: List[int] = [NULL] * num_roots
        self.stats = Stats()

    # mutator interface
    def alloc(self, nfields: int, oid: int) -> int:
        raise NotImplementedError

    def read(self, obj: int, i: int) -> int:
        self._check_field(obj, i)
        return self.mem[obj + HEADER + i]

    def write(self, obj: int, i: int, value: int) -> None:
        self._check_field(obj, i)
        self.mem[obj + HEADER + i] = value

    def set_root(self, i: int, value: int) -> None:
        self.roots[i] = value

    def collect(self) -> None:
        """Run a complete collection: afterwards no unreachable object survives."""
        raise NotImplementedError

    def objects(self) -> Iterator[int]:
        """Addresses of every allocated (not freed) object, live or not."""
        raise NotImplementedError

    def free_words(self) -> int:
        raise NotImplementedError

    def capacity(self) -> int:
        """Words usable for allocation (a semispace heap only offers half)."""
        return len(self.mem)

    # helpers
    def nfields(self, a: int) -> int:
        return self.mem[a + INFO] >> 8

    def init_object(self, a: int, size: int, nfields: int, oid: int) -> None:
        m = self.mem
        m[a + SIZE] = size
        m[a + INFO] = nfields << 8
        m[a + OID] = oid
        m[a + AUX] = 0
        base = a + HEADER
        for k in range(base, base + nfields):
            m[k] = NULL
        self.stats.objects_allocated += 1
        self.stats.words_allocated += HEADER + nfields

    def _check_field(self, a: int, i: int) -> None:
        info = self.mem[a + INFO]
        if info & (FREE | FORWARDED):
            raise UseAfterFree(f"object at {a} is {'free' if info & FREE else 'stale (forwarded)'}")
        if not 0 <= i < info >> 8:
            raise IndexError(f"field {i} out of range for object at {a}")

    def pointer_slots(self, a: int) -> Iterator[int]:
        m = self.mem
        base = a + HEADER
        for slot in range(base, base + (m[a + INFO] >> 8)):
            if m[slot] & 3 == 1:
                yield slot

    def root_addresses(self) -> List[int]:
        return [r >> 2 for r in self.roots if r & 3 == 1]

    def walk(self, start: int, end: int) -> Iterator[int]:
        a, m = start, self.mem
        while a < end:
            yield a
            a += m[a + SIZE]

    def mark_from(self, addrs: Iterable[int], keep: Optional[Callable[[int], bool]] = None) -> int:
        """Depth first marking with an explicit stack. Returns objects marked."""
        m = self.mem
        stack: List[int] = []
        marked = 0
        for a in addrs:
            if not m[a + INFO] & MARK:
                m[a + INFO] |= MARK
                stack.append(a)
                marked += 1
        deepest = len(stack)
        st = self.stats
        while stack:
            a = stack.pop()
            st.words_traced += HEADER + (m[a + INFO] >> 8)
            for slot in self.pointer_slots(a):
                t = m[slot] >> 2
                if keep is not None and not keep(t):
                    continue
                if not m[t + INFO] & MARK:
                    m[t + INFO] |= MARK
                    stack.append(t)
                    marked += 1
            if len(stack) > deepest:
                deepest = len(stack)
        if deepest > st.max_mark_stack:
            st.max_mark_stack = deepest
        return marked

    def occupancy_map(self, width: int = 64) -> str:
        """One character per heap chunk: '#' full, '+' mostly, ':' some, '.' empty."""
        live = [0] * len(self.mem)
        for a in self.objects():
            size = HEADER + self.nfields(a)
            for k in range(a, a + size):
                live[k] = 1
        chunk = max(1, len(self.mem) // width)
        out = []
        for c in range(0, len(self.mem), chunk):
            frac = sum(live[c:c + chunk]) / len(live[c:c + chunk])
            out.append("#" if frac > 0.85 else "+" if frac > 0.5 else ":" if frac > 0.0 else ".")
        return "".join(out)

"""Test suite. Runs with `python -m unittest` (no dependencies) or pytest."""
import random
import unittest

from gclab import (
    COLLECTORS, Generational, IncrementalMarkSweep, Lockstep, MarkCompact, MarkSweep,
    OutOfMemory, RefCounting, SemiSpace, UseAfterFree, VerificationError, fuzz,
)
from gclab.heap import HEADER, INFO, FREE, FreeListSpace, addr_of, is_num, is_ptr, num, num_of, ptr
from gclab.workloads import WORKLOADS

TRACING = [MarkSweep, MarkCompact, SemiSpace, Generational, IncrementalMarkSweep]


class TaggingTest(unittest.TestCase):
    def test_round_trip(self):
        for n in (0, 1, -1, 12345, -98765):
            self.assertTrue(is_num(num(n)))
            self.assertEqual(num_of(num(n)), n)
        for a in (0, 4, 1000):
            self.assertTrue(is_ptr(ptr(a)))
            self.assertEqual(addr_of(ptr(a)), a)
        self.assertFalse(is_ptr(0) or is_num(0))

    def test_wrong_tag_raises(self):
        with self.assertRaises(TypeError):
            addr_of(num(3))
        with self.assertRaises(TypeError):
            num_of(ptr(3))


class FreeListTest(unittest.TestCase):
    def test_split_release_coalesce(self):
        mem = [0] * 100
        s = FreeListSpace(mem, 0, 100)

        def take(words):
            a, size = s.allocate(words)
            mem[a], mem[a + INFO] = size, 0  # what Collector.init_object would write
            return a

        a, b, c = take(10), take(10), take(10)
        self.assertEqual((a, b, c), (0, 10, 20))
        s.release(a)
        s.release(b)
        self.assertEqual(s.free_words(), 90)
        self.assertEqual(s.largest_free(), 70)
        s.coalesce()
        self.assertEqual(s.largest_free(), 70)
        self.assertEqual(sorted(mem[f] for f in s.free), [20, 70])

    def test_tiny_remainder_is_absorbed(self):
        mem = [0] * 12
        s = FreeListSpace(mem, 0, 12)
        a, size = s.allocate(10)  # 2 word tail cannot hold a header
        self.assertEqual((a, size), (0, 12))
        self.assertIsNone(s.allocate(4))


class FuzzTest(unittest.TestCase):
    """Random programs in lockstep with the reference model, every collector."""

    def check(self, cls, heap, seeds=25, steps=900):
        for seed in range(seeds):
            with self.subTest(collector=cls.name, seed=seed, heap=heap):
                lock = fuzz(lambda: cls(heap), seed, steps=steps, verify_every=9)
                st = lock.gc.stats
                self.assertGreater(st.collections + st.minor_collections, 0)

    def test_tracing_collectors_small_heap(self):
        for cls in TRACING:
            self.check(cls, 400)

    def test_tracing_collectors_medium_heap(self):
        for cls in TRACING:
            self.check(cls, 1000, seeds=10)

    def test_incremental_tiny_slices(self):
        for seed in range(25):
            fuzz(lambda: IncrementalMarkSweep(400, slice_words=4), seed, steps=900, verify_every=5)

    def test_generational_tiny_nursery(self):
        for seed in range(25):
            fuzz(lambda: Generational(600, nursery_words=64), seed, steps=900, verify_every=5)

    def test_refcount_is_safe(self):
        for seed in range(25):
            fuzz(lambda: RefCounting(1000), seed, steps=900, complete=False)


class CycleTest(unittest.TestCase):
    def make_ring_garbage(self, gc, rings=5):
        lock = Lockstep(gc)
        for _ in range(rings):
            lock.new(0, 1)
            lock.new(1, 1)
            lock.store(0, 0, 1)
            lock.store(1, 0, 0)
            lock.clear(0)
            lock.clear(1)
        return lock

    def test_tracing_collectors_reclaim_cycles(self):
        for cls in TRACING:
            with self.subTest(collector=cls.name):
                lock = self.make_ring_garbage(cls(400))
                lock.gc.collect()
                lock.verify(complete=True)
                self.assertEqual(list(lock.gc.objects()), [])

    def test_refcount_leaks_cycles(self):
        lock = self.make_ring_garbage(RefCounting(400))
        lock.gc.collect()
        lock.verify()
        with self.assertRaises(VerificationError):
            lock.verify(complete=True)
        self.assertEqual(len(list(lock.gc.objects())), 10)

    def test_refcount_frees_acyclic_garbage_immediately(self):
        gc = RefCounting(400)
        lock = Lockstep(gc)
        lock.new(0, 1)
        lock.new(1, 1)
        lock.store(0, 0, 1)
        lock.clear(1)
        self.assertEqual(len(list(gc.objects())), 2)
        lock.clear(0)  # whole chain dies at once, no collection needed
        self.assertEqual(list(gc.objects()), [])
        self.assertEqual(gc.stats.collections, 0)


class MovingTest(unittest.TestCase):
    def test_semispace_copies_breadth_first(self):
        gc = SemiSpace(400)
        lock = Lockstep(gc)
        a = lock.new(0, 2)
        lock.new(1, 1); b = lock.model.roots[1][1]
        lock.new(2, 0); c = lock.model.roots[2][1]
        lock.new(3, 0); d = lock.model.roots[3][1]
        lock.store(0, 0, 1)
        lock.store(0, 1, 2)
        lock.store(1, 0, 3)
        for r in (1, 2, 3):
            lock.clear(r)
        lock.new(4, 0)
        lock.clear(4)  # garbage between survivors
        gc.collect()
        lock.verify(complete=True)
        order = [gc.mem[x + 2] for x in gc.objects()]
        self.assertEqual(order, [a, b, c, d])
        self.assertEqual(gc.lo, 200)

    def test_mark_compact_leaves_no_holes(self):
        gc = MarkCompact(2000)
        lock = Lockstep(gc)
        rng = random.Random(7)
        WORKLOADS["fragment"](lock, rng, 1)
        gc.collect()
        lock.verify(complete=True)
        addrs = list(gc.objects())
        expected = 0
        for x in addrs:
            self.assertEqual(x, expected)
            expected += gc.mem[x]
        self.assertEqual(gc.top, expected)

    def test_mark_compact_preserves_order(self):
        gc = MarkCompact(300)
        lock = Lockstep(gc)
        oids = []
        for r in range(6):
            oids.append(lock.new(r, 1))
        for r in (1, 3):
            lock.clear(r)
        gc.collect()
        lock.verify(complete=True)
        self.assertEqual([gc.mem[x + 2] for x in gc.objects()], [oids[i] for i in (0, 2, 4, 5)])

    def test_stale_pointer_detected(self):
        gc = SemiSpace(200)
        Lockstep(gc).new(0, 1)
        stale = addr_of(gc.roots[0])
        gc.collect()
        with self.assertRaises(UseAfterFree):
            gc.read(stale, 0)

    def test_mark_sweep_fragments_where_compaction_does_not(self):
        ms, mc = MarkSweep(4000), MarkCompact(4000)
        for gc in (ms, mc):
            WORKLOADS["fragment"](Lockstep(gc), random.Random(1), 2)
            gc.collect()
        self.assertGreater(ms.fragmentation(), 0.2)
        self.assertEqual(mc.fragmentation(), 0.0)


class GenerationalTest(unittest.TestCase):
    def setUp(self):
        self.gc = Generational(1000, nursery_words=200)
        self.lock = Lockstep(self.gc)

    def promote_old(self, r):
        self.lock.new(r, 2)
        self.gc.minor()
        self.assertFalse(self.gc.in_nursery(addr_of(self.gc.roots[r])))

    def test_remembered_set_keeps_young_object_alive(self):
        self.promote_old(0)
        self.lock.new(1, 0)
        young = self.lock.model.roots[1][1]
        self.lock.store(0, 0, 1)
        self.lock.clear(1)
        self.assertIn(addr_of(self.gc.roots[0]), self.gc.remembered)
        self.gc.minor()
        self.lock.verify()
        self.assertIn(young, {self.gc.mem[a + 2] for a in self.gc.objects()})

    def test_without_barrier_the_young_object_would_be_lost(self):
        self.promote_old(0)
        self.lock.new(1, 0)
        self.lock.store(0, 0, 1)
        self.lock.clear(1)
        self.gc.remembered.clear()  # simulate a missing write barrier
        self.gc.minor()
        with self.assertRaises(VerificationError):
            self.lock.verify()

    def test_nepotism_until_major(self):
        self.promote_old(0)
        self.lock.new(1, 0)
        self.lock.store(0, 0, 1)
        self.lock.clear(1)
        self.lock.clear(0)  # old object is now dead, but it is in the remembered set
        self.gc.minor()
        self.lock.verify()
        self.assertEqual(len(list(self.gc.objects())), 2)  # both survive: nepotism
        self.gc.collect()
        self.lock.verify(complete=True)
        self.assertEqual(list(self.gc.objects()), [])

    def test_large_objects_are_pretenured(self):
        self.lock.new(0, 100)
        self.assertFalse(self.gc.in_nursery(addr_of(self.gc.roots[0])))


class IncrementalTest(unittest.TestCase):
    def lost_object_scenario(self, barrier):
        """Hide a white object behind a black one, then cut its last grey path."""
        gc = IncrementalMarkSweep(400, barrier=barrier)
        lock = Lockstep(gc)
        lock.new(0, 1)        # G: will stay grey
        lock.new(1, 1)        # X: will be scanned (black) first
        lock.new(2, 0)        # W: white, reachable only through G
        lock.store(0, 0, 2)
        lock.clear(2)
        gc.start_cycle()      # grey stack [G, X]; LIFO pops X first
        gc.step(1)            # scan exactly one object: X turns black
        lock.load(2, 0, 0)    # r2 = W
        lock.store(1, 0, 2)   # X.0 = W  (black -> white edge)
        lock.seti(3, 0)
        lock.store(0, 0, 3)   # G.0 = 0  (last grey path to W cut)
        lock.clear(2)         # W no longer in a register either
        gc.finish_cycle()
        return lock

    def test_dijkstra_barrier_prevents_lost_object(self):
        lock = self.lost_object_scenario(barrier=True)
        lock.verify()
        self.assertGreater(lock.gc.stats.barrier_hits, 0)

    def test_no_barrier_loses_object(self):
        lock = self.lost_object_scenario(barrier=False)
        with self.assertRaises(VerificationError):
            lock.verify()

    def test_allocate_black_floating_garbage(self):
        gc = IncrementalMarkSweep(400, slice_words=0)  # marking only advances when we say so
        lock = Lockstep(gc)
        lock.new(0, 1)
        gc.start_cycle()
        lock.new(1, 0)       # allocated black while marking
        lock.clear(1)        # dies immediately
        gc.finish_cycle()
        self.assertEqual(len(list(gc.objects())), 2)  # floats to next cycle
        gc.collect()
        lock.verify(complete=True)
        self.assertEqual(len(list(gc.objects())), 1)

    def test_pauses_are_bounded_by_slice(self):
        gc = IncrementalMarkSweep(4000, slice_words=32)
        lock = Lockstep(gc)
        WORKLOADS["trees"](lock, random.Random(2), 3)
        lock.verify()
        ms = MarkSweep(4000)
        WORKLOADS["trees"](Lockstep(ms), random.Random(2), 3)
        self.assertLess(gc.stats.max_pause, ms.stats.max_pause)


class ApiTest(unittest.TestCase):
    def test_out_of_memory(self):
        for cls in [*TRACING, RefCounting]:
            with self.subTest(collector=cls.name):
                gc = cls(200)
                lock = Lockstep(gc)
                with self.assertRaises(OutOfMemory):
                    for r in range(8):
                        lock.new(r, 30)

    def test_use_after_free_detected(self):
        gc = MarkSweep(200)
        lock = Lockstep(gc)
        lock.new(0, 1)
        a = addr_of(gc.roots[0])
        lock.clear(0)
        gc.collect()
        self.assertTrue(gc.mem[a + INFO] & FREE)
        with self.assertRaises(UseAfterFree):
            gc.read(a, 0)

    def test_field_bounds(self):
        gc = MarkSweep(200)
        Lockstep(gc).new(0, 2)
        with self.assertRaises(IndexError):
            gc.read(addr_of(gc.roots[0]), 2)

    def test_every_workload_on_every_collector(self):
        for wname, work in WORKLOADS.items():
            for name, cls in COLLECTORS.items():
                if name == "refcount" and wname == "cycles":
                    continue
                with self.subTest(workload=wname, collector=name):
                    gc = cls(8000, num_roots=16)
                    lock = Lockstep(gc)
                    work(lock, random.Random(3), 2)
                    lock.verify()
                    gc.collect()
                    lock.verify(complete=(name != "refcount"))

    def test_header_size_constant(self):
        self.assertEqual(HEADER, 4)


if __name__ == "__main__":
    unittest.main()

# gclab

Six garbage collectors written from scratch over one simulated, word addressed heap, plus a verifier that proves after every collection that none of them lost, corrupted or duplicated a single object.

Garbage collection is usually explained with box and arrow diagrams and then hidden inside a runtime where you can never watch it work. gclab puts the classic algorithms side by side on the same heap layout, drives them with the same mutator programs, and measures them with the same counters, so the textbook tradeoffs (copying vs compacting, pause time vs throughput, why generational collectors need write barriers, why reference counting leaks cycles) show up as numbers you can reproduce in under a second.

Pure Python 3.9+, no dependencies.

## The collectors

* **`marksweep`** (`marksweep.py`): McCarthy 1960. Depth first mark with an explicit stack, one sweep pass that frees and coalesces. Address ordered first fit free list. Never moves objects, so it fragments.
* **`markcompact`** (`markcompact.py`): Lisp 2 sliding compaction: compute forwarding addresses, rewrite every reference, slide survivors down. Bump allocation, zero fragmentation, allocation order preserved.
* **`semispace`** (`semispace.py`): Cheney 1970. Copy live objects into the other half; to space doubles as the BFS queue. Work proportional to live data only, at the price of half the heap.
* **`generational`** (`generational.py`): Bump allocated nursery collected by copying, promoted into a mark sweep old space. Remembered set fed by a write barrier, large objects pretenured.
* **`incremental`** (`incremental.py`): Tri colour incremental marking in bounded slices, Dijkstra insertion barrier, allocate black, short stop the world root rescan at mark termination.
* **`refcount`** (`refcount.py`): Naive reference counting as the baseline. Frees garbage instantly, never reclaims cycles.

## The heap

Memory is a flat list of integers. Every block, live or free, starts with a 4 word header `[size, info, oid, aux]` so the heap can always be parsed with a linear walk. Field values are tagged words: `0` is null, `(addr << 2) | 1` is a pointer, `(n << 2) | 2` is an integer. Tagging is what lets a precise collector tell pointers from data without type maps, the same trick OCaml and V8 Smis use.

The `aux` word does different jobs per collector: forwarding address in the copying and compacting collectors, reference count in `refcount`. The `oid` word is a stable identity that exists only so the verifier can follow objects as they move; no collector reads it.

## The mutator and the verifier

The mutator is a register machine with six instructions (`new`, `load`, `store`, `seti`, `mov`, `clear`). Its registers are the root set and they belong to the collector, which is how a moving collector can relocate objects behind the program's back.

`Lockstep` runs every instruction twice: once on the real heap and once on a `Model` that stores objects in plain dicts keyed by id, with no addresses and no GC. `verify()` then checks that the heap is isomorphic to the model's reachable graph:

* every root decodes to the object the model expects,
* every reachable object exists, has the right number of fields, and every field points at the right object or holds the right integer,
* no pointer dangles, no object exists twice,
* and after `collect()` (`complete=True`) not one unreachable object survived.

That single check catches every class of GC bug I know of: missed root updates, a forgotten field during forwarding, a stale pointer to from space, a premature free, a broken barrier.

## Running it

```bash
git clone https://github.com/MohammadHossinzehi/gclab
cd gclab

python -m unittest                 # 30 tests, about 2 seconds
python -m gclab fuzz --seeds 50    # random programs on all six collectors, verified
python -m gclab bench              # compare collectors on five workloads
python -m gclab heapmap            # see fragmentation with your own eyes
```

`pip install .` also gives you a `gclab` command. Every `bench` run is itself a correctness check: the workload runs in lockstep with the model and is verified at the end.

## What the numbers show

`GC work` counts heap words the collector touched (traced, copied or swept). `max pause` is the most work done in a single stop. Excerpt from `python -m gclab bench` (4000 word heap):

```
== trees: one long lived tree plus many short lived ones (GCBench in miniature)
collector     status       GCs  minor   GC work  max pause   copied  barrier   frag
marksweep     ok            11      0     50234       4954        0        0   0.44
markcompact   ok            11      0     75132       7992     4056        0   0.00
semispace     OOM            3      0     10500       3996     5250        0   0.00
generational  ok            11     25     64552       4166    13092       75   0.00
incremental   ok            26      0    103098       2044        0     1081   0.44
refcount      ok             0      0     23832        762        0        0   0.31

== cycles: rings of 2 to 8 objects that die together
marksweep     ok             4      0     10732       2698        0        0   0.00
semispace     ok             9      0       288         60      144        0   0.00
generational  ok             0     18      1032        162      480       13   0.00
refcount      OOM            1      0      2668       2668        0        0   0.00
```

A few things worth noticing:

* **Copying cost tracks live data, not heap size.** On `cycles` almost everything is garbage, and `semispace` does 288 words of work where `marksweep` does 10732, because Cheney never looks at dead objects.
* **That speed costs half the heap.** On `trees` the live set no longer fits in one semispace and it runs out of memory where every other collector survives.
* **Incremental marking trades throughput for latency.** Its worst pause is less than half of `marksweep`'s, but it does about twice the total work: allocating black and rescanning roots keep floating garbage alive, so cycles run more often.
* **The generational hypothesis is a hypothesis.** `trees` builds medium lived structures, which get promoted and then die in old space, so `generational` copies a lot for nothing. On `churn`, where nearly everything dies young, it is the cheapest collector of all by a wide margin.
* **Reference counting never pauses for long but never finds cycles.** It leaks every ring on `cycles` until it runs out of memory.

`heapmap` shows why compaction exists. After the `fragment` workload:

```
marksweep     |:::::::......................................::.:::.::::.::::...:|  frag 0.36
markcompact   |####:............................................................|  frag 0.00
```

## Tests

`tests/test_gclab.py` mixes property style fuzzing with targeted scenarios:

* **Lockstep fuzzing** of all six collectors over hundreds of random programs, on tiny heaps (so collections happen every few dozen instructions), with verification every few steps, including mid way through incremental mark cycles.
* **The lost object problem, reproduced.** A deterministic scenario hides a white object behind a black one and then cuts its last grey path. With the Dijkstra barrier the verifier passes; with `barrier=False` the object is freed while still reachable and the verifier reports the dangling pointer.
* **Remembered set and nepotism.** An old to young pointer keeps the young object alive across a minor collection; clearing the remembered set by hand (simulating a missing barrier) loses it; a dead old object keeps its young referent alive until the next major collection.
* **Collector specific invariants:** Cheney copies in breadth first order, Lisp 2 leaves no holes and preserves allocation order, mark sweep fragments where compaction does not, stale from space pointers and use after free are detected, reference counting frees acyclic garbage immediately and leaks cycles.
* **Floating garbage:** an object allocated black during a cycle survives that cycle and is reclaimed by the next.

## Design decisions

* **One heap format for everything.** Sharing the header layout and the tagged words means every collector can be checked by the same verifier and measured by the same counters, which is what makes the comparison fair.
* **Work counts instead of wall clock.** Python timing would mostly measure the interpreter. Counting words touched is deterministic, reproducible, and closer to what dominates in a real collector (memory traffic).
* **Explicit stacks and queues everywhere.** No recursion in any collector, so a 100k element linked list does not hit Python's recursion limit.
* **Registers are not barriered.** That mirrors real VMs, where barriering every stack write would be too slow. The incremental collector pays for it with a short root rescan at mark termination, and the generational collector treats registers as roots on every minor collection.
* **En masse promotion.** The nursery promotes every survivor on its first collection instead of aging objects in survivor spaces. It keeps the code readable and makes the cost of violating the generational hypothesis easy to see in `trees`.
* **Sweeping is not incremental.** Marking is sliced; sweeping is one linear pass over block headers. Lazy sweeping is the obvious next step and the reason the incremental collector's worst pause is not smaller still.

## Layout

```
gclab/
  heap.py          tagged words, header layout, free list space, Collector base, marking
  marksweep.py     markcompact.py   semispace.py
  generational.py  incremental.py   refcount.py
  mutator.py       Mutator, reference Model, verify(), Lockstep, fuzz()
  workloads.py     churn, trees, cache, cycles, fragment
  __main__.py      bench, fuzz, heapmap
tests/test_gclab.py
```

## References

* Jones, Hosking, Moss, *The Garbage Collection Handbook*, 2nd ed.
* C. J. Cheney, *A Nonrecursive List Compacting Algorithm*, CACM 1970.
* Dijkstra, Lamport et al., *On the Fly Garbage Collection: An Exercise in Cooperation*, CACM 1978.
* Ungar, *Generation Scavenging*, 1984.

License: MIT.

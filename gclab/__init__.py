"""gclab: six memory managers over one simulated heap, checked against a reference model."""
from .generational import Generational
from .heap import Collector, OutOfMemory, Stats, UseAfterFree
from .incremental import IncrementalMarkSweep
from .markcompact import MarkCompact
from .marksweep import MarkSweep
from .mutator import Lockstep, Model, Mutator, VerificationError, fuzz, verify
from .refcount import RefCounting
from .semispace import SemiSpace

COLLECTORS = {
    cls.name: cls
    for cls in (MarkSweep, MarkCompact, SemiSpace, Generational, IncrementalMarkSweep, RefCounting)
}

__all__ = [
    "COLLECTORS", "Collector", "Generational", "IncrementalMarkSweep", "Lockstep",
    "MarkCompact", "MarkSweep", "Model", "Mutator", "OutOfMemory", "RefCounting",
    "SemiSpace", "Stats", "UseAfterFree", "VerificationError", "fuzz", "verify",
]
__version__ = "1.0.0"

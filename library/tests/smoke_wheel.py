"""Check that an installed wheel uses its packaged C++ library."""

from pathlib import Path

import numpy as np

import adaptive_cs as cs
from adaptive_cs._cpp import _find_library


library = _find_library()
assert library is not None
assert library.parent == Path(cs.__file__).resolve().parent

tracker = cs.ClassSelection(alpha=0.05, max_samples=3)
for _ in range(3):
    tracker.update(np.array([0.9, 0.1]))
assert tracker.done
assert int(tracker.n_samples[0]) == 3

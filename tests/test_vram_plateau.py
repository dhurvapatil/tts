"""Tests for the VRAM plateau regression (issue 002-004).

`compute_vram_plateau(per_run_vram_mb)` returns a small stats object
classifying whether the per-run peak VRAM has plateaued.  The plateau
detection is the gate for `--assert-vram-plateau-under-mb K` in
bench_ttfw: an open question from PRD-002 (is the 1236->1538 MB growth
between two af_nicole runs a leak or arena caching?) becomes a
CI-asserted answer.

Pure unit tests.  No GPU, no model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# bench_ttfw.py is a script, but it can be imported as a module since
# it has no __name__ == "__main__" guard at the top of module scope.
import bench_ttfw  # noqa: E402


def test_empty_list_returns_sensible_default():
    s = bench_ttfw.compute_vram_plateau([])
    assert s.first is None
    assert s.last is None
    assert s.delta == 0
    # An empty series trivially counts as "no leak visible" (vacuously).
    assert s.is_plateau is True


def test_single_element_has_zero_delta_and_is_plateau():
    s = bench_ttfw.compute_vram_plateau([1234])
    assert s.first == 1234
    assert s.last == 1234
    assert s.delta == 0
    assert s.is_plateau is True


def test_plateau_series_passes():
    """A series whose delta is well within the default threshold
    (64 MB) must be classified as a plateau."""
    s = bench_ttfw.compute_vram_plateau([1000, 1005, 1003, 1002, 1004])
    assert s.first == 1000
    assert s.last == 1004
    assert s.delta == 4
    assert s.is_plateau is True


def test_growing_series_fails():
    """A series whose delta exceeds the default threshold is a leak
    (or at least a non-plateau) and must be flagged."""
    s = bench_ttfw.compute_vram_plateau([1000, 1200, 1400, 1600])
    assert s.first == 1000
    assert s.last == 1600
    assert s.delta == 600
    assert s.is_plateau is False


def test_custom_threshold_changes_verdict():
    """A growing series below the custom threshold should pass."""
    s = bench_ttfw.compute_vram_plateau(
        [1000, 1100, 1200, 1300], threshold_mb=500
    )
    # delta = 300, threshold = 500
    assert s.delta == 300
    assert s.is_plateau is True


def test_absolute_delta_handles_shrinkage():
    """The plateau check uses ABSOLUTE delta.  A hypothetical
    negative delta (VRAM going down) is also caught as a non-plateau
    if its magnitude exceeds the threshold (it would indicate some
    other bug, e.g. a stale sample)."""
    s = bench_ttfw.compute_vram_plateau([2000, 1500, 1000, 500])
    # delta = 500 - 2000 = -1500, abs(delta) = 1500
    assert s.delta == -1500
    assert s.is_plateau is False


def test_returns_plateau_stats_dataclass():
    """Return value must expose the four documented fields."""
    s = bench_ttfw.compute_vram_plateau([1000, 1050, 1100])
    for field in ("first", "last", "delta", "is_plateau"):
        assert hasattr(s, field), f"PlateauStats must expose {field!r}"


if __name__ == "__main__":
    import traceback
    fns = [(n, g) for n, g in list(globals().items()) if n.startswith("test_") and callable(g)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"  ✓ {n}")
        except Exception:
            failed += 1
            print(f"  ✗ {n}")
            traceback.print_exc()
    if failed:
        sys.exit(1)
    print(f"\n  {len(fns)} passed")

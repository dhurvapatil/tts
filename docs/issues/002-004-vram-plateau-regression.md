---
id: 002-004
title: VRAM plateau regression in bench_ttfw
triage: ready-for-agent
parent: PRD-002-warmup-and-buffering
blocked_by: none
---

## What to build

Live runs show VRAM growing between consecutive same-shape
generations (1236 → 1538 MB observed across two af_nicole runs in
PRD-002's problem statement). It's unclear whether this is a real
leak (bad) or expected cuDNN workspace / ORT arena caching per
shape (benign, plateaus). Today there's no way to answer that
question from a single CLI run.

Add a `--vram-plateau-runs N` mode to `bench_ttfw.py` that runs
the same paragraph N times and reports the peak VRAM per run plus
the delta between the last and first runs. Add a companion
`--assert-vram-plateau-under-mb K` that exits 1 if the delta
exceeds K. Default K = 64 MB, which comfortably absorbs cuDNN
workspace caching but would catch a real leak (typically +5-50
MB/iteration in the wild).

Also add a small pure helper `compute_vram_plateau(per_run_vram_mb: list[int]) -> PlateauStats`
that encapsulates the plateau-detection logic, so it can be
unit-tested without running the model.

End-to-end: `uv run python bench_ttfw.py --vram-plateau-runs 5
--assert-vram-plateau-under-mb 64` runs the standard paragraph
5 times, prints a small VRAM table, and exits 0 if the
last-vs-first delta is ≤ 64 MB, 1 otherwise. This turns the
open question from PRD-002 into a CI-asserted answer.

## Acceptance criteria

- [ ] A new pure function `compute_vram_plateau(per_run_vram_mb: list[int]) -> PlateauStats`
  exists in `bench_ttfw.py` (or a small helper module). The
  return type is a simple dataclass / namedtuple with at least
  `first`, `last`, `delta`, and `is_plateau: bool` (where
  `is_plateau` is True iff `|delta| <= threshold` and threshold
  is a function arg with a sensible default).
- [ ] `bench_ttfw.py` gains two CLI flags:
  `--vram-plateau-runs N` (default 0 = disabled) and
  `--assert-vram-plateau-under-mb K` (default 64).
- [ ] When `--vram-plateau-runs N` is set with N >= 2, the bench
  runs the standard paragraph N times (in the same session, same
  model, same voice), records peak VRAM per run via the existing
  `ResourceMonitor.peak_gpu_mem`, and prints a small table:
  `vram_plateau: [1236, 1380, 1450, 1490, 1490] MB  delta: +254 MB`.
- [ ] When `--assert-vram-plateau-under-mb K` is also set, the
  bench exits 1 if the delta exceeds K. The exit-code behavior
  matches the existing `--assert-ttfw-under-ms` pattern.
- [ ] A new test file `tests/test_vram_plateau.py` covers:
    - Empty list → returns a sensible empty/default `PlateauStats`
    - Single-element list → delta is 0, `is_plateau` is True
    - Plateau list `[1000, 1005, 1003, 1002]` → `is_plateau=True`
      with default threshold
    - Growing list `[1000, 1200, 1400, 1600]` → `is_plateau=False`
    - The plateau detection uses the absolute delta (growing OR
      shrinking) so that a hypothetical negative delta (which
      would be a different bug) is also caught
- [ ] All existing tests still pass.
- [ ] Manually verified: `uv run python bench_ttfw.py --vram-plateau-runs 3`
  produces the plateau report and exits 0 on this hardware
  (assuming the observed growth is within the threshold).

## Blocked by

None - can start immediately.

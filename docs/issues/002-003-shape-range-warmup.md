---
id: 002-003
title: Shape-range warmup to eliminate first-chunk cold-compile spike
triage: ready-for-agent
parent: PRD-002-warmup-and-buffering
blocked_by: none
---

## What to build

The current warmup uses two *fixed* texts (a 42-phoneme lead-in
and a 226-phoneme long shape) that leave a large shape gap. Real
lead-ins phonemize to unpredictable counts (30, 55, 80...) and
the first prompt's chunk-0 infer then triggers an on-demand cuDNN
kernel compile, producing spikes like 377ms for a 6-word af_nicole
chunk (worse than af_bella's 270ms for a 15-word chunk).

Replace the two fixed-text warmups with a new function
`run_warmups(sess, kokoro, target_phoneme_counts: list[int]) -> list[tuple[int, float]]`
that phonemizes a single long source text once, then truncates
and pads the resulting phoneme string to each target count and
runs `kokoro.create` on it. The default shape set is
`[20, 40, 60, 100, 160, 256]` phonemes, which covers short
command-style leads up through typical 25-word chunks.

End-to-end: user runs `uv run tts.py`, sees a single new banner
line `warmed 6 shapes (20, 40, 60, 100, 160, 256 phonemes) in T ms`,
and chunk-0 infer is now stable across any voice/text (no 377ms
spike for af_nicole-style shapes).

A new `--warmup-shapes "20,40,60,100,160,256"` CLI flag overrides
the default. `--no-warm` still skips all warmups and wins over
`--warmup-shapes`.

## Acceptance criteria

- [ ] A new pure function `run_warmups(sess, kokoro, target_phoneme_counts)`
  exists in `tts.py` near the existing warmup code. The signature
  is stable and matches the PRD. The function returns a list of
  `(phoneme_count, ms)` pairs in input order.
- [ ] The function phonemizes a single long source text once,
  then truncates/pads that phoneme string to each target count
  and runs inference. Truncation is just `[:target_count]`;
  padding repeats the last phoneme token. Audio is discarded.
- [ ] The two existing fixed-text warmup blocks (`warmup_short`,
  `warmup_lead`, `warmup_long` with their hard-coded strings) are
  removed from `main()` and replaced by a single call to
  `run_warmups(...)` with the default shape set
  `[20, 40, 60, 100, 160, 256]`.
- [ ] The startup banner changes to a single line:
  `warmed 6 shapes (20, 40, 60, 100, 160, 256 phonemes) in T ms`.
- [ ] CLI flag `--warmup-shapes "..."` (comma-separated ints) is
  parsed and overrides the default. Empty string disables shape
  warmups (no shape warmup calls are made, but the other warmups
  remain).
- [ ] `--no-warm` continues to skip ALL warmups and overrides
  `--warmup-shapes`.
- [ ] A new test file `tests/test_warmup_shapes.py` covers:
    - The function calls `kokoro.create` exactly once per target
      count, in order (use a stub kokoro that records the phoneme
      length of each call)
    - Truncation and padding are correct (the phoneme-length of
      each call matches the target count)
    - When `target_phoneme_counts` is empty, no calls are made
    - The function returns one `(count, ms)` pair per input count
- [ ] All existing tests still pass.
- [ ] `uv run python bench_ttfw.py --runs 3` produces a
  chunk-0_infer that is within ±20% of the median per-chunk infer
  time (i.e. the first chunk is no longer the slowest).

## Blocked by

None - can start immediately.

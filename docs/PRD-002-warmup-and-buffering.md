# PRD-002: Hardening, Shape-Range Warmup, and Pre-Generate Buffer

Status: ready-for-agent
Parent: PRD-001-first-word-latency.md (shipped)
Targets: `uv run tts.py` interactive TTFW on CUDA + streaming

## Problem Statement

PRD-001 collapsed TTFW from 775ms to ~228ms median on CUDA, but live
runs on different voices and texts surfaced a new class of problems
underneath the now-clean overhead breakdown:

1. **First-chunk cold-compile spike hits unpredictable shapes.** The
   current warmup uses two *fixed* texts: a 6-8 word lead-in
   ("the system is ready to start now please" → 42 phonemes) and a
   ~25-word long shape (226 phonemes). Between 42 and 226 there is
   a large gap. A real lead-in like "To get started, please check
   that your device is securely plugged into a power source" may
   phonemize to ~30 phonemes, or 55, or 80, depending on the words.
   The first prompt's chunk-0 infer then triggers an on-demand cuDNN
   kernel compile, which on af_nicole produced 377ms for a *6-word*
   chunk (worse than af_bella's 270ms for a *15-word* chunk). This
   is voice-independent — it is a shape-compile problem.

2. **Confusing user-facing metric.** The top-of-prompt summary line
   still prints `RTF 1.01 · slower than realtime ✗` using wall-RTF
   with the old verdict, while the Latency section immediately below
   prints `RTF (generation) 0.06x ✓` with the corrected verdict. The
   user sees `✗` first and `✓` two lines later, which is misleading
   and undoes the careful messaging of PRD-001 Issue 2.

3. **VRAM grows between consecutive same-shape generations** (1236 →
   1538 MB observed across two af_nicole runs). The user can't tell
   whether this is a leak (bad) or arena caching per cuDNN algo
   candidate (benign, plateaus). Without a regression assertion,
   the question stays open every time it happens.

4. **A deprecation warning pollutes every prompt.** `pynvml` was
   deprecated; the actual installed package is `nvidia-ml-py`
   (already in the lock file from `onnxruntime-gpu[cuda,cudnn]`).
   We `import pynvml` which emits `FutureWarning: The pynvml package
   is deprecated...` once per run.

5. **The 17× generation slack is unused.** With generation at
   0.06× realtime and playback at 1.0×, the worker finishes all
   chunks in ~1.4s and then sits idle for ~22s while the speaker
   drains. We still back-pressure the worker on playback
   (`QUEUE_MAXSIZE = 4` + blocking writes). This is the right
   architecture for a system that's *trying* to keep up, but ours
   doesn't need to try — it always finishes. Embracing this turns
   the architecture from "streaming to keep up" into "streaming
   because the speaker is slow," and unlocks the ability to
   detect hiccups and to compute better crossfades (we have the
   whole prompt's audio in hand by the time chunk 0 starts
   playing).

## Solution

Five changes, all building on PRD-001's foundation. None of them
change the CLI's external shape beyond new flags. All have a
regression assertion in `bench_ttfw.py`.

1. **Shape-range warmup.** Replace the two fixed-string warmups
   with a function `run_warmups(sess, kokoro, target_phoneme_counts)`
   that phonemizes a long source text, then *truncates and pads* it
   to each of a small set of representative phoneme counts (e.g.
   {20, 40, 60, 100, 160, 256}) and runs inference at each. This
   pre-compiles cuDNN kernels for every shape a real lead-in or
   chunk is likely to hit. Cost: a few hundred extra ms at startup.
   Benefit: chunk-0 infer no longer spikes on any voice/text.

2. **WAV summary line uses generation-RTF.** The top-of-prompt line
   prints `RTF (gen) 0.06x ✓` (or just drops the verdict and shows
   both numbers) so the messaging is consistent with the Latency
   section. Implementation: call the existing
   `compute_generation_rtf()` and reuse the verdict logic.

3. **VRAM plateau regression.** Add a `--vram-plateau-runs N` mode
   to `bench_ttfw.py` that runs the same paragraph N times and
   asserts VRAM does not grow more than `K` MB between the last
   two runs (with a generous K to absorb cuDNN workspace caching).
   Default K=64. This turns the open question into a CI-asserted
   answer.

4. **Switch `pynvml` → `nvidia-ml-py`.** `nvidia-ml-py` is already
   in the lock file as a transitive dep; we just import it under
   the same name and drop `pynvml` from `pyproject.toml`. One-line
   change, no behavior change.

5. **Pre-generate-whole-prompt + buffer health.** The streaming
   worker submits ALL chunks up front (the worker's `submit()` calls
   are already in a loop), and `play_streaming` waits for the *first*
   chunk to start writing — the same TTFW as before. The new behavior
   is: the report surfaces a `buffer_health` metric (chunks queued
   ahead of playback at peak), a `hiccup_count` (any chunk whose
   infer took > 1× its audio duration — a stall), and a `playback_underrun_count`
   (any moment the audio device was drained before the next chunk
   arrived). The architecture now has a name: "look-ahead buffer."
   This is the foundation for future concurrency work without
   requiring it in this PRD.

## User Stories

### Warmup
1. As a CLI user, I want the first chunk of any voice/text to be
   served from pre-compiled kernels, so that TTFW is stable (no
   377ms first-call spike for af_nicole-style shapes).
2. As a CLI user, I want the warmup to cover a range of shapes,
   not just one fixed text, so that changing the lead-in size or
   the input text doesn't reintroduce the cold-compile cost.
3. As a CLI user, I want the warmup shape coverage to be tunable
   via a `--warmup-shapes` flag, so I can add shapes for my domain
   (e.g. very short leads for command-prompt UIs).
4. As a CLI user, I want `--no-warm` to still skip ALL warmups (it
   currently skips them all, which is preserved).
5. As a developer, I want the warmup function to be unit-testable
   in isolation (it accepts a list of phoneme counts, runs them,
   returns per-shape timing) so I can assert each shape was
   exercised.

### Output messaging
6. As a CLI user, I want the top-of-prompt summary line to be
   consistent with the Latency section — either both show
   generation-RTF, or both show both metrics clearly labeled.
7. As a CLI user, I want to immediately understand from the
   first printed line whether the *model* (not the speaker) is
   faster than realtime.

### VRAM
8. As a CLI user, I want to know whether VRAM growth between
   runs is a leak or expected caching, via a bench mode that
   runs N times and reports the plateau.
9. As a developer, I want `bench_ttfw.py --vram-plateau-runs 5`
   to exit 1 if VRAM doesn't plateau, so CI catches a real leak.
10. As a CLI user, I want an optional `--vram-cap-mb N` flag that
    sets `gpu_mem_limit` in the CUDA provider options, so I can
    constrain VRAM on shared machines.

### Deprecation
11. As a CLI user, I want no deprecation warning on startup, so
    the output is clean and the warning doesn't make me think
    something is broken.

### Buffering
12. As a CLI user, I want the post-run report to show
    `buffer_health: N chunks ahead of playback` (peak, median,
    final), so I can see how much slack the system has.
13. As a CLI user, I want the report to count and surface
    `hiccups: N events` (any chunk whose infer took longer
    than its audio duration), so I can detect thermal throttling
    or background load.
14. As a CLI user, I want the report to count and surface
    `playback_underruns: N events` (any moment the audio device
    drained before the next chunk arrived), so I can tell if
    generation is genuinely failing to keep up.
15. As a developer, I want the buffer-health data captured
    during the run (not derived after the fact) so the report
    is accurate even when the audio is fully drained by the
    time the worker finishes.
16. As a CLI user, I want the buffering change to be opt-in via
    a `--pre-generate` flag (default ON for streaming mode),
    with `--no-pre-generate` preserving today's "submit chunks
    one-by-one" behavior for back-compat.

### Architecture
17. As a developer, I want the pre-generate behavior to be
    encapsulated in a small `LookAheadBuffer` class (or
    equivalent) so the streaming path is decoupled from the
    scheduling decision and can be swapped for a concurrent
    scheduler in a future PRD.
18. As a developer, I want the new metrics to be in the
    `emit_profile_line` JSON output so `--profile` captures
    them in machine-readable form for diffs across runs.
19. As a developer, I want the warmup function to return
    per-shape timings so the startup banner can show "warmed 6
    shapes (20, 40, 60, 100, 160, 256 phonemes)" instead of the
    current two-line report.

## Implementation Decisions

### Warmup

- **New function** `run_warmups(sess, kokoro, target_phoneme_counts: list[int]) -> list[tuple[int, float]]`
  with a stable public contract: input is the list of phoneme
  counts to pre-compile, output is a list of `(phoneme_count, ms)`
  pairs (one per shape). The function picks a single long source
  text (e.g. the 226-phoneme text from the existing long warmup),
  phonemizes it once, then truncates/pads that phoneme string to
  each target count and runs `kokoro.create` on it. Truncation is
  always safe (model handles up to MAX_PHONEME_LENGTH = 510);
  padding to a target is by repeating the last phoneme token to
  reach the count, which produces *some* audio but the only
  purpose is to force the kernel to compile for that shape. The
  audio is discarded.
- **Default shape set** is `[20, 40, 60, 100, 160, 256]`. This
  covers: short command-style leads (20), typical 6-word leads
  (40), the 6-8 word leads the existing warmup uses (40-60),
  typical 15-word chunks (100), the existing long-warmup shape
  (160), and an upper bound above the typical chunk size (256).
- **CLI flag** `--warmup-shapes "20,40,60,100,160,256"` overrides
  the default. Empty string means "no shape warmups at all" (a
  faster startup for users who don't care about first-call cost).
- **`--no-warm`** still skips all warmups. The two flags compose:
  `--no-warm` wins.
- **Existing single-text warmups** (`warmup_short`, `warmup_lead`)
  are removed. The new shape range subsumes them. The startup
  banner changes from three lines to one: `warmed N shapes
  (20, 40, 60, 100, 160, 256 phonemes) in T ms`.

### Output messaging

- The top-of-prompt summary line uses `compute_generation_rtf`
  (already exists) instead of `time_total / duration_s` (wall).
- Verdict applies to generation-RTF, matching the Latency section.
- Line shape: `✓ af_nicole_...wav  (23.82s audio @ 24000 Hz,  RTF (gen) 0.06x ✓)`
  (when generation is faster than realtime). When it's not (rare;
  only on very old CPUs or extreme background load), the verdict
  flips to `✗` with a clear label.
- The optional "RTF (wall) 1.01x" can be shown on the same line
  for the audio-sink-aware user, but the headline number is gen.

### VRAM plateau

- `bench_ttfw.py` gains `--vram-plateau-runs N` (default 0 =
  disabled). When N >= 2, it runs the paragraph N times and
  reports `vram_plateau: <last> MB  delta: <last - first> MB  over N runs`.
- Adds `--assert-vram-plateau-under-mb K` for CI: exit 1 if
  the delta exceeds K. Default K = 64 MB, which comfortably
  absorbs cuDNN workspace caching but would catch a real leak
  (typically +5-50 MB/iteration).
- The plateau assertion reuses the existing
  `ResourceMonitor.peak_gpu_mem` field; no new code paths.

### VRAM cap

- New provider option: `gpu_mem_limit: <int>` (in bytes; ORT
  accepts a number). When `--vram-cap-mb N` is passed, set
  `gpu_mem_limit = N * 1024 * 1024` in the CUDA provider options
  dict. Default = unlimited (current behavior).
- If ORT raises on a shape that exceeds the cap, the existing
  per-chunk error handling in `play_streaming` catches it and
  marks the chunk as failed; the run continues with the remaining
  chunks.

### pynvml → nvidia-ml-py

- Single-line change: `import pynvml` → `import nvidia_ml_py as pynvml`.
  Same API surface (`nvmlInit`, `nvmlDeviceGetHandleByIndex`,
  `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetUtilizationRates`).
- Drop `pynvml` from `pyproject.toml` `dependencies`. `nvidia-ml-py`
  is already in the lock file as a transitive dep of
  `onnxruntime-gpu[cuda,cudnn]`; we add it as an explicit
  dependency to make the import direct.
- The `FutureWarning` disappears from startup output.

### Look-ahead buffer

- **Class** `LookAheadBuffer` with a small public surface:
  - `submit_all(jobs: list[Job])` — schedule every chunk up front
  - `drain_to_sink(sink: AudioSink) -> Stats` — wait for first
    chunk, start writing to sink, then drain the rest with
    crossfade
  - `health() -> BufferHealth` — peak/median/final queue depth
    and any hiccup/underrun events
- **Default behavior**: in streaming mode, the main loop calls
  `buffer.submit_all(jobs)` immediately after chunking, then
  `buffer.drain_to_sink(session_sink)`. The `WorkerPool` is
  unchanged; the buffer is a *thin coordinator* on top of it.
- **`--no-pre-generate`** preserves the current behavior (submit
  chunks one-by-one as the consumer reads) for back-compat with
  any tooling that depends on the visible submission order.
- **Buffer health tracking** lives on the buffer, not the
  consumer. A background thread (or the consumer's loop) records
  queue depth at each `q.get()` and the wall-clock time spent
  waiting. A hiccup is a single chunk whose infer exceeded its
  own audio duration. An underrun is a moment the consumer
  called `q.get()` and found it empty after the warmup phase
  (i.e., generation fell behind playback, not the initial
  chunk-0 wait).
- These metrics are added to the stats dict and to
  `emit_profile_line` output as new fields.

### Cross-cutting

- **No model changes.** All five items are host-side.
- **Public surface**: the only new CLI flags are
  `--warmup-shapes`, `--pre-generate` / `--no-pre-generate`,
  `--vram-cap-mb`, and `--vram-plateau-runs` /
  `--assert-vram-plateau-under-mb` on the bench.
- **No breaking changes**: every existing flag and behavior
  (including the no-stream path, the winsound fallback, and the
  per-prompt output format) is preserved. The pre-generate
  default is ON, but the old per-chunk submission still works
  behind `--no-pre-generate`.

## Testing Decisions

A good test in this codebase measures observable behavior (TTFW
in ms, VRAM in MB, output strings the user sees) through the
public CLI or bench surface. No internal call counts, no
mocking of ORT internals.

### What gets tested

- **Warmup coverage** (`tests/test_warmup_shapes.py`): pure unit
  test of `run_warmups`. Pass a stub `kokoro` that records the
  phoneme count of each call; assert the function was invoked
  once per target count, in order, and that the returned
  timings list has the right shape. The test does NOT use the
  real model — a stub is sufficient and is what the existing
  tests already do for chunker / sink / pool.

- **WAV summary line uses generation-RTF** (`tests/test_summary_line.py`):
  pass a fake stats dict with `time_total=24.14, audio_duration_s=23.82`
  (the values from the live af_nicole run). Call
  `format_summary_line(stats)`. Assert the returned string
  contains `RTF (gen)` and `0.06x` and does NOT contain
  `slower than realtime`.

- **VRAM plateau regression** (`tests/test_vram_plateau.py`):
  pure unit test on the plateau-detection helper. Feed a list
  of (run_idx, vram_mb) tuples; assert the function correctly
  reports "plateau" vs "growing" and the delta is computed
  right. The actual multi-run VRAM measurement is a bench
  concern, not a unit concern.

- **`--no-warm` still works** (regression test in
  `test_warmup_shapes.py`): the function returns an empty list
  when `--no-warm` is set, and no kernel calls are made.

- **`nvidia_ml_py` import works** (`tests/test_provider_options.py`
  extension): assert that `_nvidia_query` can be called and
  returns a `(mem, util)` pair when the system has a GPU, and
  `(None, None)` when it doesn't. This is a smoke test, not
  a behavior test.

- **Look-ahead buffer** (`tests/test_look_ahead_buffer.py`):
  pure unit test with a stub `WorkerPool` and a stub
  `AudioSink`. Submit a list of 4 fake jobs. Assert that
  `submit_all` schedules all 4, `drain_to_sink` writes 4
  chunks to the sink in order, and `health()` reports
  peak_queue_depth >= 3 (the look-ahead is real).

- **End-to-end**: the existing `bench_ttfw.py` continues to
  serve as the regression harness. New flags
  (`--vram-plateau-runs`, `--assert-vram-plateau-under-mb`)
  are documented and used in the same `--assert-ttfw-under-ms`
  pattern.

### Prior art in the codebase

- `tests/test_chunker.py` for the warmup test pattern (pure
  unit, stub backend).
- `tests/test_audio_sink.py` for the buffer test pattern
  (real `AudioSink` with the underlying `sd.OutputStream`
  treated as a no-op in tests).
- `tests/test_worker_pool.py` for the pool-stub pattern the
  buffer tests reuse.
- `bench_ttfw.py --assert-*` for the bench-level regression
  pattern.

### What we explicitly do NOT test

- We do not test that cuDNN kernels are actually compiled (we
  observe the *effect* — stable TTFW — not the mechanism).
- We do not test the exact VRAM numbers (those are hardware-
  dependent); we test the plateau behavior, not the magnitude.
- We do not test the deprecation warning is gone (it would
  be testing the absence of a thing); we just verify the
  import works.

## Out of Scope

- **CUDA Graphs (fixed-shape padded inference).** This is the
  next big latency lever (2-4× infer speedup) but requires a
  prototype to validate output equivalence. Deferred to a
  follow-up PRD (the deferred work from PRD-001, plus the
  shape-range warmup here will further reduce the urgency).
- **Concurrent multi-stream serving.** The look-ahead buffer
  is the foundation, but the actual scheduler that batches
  across concurrent requests is a separate project. Out of
  scope.
- **Voice blending / new voice models.** Fun and a real
  product feature, but unrelated to latency.
- **Text normalization improvements (numbers, units, URLs).**
  Audio quality work, not latency.
- **Changing the CLI's interactive loop or adding a
  server/network mode.** The CLI stays a CLI.
- **GPU memory pre-allocation tuning beyond `--vram-cap-mb`.**
  If users want fine-grained control, that's a future flag.

## Further Notes

- **Why shape-range warmup over "warmup with the actual first
  text"?** We don't know the first text at startup time. We
  could lazy-warmup on the first prompt (warmup with chunk 0
  before submitting to the pool), but that *moves* the cold-
  compile cost to the user-visible TTFW instead of startup.
  Pre-compiling a range of shapes at startup is the only way
  to make TTFW stable for any input without re-introducing
  the 377ms spike.

- **Why pre-generate-whole-prompt is "look-ahead buffer" not
  "batch inference"?** Batching inference would change the
  compute path (one `sess.run` with multiple inputs) and
  potentially improve GPU utilization. We are NOT doing that
  here. We are simply removing the artificial back-pressure
  in the worker so the GPU has chunks ready before playback
  needs them. The actual GPU work per chunk is unchanged;
  only the *scheduling* is. Future PRDs can add batching on
  top of this foundation.

- **The `--pre-generate` default.** It's ON because the
  performance benefit is real (buffer_health data is
  meaningful, hiccups are detectable, the architecture
  matches the new reality) and the cost is zero (we already
  submit all chunks in a tight loop). Users who specifically
  want the old behavior have `--no-pre-generate`. The
  default-flip is justified because the old behavior was
  based on a wrong assumption (that generation was the
  bottleneck, when in fact it has 17× headroom).

- **The pynvml → nvidia-ml-py migration is trivial but worth
  doing in this PRD** because the deprecation warning has been
  noise-on-every-startup since the swap in PRD-001. Cleaning
  it up improves the UX of every run.

- **Risk: `--warmup-shapes` with very large counts.** A user
  who sets `--warmup-shapes 510` would pre-compile the
  MAX_PHONEME_LENGTH shape, which is fine but takes ~700ms
  (the existing long warmup). The default cap of 256 is
  chosen to stay under the typical 25-word chunk (~150
  phonemes) and not waste startup time. A `--warmup-shapes-max`
  safety cap is not strictly needed but could be added if
  this becomes an abuse vector.

- **Risk: look-ahead buffer masks real performance issues.**
  If generation *does* slow down (thermal throttle, background
  load), the buffer absorbs it for the first few chunks, so
  the user sees no degradation in playback. The `hiccup_count`
  metric is the counterweight: it's visible in the report and
  in `--profile` JSON, so a real problem doesn't go unnoticed.
  This is the right trade — silent degradation is worse than
  a visible hiccup metric.

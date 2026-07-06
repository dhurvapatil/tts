---
id: 002-005
title: Look-ahead buffer: pre-generate the whole prompt and report buffer health
triage: ready-for-agent
parent: PRD-002-warmup-and-buffering
blocked_by: none
---

## What to build

With generation at 0.06× realtime and playback at 1.0×, the worker
finishes all chunks in ~1.4s and then sits idle for ~22s while
the speaker drains. We still back-pressure the worker on playback
(`QUEUE_MAXSIZE = 4` + blocking writes). The architecture is right
for a system trying to keep up; ours doesn't need to.

Embrace the slack: submit all chunks to the worker at the start
of the prompt, then drain the queue to the AudioSink with the
existing crossfade. Add a `LookAheadBuffer` class (or equivalent
small coordinator on top of `WorkerPool` and `AudioSink`) that
schedules all chunks, drains them to the sink, and tracks three
new metrics:

- `buffer_health`: peak / median / final queue depth (chunks
  queued ahead of playback at each measurement)
- `hiccup_count`: number of chunks whose individual infer time
  exceeded that chunk's own audio duration (a stall event)
- `playback_underrun_count`: number of moments the consumer's
  `q.get()` found the queue empty *after the first chunk*
  (i.e. generation fell behind playback, not the initial
  chunk-0 wait)

Add the three new fields to the stats dict and to
`emit_profile_line` JSON output so `--profile` captures them in
machine-readable form.

A new `--pre-generate` flag (default ON for streaming) opts into
the new behavior. `--no-pre-generate` preserves today's submit-
as-you-read behavior for back-compat with any tooling that
depends on the visible submission order.

End-to-end: user runs `uv run tts.py`, types a prompt, sees a
new "Buffer health" section in the report (`peak: 4 chunks ahead ·
med: 3.5 · final: 0 · hiccups: 0 · underruns: 0`), and the
profile JSON line now includes those fields.

## Acceptance criteria

- [ ] A new `LookAheadBuffer` class exists with a small public
  surface:
    - `submit_all(jobs: list[tuple[str, str]])` — schedule every
      (text, voice) pair on the underlying `WorkerPool`
    - `drain_to_sink(sink: AudioSink, n_fade: int) -> Stats` —
      wait for the first chunk, start writing to the sink with
      crossfade, then drain the rest; returns the full stats
      dict including the three new fields
    - `health() -> BufferHealth` — returns the captured
      peak/median/final queue depth plus hiccup/underrun counts
- [ ] `play_streaming` integrates the buffer: by default, calls
  `buffer.submit_all(jobs)` immediately after chunking and
  `buffer.drain_to_sink(session_sink, n_fade)`. The existing
  `pool=session_pool` parameter and `WorkerPool` are unchanged
  underneath.
- [ ] `--pre-generate` (default ON for streaming) and
  `--no-pre-generate` flags are added. `--no-pre-generate`
  preserves today's submit-as-you-read path.
- [ ] The post-run report gains a "Buffer health" section with
  the three metrics. The format matches the existing report
  style (box-drawing separators, consistent width).
- [ ] `emit_profile_line` and `format_profile_batch` are
  extended to include `buffer_peak`, `buffer_median`,
  `buffer_final`, `hiccup_count`, and `playback_underrun_count`
  as new fields.
- [ ] A new test file `tests/test_look_ahead_buffer.py` covers:
    - Submitting N jobs schedules all N on the pool
    - `drain_to_sink` writes N chunks to a stub sink in order
    - `health().peak` is at least N-1 immediately after submit
      (i.e. the look-ahead is real)
    - A synthetic stall (a chunk whose backend sleeps longer
      than its audio duration) increments `hiccup_count`
    - `submit_all([])` is a no-op and `drain_to_sink` is a
      no-op on an empty buffer
    - The buffer survives an exception in the worker backend
      (doesn't deadlock the consumer)
- [ ] The existing `tests/test_profile_line.py` and
  `tests/test_worker_pool.py` are extended (or new tests added)
  to cover the new profile fields.
- [ ] All existing tests still pass.
- [ ] `uv run python bench_ttfw.py --runs 3` runs with the
  default `--pre-generate` and the bench's profile line includes
  the new fields with sensible values (peak >= 2, hiccups = 0,
  underruns = 0 on this hardware).

## Blocked by

None - can start immediately. (Logically benefits from 002-002
having cleaned up the WAV summary line, but no code-level
dependency.)

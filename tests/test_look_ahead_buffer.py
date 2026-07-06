"""Tests for the LookAheadBuffer (issue 002-005).

The LookAheadBuffer is a thin coordinator on top of WorkerPool and
AudioSink.  Its public surface:
    - submit_all(jobs): schedule every (text, voice) on the pool
    - drain_to_sink(sink, n_fade): wait for first chunk, start writing
      to the sink with crossfade, then drain the rest
    - health(): captured peak/median/final queue depth + hiccup +
      underrun counts

Pure unit tests.  No GPU, no real model, no real sounddevice.
"""
from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import tts  # noqa: E402


def _make_pool_with_chunks(n_chunks: int, chunk_dur_s: float = 0.05,
                            backend=None):
    """Build a WorkerPool whose backend returns a synthetic chunk
    (zeros of `chunk_dur_s` at 24000 Hz) for each (text, voice) job.
    Returns (pool, sample_rate).  If `backend` is given, that is used
    instead of the default synthetic backend."""
    sr = 24000
    if backend is None:
        n = int(sr * chunk_dur_s)

        def default_backend(text, voice):
            # Simulate a small per-chunk inference cost proportional
            # to the chunk index so the worker is realistically
            # "ahead" of the (non-existent) sink.
            time.sleep(0.001)
            return (np.zeros(n, dtype=np.float32), sr, 0.5, 5.0)
        backend = default_backend

    pool = tts.WorkerPool()
    pool.start(backend=backend)
    return pool, sr


class _StubSink:
    """Minimal stand-in for AudioSink: records what was written.

    The LookAheadBuffer only needs .write(samples) and a sample_rate
    to compute the crossfade length.  Real AudioSink isn't used here.
    """
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        self.written: list[np.ndarray] = []

    def write(self, samples: np.ndarray) -> None:
        self.written.append(samples.copy())


def test_submit_all_schedules_every_job():
    """submit_all() must put one job on the pool per input pair."""
    pool, _sr = _make_pool_with_chunks(n_chunks=0)
    try:
        buf = tts.LookAheadBuffer(pool=pool)
        buf.submit_all([("hello", "af_bella"), ("world", "af_bella"),
                        ("foo",   "af_bella")])
        # Peak depth should now be ~N (worker hasn't started yet on slow CI)
        h = buf.health()
        assert h.peak >= 2, (
            f"expected peak >= 2 (3 jobs queued), got {h.peak}"
        )
    finally:
        pool.close()


def test_drain_to_sink_writes_all_chunks_in_order():
    """drain_to_sink must write all chunks to the sink in order."""
    # Use a fast backend (no sleep) so the worker stays well ahead of
    # the consumer and the underrun count stays 0.
    sr = 24000
    n = int(sr * 0.05)
    def fast_backend(text, voice):
        return (np.zeros(n, dtype=np.float32), sr, 0.5, 5.0)
    pool = tts.WorkerPool()
    pool.start(backend=fast_backend)
    sink = _StubSink(sample_rate=sr)
    try:
        buf = tts.LookAheadBuffer(pool=pool)
        jobs = [(f"text-{i}", "af_bella") for i in range(4)]
        buf.submit_all(jobs)
        stats = buf.drain_to_sink(sink=sink, n_fade=int(sr * 0.04))
        # Sink must have 4 writes (one per chunk, possibly after crossfade
        # merge; we assert >= 4 since real drain produces >= N writes).
        assert len(sink.written) >= 4, (
            f"expected >= 4 writes, got {len(sink.written)}"
        )
        # And stats.n_chunks == 4.
        assert stats["n_chunks"] == 4
        # Hiccups and underruns should be 0 on a healthy path.
        assert stats["hiccup_count"] == 0
        assert stats["playback_underrun_count"] == 0
    finally:
        pool.close()


def test_health_peak_at_least_n_minus_one_after_submit():
    """After submit_all(N), the peak queue depth must be at least N-1
    (look-ahead is real: the worker can't drain before the consumer
    starts reading)."""
    pool, _sr = _make_pool_with_chunks(n_chunks=0)
    try:
        buf = tts.LookAheadBuffer(pool=pool)
        buf.submit_all([(f"t{i}", "af_bella") for i in range(5)])
        h = buf.health()
        assert h.peak >= 4, (
            f"expected peak >= 4 after submitting 5 jobs, got {h.peak}"
        )
        assert h.final >= 0
    finally:
        pool.close()


def test_hiccup_increments_when_chunk_infer_exceeds_audio_duration():
    """A synthetic stall (a chunk whose backend sleeps longer than
    the chunk's audio duration) must increment hiccup_count."""
    sr = 24000
    audio_dur_s = 0.05  # 1200 samples
    n = int(sr * audio_dur_s)

    def slow_backend(text, voice):
        # Sleep 0.1s (twice the audio duration) to trigger a hiccup.
        time.sleep(0.1)
        return (np.zeros(n, dtype=np.float32), sr, 0.5, 100.0)

    pool = tts.WorkerPool()
    pool.start(backend=slow_backend)
    sink = _StubSink(sample_rate=sr)
    try:
        buf = tts.LookAheadBuffer(pool=pool)
        buf.submit_all([(f"t{i}", "af_bella") for i in range(3)])
        stats = buf.drain_to_sink(sink=sink, n_fade=int(sr * 0.04))
        # At least one chunk had infer=100ms > 50ms audio duration.
        assert stats["hiccup_count"] >= 1, (
            f"expected >= 1 hiccup, got {stats['hiccup_count']}"
        )
    finally:
        pool.close()


def test_empty_submit_and_drain_is_noop():
    """submit_all([]) and drain_to_sink on an empty buffer must not
    raise and must produce a stats dict with zero counts."""
    pool, _sr = _make_pool_with_chunks(n_chunks=0)
    sink = _StubSink()
    try:
        buf = tts.LookAheadBuffer(pool=pool)
        buf.submit_all([])
        stats = buf.drain_to_sink(sink=sink, n_fade=10)
        assert stats["n_chunks"] == 0
        assert stats["hiccup_count"] == 0
        assert stats["playback_underrun_count"] == 0
        assert len(sink.written) == 0
    finally:
        pool.close()


def test_buffer_survives_worker_exception():
    """If the worker backend raises on one job, the buffer must not
    deadlock the consumer; subsequent jobs must still be drainable
    (the test asserts drain_to_sink returns within a reasonable time
    rather than hanging)."""
    sr = 24000
    n = int(sr * 0.05)
    jobs_processed = []

    def flaky_backend(text, voice):
        jobs_processed.append(text)
        if text == "boom!":
            raise RuntimeError("synthetic backend failure")
        return (np.zeros(n, dtype=np.float32), sr, 0.5, 5.0)

    pool = tts.WorkerPool()
    pool.start(backend=flaky_backend)
    sink = _StubSink(sample_rate=sr)
    try:
        buf = tts.LookAheadBuffer(pool=pool)
        buf.submit_all([("ok1", "af_bella"), ("boom!", "af_bella"),
                        ("ok2", "af_bella")])
        # Must return within 5s (no hang).
        t0 = time.perf_counter()
        stats = buf.drain_to_sink(sink=sink, n_fade=int(sr * 0.04))
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"drain_to_sink hung for {elapsed:.1f}s"
        # Backend must have processed all 3 jobs (failure does not
        # block the worker).
        assert jobs_processed == ["ok1", "boom!", "ok2"]
    finally:
        pool.close()


def test_health_returns_buffer_health_dataclass():
    """health() must return an object exposing the four captured metrics."""
    pool, _sr = _make_pool_with_chunks(n_chunks=2)
    try:
        buf = tts.LookAheadBuffer(pool=pool)
        buf.submit_all([("a", "af_bella"), ("b", "af_bella")])
        h = buf.health()
        for attr in ("peak", "median", "final", "hiccup_count",
                     "playback_underrun_count"):
            assert hasattr(h, attr), f"BufferHealth must expose {attr!r}"
    finally:
        pool.close()


def test_emit_profile_line_includes_buffer_health_fields():
    """The profile line JSON must include the new buffer-health fields
    so --profile captures them in machine-readable form."""
    stats = {
        "t_ttfw": 0.1,
        "chunk0_phonemize_ms": 0.5,
        "chunk0_infer_ms": 50.0,
        "per_chunk_infer_ms": [50.0, 50.0],
        "time_total": 1.0,
        "sample_rate": 24000,
        "audio_duration_s": 1.0,
        "voice": "af_bella",
        "lead_in_words": 6,
        "n_chunks": 2,
        "stream_open_ms": 0.0,
        # New issue 002-005 fields:
        "buffer_peak": 2,
        "buffer_median": 1.5,
        "buffer_final": 0,
        "hiccup_count": 0,
        "playback_underrun_count": 0,
    }
    import json
    obj = json.loads(tts.emit_profile_line(stats))
    assert obj["buffer_peak"] == 2
    assert obj["buffer_median"] == 1.5
    assert obj["buffer_final"] == 0
    assert obj["hiccup_count"] == 0
    assert obj["playback_underrun_count"] == 0


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

"""Tests for play_streaming() submission behavior (issue: double-submit bug).

Regression test for a bug where play_streaming with `pre_generate=True`
submitted every chunk to the pool TWICE:
  1. Once in the unconditional `if use_pool:` block (line ~1730)
  2. Once in the `if use_pool and pre_generate:` block via
     LookAheadBuffer.submit_all()

The worker is a single thread processing jobs FIFO, so the consumer
waited for the worker to finish the first N jobs (whose results were
never read) before it could read the first result of the second N
jobs.  This added ~3.6s of "other" to TTFW (everything 1× the chunk
count + a bit more for queue contention).

This test pins the contract: each chunk is submitted to the pool
EXACTLY ONCE regardless of `pre_generate`.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import tts  # noqa: E402


def _make_pool_with_counting_submit():
    """Build a WorkerPool whose .submit() is wrapped to count calls.

    Returns (pool, submit_count_dict).  The pool uses a fast stub
    backend that returns 50ms of silence; the test only cares about
    the submit-call count, not the audio quality.
    """
    sr = 24000
    n = int(sr * 0.05)

    def fast_backend(*args):
        return (np.zeros(n, dtype=np.float32), sr, 0.5, 5.0)

    pool = tts.WorkerPool()
    pool.start(backend=fast_backend)

    submit_count = {"n": 0}
    original_submit = pool.submit

    def counting_submit(text, voice):
        submit_count["n"] += 1
        return original_submit(text, voice)

    pool.submit = counting_submit
    return pool, submit_count


class _StubKokoro:
    """Minimal stand-in for the real Kokoro object.

    play_streaming's internal pool_backend closure calls
    `_timed_generate_chunk_audio(_kokoro, text, _voice)`, which
    accesses `kokoro.tokenizer.phonemize` and `kokoro.create`.
    The pool's own backend is what actually runs (fast_backend
    above) and ignores these, so we just need the object to
    exist with a `tokenizer` attribute for set_call_args().
    """
    class _StubTokenizer:
        def phonemize(self, *_args, **_kwargs):
            return "stub"

    def __init__(self):
        self.tokenizer = self._StubTokenizer()

    def create(self, *_args, **_kwargs):
        return (np.zeros(100, dtype=np.float32), 24000)


class _StubSink:
    """Minimal stand-in for AudioSink.

    The legacy pre_generate=False path calls sink.open() on the
    first chunk (see _consume_chunk).  Pre-generate path doesn't,
    so .open() is only needed for the legacy test.
    """
    def open(self, sample_rate):
        pass

    def write(self, samples):
        pass


def test_play_streaming_pre_generate_submits_each_chunk_exactly_once():
    """With pre_generate=True, each chunk must be submitted to the
    pool exactly once.  The double-submit bug caused 2N submits."""
    pool, submit_count = _make_pool_with_counting_submit()
    kokoro = _StubKokoro()
    sink = _StubSink()
    chunks = ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.wav"
        try:
            stats = tts.play_streaming(
                chunks, "af_bella", kokoro, out_path,
                t_user_press=time.perf_counter(),
                sink=sink, pool=pool, pre_generate=True,
            )
        finally:
            pool.close()
    assert submit_count["n"] == len(chunks), (
        f"play_streaming with pre_generate=True must submit each "
        f"chunk exactly once; got {submit_count['n']} submits for "
        f"{len(chunks)} chunks (double-submission regression)."
    )
    # Sanity: the stats should still be populated from the pre-generate
    # path (n_chunks matches, buffer_peak > 0, etc.).
    assert stats["n_chunks"] == len(chunks)
    assert stats["buffer_peak"] >= 1


def test_play_streaming_legacy_pool_path_submits_each_chunk_exactly_once():
    """With pre_generate=False, the legacy pool path must also
    submit each chunk exactly once.  The unconditional `if use_pool:`
    block historically did the submission; we must ensure that path
    still works after the refactor."""
    pool, submit_count = _make_pool_with_counting_submit()
    kokoro = _StubKokoro()
    sink = _StubSink()
    chunks = ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.wav"
        try:
            stats = tts.play_streaming(
                chunks, "af_bella", kokoro, out_path,
                t_user_press=time.perf_counter(),
                sink=sink, pool=pool, pre_generate=False,
            )
        finally:
            pool.close()
    assert submit_count["n"] == len(chunks), (
        f"play_streaming with pre_generate=False must submit each "
        f"chunk exactly once; got {submit_count['n']} submits for "
        f"{len(chunks)} chunks."
    )
    # Legacy path doesn't go through LookAheadBuffer, so buffer_peak=0.
    assert stats["buffer_peak"] == 0


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

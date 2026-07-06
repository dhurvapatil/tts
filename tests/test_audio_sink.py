"""Tests for the AudioSink abstraction.

The AudioSink encapsulates a single sd.OutputStream that is opened
once and reused across prompts.  Tests cover the public behavior:
- open() opens a stream at the given sample rate
- write() forwards samples to the stream
- close() is idempotent
- re-open on a write failure is automatic (the sink survives
  device changes / stream errors)
- is_open() reports state
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import tts  # noqa: E402


def test_audio_sink_open_close():
    sink = tts.AudioSink()
    assert not sink.is_open()
    sink.open(sample_rate=24000)
    assert sink.is_open()
    sink.close()
    assert not sink.is_open()


def test_audio_sink_close_is_idempotent():
    sink = tts.AudioSink()
    sink.close()  # closing an unopened sink must not raise
    sink.open(sample_rate=24000)
    sink.close()
    sink.close()  # closing twice must not raise


def test_audio_sink_write_succeeds_when_open():
    sink = tts.AudioSink()
    sink.open(sample_rate=24000)
    # 100ms of silence
    samples = np.zeros(2400, dtype=np.float32)
    # write should not raise (the actual audio device may be a dummy
    # in CI; we only assert no exception, not that sound was heard)
    sink.write(samples)
    sink.close()


def test_audio_sink_reopen_on_failure():
    """If a write fails because the device went away, the sink must
    auto-reopen on the next write rather than raising."""
    sink = tts.AudioSink()
    sink.open(sample_rate=24000)
    # Simulate a stale stream by closing the underlying stream out from
    # under the sink.
    sink._stream = None
    samples = np.zeros(100, dtype=np.float32)
    sink.write(samples)  # should auto-reopen and not raise
    assert sink.is_open()
    sink.close()


def test_audio_sink_context_manager():
    with tts.AudioSink() as sink:
        sink.open(sample_rate=24000)
        assert sink.is_open()
    assert not sink.is_open()


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

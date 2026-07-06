"""Tests for the long-lived worker pool.

A WorkerPool owns a single daemon thread and a queue of jobs.  Each job
is a (text, voice) pair; the worker pulls jobs, runs inference, and
puts the result on a per-job result queue.  Reusing a single thread
across prompts avoids the per-prompt thread-spawn latency (~5-20ms
on Windows) which is variance on the TTFW measurement.

Public behavior tested:
- submit() returns a result queue for the caller to read
- jobs run in submission order (FIFO)
- failures in a job don't kill the worker (it stays alive for the next)
- the pool can be closed cleanly
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


def _make_pool():
    """Make a WorkerPool with a stub backend (no GPU/model)."""
    pool = tts.WorkerPool()
    pool.start(backend=lambda text, voice: f"audio:{text}:{voice}")
    return pool


def test_pool_runs_job_in_order():
    pool = _make_pool()
    try:
        r1 = pool.submit("hello", "af_bella")
        r2 = pool.submit("world", "af_bella")
        assert r1.get(timeout=2.0) == "audio:hello:af_bella"
        assert r2.get(timeout=2.0) == "audio:world:af_bella"
    finally:
        pool.close()


def test_pool_survives_job_failure():
    pool = tts.WorkerPool()

    def flaky(text, voice):
        if text == "boom!":
            raise RuntimeError("synthetic failure")
        return f"ok:{text}"

    pool.start(backend=flaky)
    try:
        r1 = pool.submit("hello", "af_bella")
        r2 = pool.submit("boom!", "af_bella")
        r3 = pool.submit("after boom", "af_bella")
        assert r1.get(timeout=2.0) == "ok:hello"
        # r2 should surface the exception
        try:
            r2.get(timeout=2.0)
            assert False, "expected exception on r2"
        except RuntimeError as e:
            assert "synthetic failure" in str(e)
        # Worker is still alive for r3
        assert r3.get(timeout=2.0) == "ok:after boom"
    finally:
        pool.close()


def test_pool_close_is_safe():
    pool = _make_pool()
    pool.close()
    # double-close should not raise
    pool.close()


def test_pool_close_waits_for_in_flight():
    pool = tts.WorkerPool()
    seen = {"done": False}

    def slow(text, voice):
        time.sleep(0.1)
        seen["done"] = True
        return "ok"

    pool.start(backend=slow)
    pool.submit("a", "af_bella")
    pool.close()
    # The close() must have waited for the in-flight job to finish
    assert seen["done"] is True


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

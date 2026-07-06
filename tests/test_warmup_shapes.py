"""Tests for the shape-range warmup (issue 002-003).

The old warmup used two fixed texts (42 and 226 phonemes) that left a
big shape gap.  Real lead-ins phonemize to 30, 55, 80, etc. and the
first prompt's chunk-0 would trigger an on-demand cuDNN kernel compile.

The new `run_warmups` phonemizes one long source text once, then
truncates/pads to each target phoneme count and runs inference.  This
warms the shape range so any lead-in falls on a pre-compiled kernel.

Pure unit tests: stub `kokoro` and assert the call shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


class StubKokoro:
    """Records every call to .create() so tests can assert the
    phoneme-length sequence, the call order, and the call count.

    `phonemize` is a method (matching the real kokoro API).
    """

    def __init__(self, long_phonemes: str) -> None:
        # The "long" phoneme string that the warmup will truncate/pad.
        self._long = long_phonemes
        self.calls: list[int] = []  # phoneme length of each create() call
        # tokenizer.phonemize returns a fixed string so all we care about
        # is the length and the suffix that the warmup truncates from.
        class _Tok:
            def __init__(self, s):
                self._s = s
            def phonemize(self, _text, lang="en-us", norm=True):
                # Return the long string regardless of input (the warmup
                # uses one fixed source text).  The real implementation
                # would parse a different text; for the test we just
                # need a long enough string to truncate.
                return self._s
        self.tokenizer = _Tok(self._long)

    def create(self, phonemes, voice, lang, is_phonemes, speed):
        # Record the length of the phoneme string we received.
        self.calls.append(len(phonemes))
        # Return dummy audio (1 sample at 24000 Hz).  Audio is discarded
        # by the warmup, so the actual values don't matter.
        return ([0.0] * 100, 24000)


def _long_phonemes(n: int) -> str:
    return "a" * n  # any token works for length-based logic


def test_run_warmups_calls_kokoro_once_per_target_in_order():
    """Each target phoneme count must result in exactly one create() call,
    in the same order as the input list."""
    kokoro = StubKokoro(_long_phonemes(300))
    targets = [20, 60, 120, 256]
    pairs = tts.run_warmups(sess=None, kokoro=kokoro, target_phoneme_counts=targets)
    assert kokoro.calls == targets, (
        f"expected one create() per target, in order {targets}; got {kokoro.calls}"
    )
    # Returns one (count, ms) pair per input count, in input order.
    assert [c for c, _ in pairs] == targets
    assert all(ms >= 0.0 for _, ms in pairs)


def test_run_warmups_truncates_long_phoneme_string_to_target():
    """If the source is longer than the target, the warmup must
    truncate to exactly `target` phonemes."""
    kokoro = StubKokoro(_long_phonemes(500))
    tts.run_warmups(sess=None, kokoro=kokoro, target_phoneme_counts=[100])
    assert kokoro.calls == [100]


def test_run_warmups_pads_short_phoneme_string_to_target():
    """If the source is shorter than the target, the warmup must
    pad to exactly `target` phonemes."""
    kokoro = StubKokoro(_long_phonemes(30))
    tts.run_warmups(sess=None, kokoro=kokoro, target_phoneme_counts=[200])
    assert kokoro.calls == [200]


def test_run_warmups_empty_target_list_is_noop():
    """An empty target list must make zero create() calls and return []."""
    kokoro = StubKokoro(_long_phonemes(100))
    pairs = tts.run_warmups(sess=None, kokoro=kokoro, target_phoneme_counts=[])
    assert kokoro.calls == []
    assert pairs == []


def test_run_warmups_returns_one_pair_per_input():
    """Returned list length must match input length, even with duplicates."""
    kokoro = StubKokoro(_long_phonemes(300))
    targets = [20, 20, 40, 80, 80, 80]
    pairs = tts.run_warmups(sess=None, kokoro=kokoro, target_phoneme_counts=targets)
    assert len(pairs) == len(targets)
    assert [c for c, _ in pairs] == targets


def test_run_warmups_phonemizes_source_only_once():
    """The warmup must phonemize its source text once, not per target."""
    call_count = {"n": 0}

    class _CountingTok:
        def phonemize(self, _text, lang="en-us", norm=True):
            call_count["n"] += 1
            return "a" * 500  # long enough for any target
    kokoro = StubKokoro(_long_phonemes(500))
    kokoro.tokenizer = _CountingTok()
    tts.run_warmups(sess=None, kokoro=kokoro, target_phoneme_counts=[20, 60, 100, 200])
    assert call_count["n"] == 1, (
        f"phonemize must be called once, not {call_count['n']} times"
    )


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

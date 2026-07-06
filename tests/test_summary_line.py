"""Tests for the WAV summary line (issue 002-002).

The summary line printed at the top of each prompt's output must use
generation-RTF as the headline number (with the faster/slower verdict),
matching the Latency section below.  Wall-RTF is shown as a secondary,
un-verdicted number on the same line.

Pure unit tests: feed a stats dict, assert on the returned string.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


def _stats(**overrides):
    base = dict(
        t_ttfw=0.662,
        chunk0_phonemize_ms=1.0,
        chunk0_infer_ms=120.0,
        per_chunk_infer_ms=[120.0, 177.0, 283.0, 254.0, 100.0, 80.0, 90.0, 60.0],
        time_total=43.23,
        t_user_press=0.0,
        t_first_audio=0.662,
        t_wall_end=43.23,
        sample_rate=24000,
        audio_duration_s=41.75,
        voice="am_liam",
        lead_in_words=8,
        n_chunks=8,
    )
    base.update(overrides)
    return base


def test_format_summary_line_uses_generation_rtf():
    """The headline RTF number must be the generation-RTF (model speed),
    not the wall-RTF (audio sink speed)."""
    s = tts.format_summary_line(_stats())
    # The generation RTF for these stats is sum(per_chunk)/audio_ms.
    gen_ms = sum([120.0, 177.0, 283.0, 254.0, 100.0, 80.0, 90.0, 60.0])
    expected_gen = gen_ms / (41.75 * 1000.0)
    # Expected gen ≈ 0.028x
    # The summary line must show this number, labeled "gen" (not "wall")
    assert f"{expected_gen:.2f}" in s, (
        f"summary line must show generation-RTF {expected_gen:.2f}; got: {s!r}"
    )


def test_format_summary_line_contains_gen_label():
    """The line must label the headline number as generation-RTF so
    it can be distinguished from the wall-RTF."""
    s = tts.format_summary_line(_stats())
    assert "gen" in s.lower(), (
        f"summary line must label the generation-RTF (e.g. 'RTF (gen)'); got: {s!r}"
    )


def test_format_summary_line_does_not_say_slower_when_gen_under_one():
    """The verdict on the summary line must follow generation-RTF,
    not wall-RTF.  Wall-RTF is 43.23/41.75 ≈ 1.04 (slower than realtime)
    while generation-RTF is 0.028 (way faster).  The summary line must
    NOT say 'slower than realtime' here."""
    s = tts.format_summary_line(_stats())
    assert "slower than realtime" not in s.lower(), (
        f"summary line must not apply 'slower than realtime' verdict "
        f"to the generation-RTF; got: {s!r}"
    )


def test_format_summary_line_contains_wall_rtf():
    """The wall-RTF value must still appear on the line as a fact,
    just without the verdict."""
    s = tts.format_summary_line(_stats())
    expected_wall = 43.23 / 41.75
    assert f"{expected_wall:.2f}" in s, (
        f"summary line must still show wall-RTF {expected_wall:.2f}; got: {s!r}"
    )
    assert "wall" in s.lower(), (
        f"summary line must label the wall-RTF (e.g. 'wall 1.04x'); got: {s!r}"
    )


def test_format_summary_line_says_slower_when_gen_over_one():
    """The verdict on the summary line DOES follow generation-RTF,
    so when gen >= 1.0 the line must say 'slower' (or similar)."""
    # Construct stats where generation-RTF is huge: 4 chunks of 5000ms
    # of inference each, audio is 1s.
    s = tts.format_summary_line(_stats(
        per_chunk_infer_ms=[5000.0, 5000.0, 5000.0, 5000.0],
        time_total=20.0,
        audio_duration_s=1.0,
    ))
    # gen = 20000ms / 1000ms = 20.0x, wall = 20.0x
    assert "slower" in s.lower() or "✗" in s, (
        f"summary line must show slower-verdict when gen >= 1.0; got: {s!r}"
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

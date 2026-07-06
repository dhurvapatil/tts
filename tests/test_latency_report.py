"""Tests for the latency / streaming-health report formatting.

Pure unit tests.  The public behavior: given a stats dict, the report
must surface generation RTF and wall RTF as separate, clearly-labeled
lines, and the "faster / slower than realtime" verdict must apply to
the generation RTF (a model property) — not the wall RTF (a property
of the audio sink).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


def _stats(**overrides):
    base = dict(
        t_ttfw=0.775,
        chunk0_phonemize_ms=1.0,
        chunk0_infer_ms=270.0,
        per_chunk_infer_ms=[270.0, 177.0, 283.0, 254.0],
        time_total=17.05,
        t_user_press=0.0,
        t_first_audio=0.775,
        t_wall_end=17.05,
        sample_rate=24000,
        audio_duration_s=15.88,
        voice="af_bella",
        lead_in_words=15,
        n_chunks=4,
    )
    base.update(overrides)
    return base


def test_compute_generation_rtf():
    # sum of infer ms / audio ms
    rtf = tts.compute_generation_rtf(
        per_chunk_infer_ms=[270.0, 177.0, 283.0, 254.0],
        audio_duration_s=15.88,
    )
    expected = (270.0 + 177.0 + 283.0 + 254.0) / (15.88 * 1000.0)
    assert abs(rtf - expected) < 1e-9


def test_compute_generation_rtf_zero_audio():
    assert tts.compute_generation_rtf([], 0.0) == 0.0


def test_compute_wall_rtf():
    assert abs(tts.compute_wall_rtf(17.05, 15.88) - 17.05 / 15.88) < 1e-9


def test_format_rtf_verdict_applies_to_generation_only():
    # generation RTF is good (< 1) and wall RTF is bad (> 1)
    line = tts.format_rtf_line(
        generation_rtf=0.06, wall_rtf=1.07,
    )
    # Must contain BOTH numbers, both clearly labeled
    assert "generation" in line.lower()
    assert "wall" in line.lower()
    # Verdict applies to generation (0.06 < 1.0)
    assert "faster than realtime" in line.lower()
    assert "✓" in line
    # Wall RTF (1.07) is NOT the verdict — should be presented as a fact
    # but not labeled "slower than realtime"
    parts = [p.strip() for p in line.split("|") if p.strip()]
    wall_parts = [p for p in parts if "wall" in p.lower()]
    assert wall_parts, "wall RTF must appear in the line"
    for p in wall_parts:
        assert "slower than realtime" not in p.lower(), (
            "wall RTF must not carry the verdict (it's a sink property)"
        )


def test_format_ttfw_breakdown_includes_all_phases():
    line = tts.format_ttfw_breakdown(
        chunk0_phonemize_ms=1.0,
        chunk0_infer_ms=120.0,
        stream_open_ms=4.0,
        t_ttfw_ms=125.0,
    )
    # Must include each labeled phase
    assert "phon" in line.lower()
    assert "infer" in line.lower()
    assert "stream-open" in line.lower() or "stream open" in line.lower()
    # Must include the total
    assert "125" in line or "125.0" in line


def test_format_ttfw_breakdown_sums_consistently():
    # The line must show every named phase and an "other" residual
    # that absorbs the difference vs t_ttfw_ms.  Specifically:
    #   2 + 180 + 10 = 192 (named) ; 300 - 192 = 108 (other)
    line = tts.format_ttfw_breakdown(
        chunk0_phonemize_ms=2.0,
        chunk0_infer_ms=180.0,
        stream_open_ms=10.0,
        t_ttfw_ms=300.0,
    )
    # Named phases must each be visible
    assert "2ms" in line or "2.0" in line
    assert "180ms" in line or "180.0" in line
    assert "10ms" in line or "10.0" in line
    # Other (residual) must be shown so the breakdown sums to t_ttfw_ms
    assert "108ms" in line or "108.0" in line
    assert "other" in line.lower() or "queue" in line.lower() or "overhead" in line.lower()


def test_print_latency_report_signature_does_not_require_legacy_rtf_arg():
    """Regression test: main() historically passed a wall-RTF float
    as a positional arg to print_latency_report.  After the 002-002
    refactor, that float went away (we compute RTFs inside the report
    from `stats` and `duration_s`), so the parameter was removed and
    the call site must NOT pass it.  This test pins the new signature
    so the call site doesn't drift back to passing a now-undefined
    local (the original bug: a live CLI run crashed with
    `NameError: name 'rtf' is not defined`)."""
    import inspect
    params = list(inspect.signature(tts.print_latency_report).parameters.keys())
    assert "rtf" not in params, (
        f"print_latency_report must not have an `rtf` parameter (issue "
        f"002-002 refactor moved RTF computation inside the function "
        f"from the stats dict).  Current params: {params}"
    )
    # The remaining params are the public surface; pin them so a future
    # accidental change is caught.
    assert params == [
        "stats", "n_chunks", "word_counts", "duration_s",
        "is_streaming_path",
    ], f"print_latency_report signature changed unexpectedly: {params}"


def test_print_latency_report_runs_without_rtf_arg(capsys=None):
    """Smoke test: the new signature (no rtf param) must work end-to-end.
    The function must print the Latency section without raising."""
    import io
    import contextlib
    stats = dict(
        t_ttfw=0.1, chunk0_phonemize_ms=1.0, chunk0_infer_ms=50.0,
        per_chunk_infer_ms=[50.0, 50.0], time_total=1.0,
        t_wall_end=1.0, t_user_press=0.0, sample_rate=24000,
        audio_duration_s=1.0,
        q_get_blocks=0, q_get_block_total_ms=0.0,
        stream_open_ms=0.0,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tts.print_latency_report(
            stats, 2, [5, 5], 1.0,
            is_streaming_path=True,
        )
    out = buf.getvalue()
    assert "Latency" in out
    assert "Streaming health" in out
    # Buffer health section should appear when buffer_peak > 0.
    stats_with_buffer = dict(stats, buffer_peak=3, buffer_median=2.0,
                             buffer_final=0, hiccup_count=0,
                             playback_underrun_count=0)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        tts.print_latency_report(
            stats_with_buffer, 2, [5, 5], 1.0,
            is_streaming_path=True,
        )
    assert "Buffer health" in buf2.getvalue()


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

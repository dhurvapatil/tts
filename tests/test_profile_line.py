"""Tests for the per-prompt JSON profile line.

These tests verify the public observable behavior of the profile emitter:
a single line of JSON on stdout (or a returned string) that captures the
metrics an agent or CI run can diff across runs.

Pure unit tests. No GPU, no model, no sounddevice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `uv run python tests/test_profile_line.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts import emit_profile_line, format_profile_batch  # noqa: E402


# These also get exercised in the bench path, but their *public* output
# contract is the stats dict: a real agent can import and reuse it.


def _stats(**overrides):
    base = dict(
        t_ttfw=0.775,
        chunk0_phonemize_ms=1.0,
        chunk0_infer_ms=270.0,
        per_chunk_infer_ms=[270.0, 177.0, 283.0, 254.0],
        time_total=17.05,
        sample_rate=24000,
        audio_duration_s=15.88,
        voice="af_bella",
        lead_in_words=15,
        n_chunks=4,
        stream_open_ms=0.0,
    )
    base.update(overrides)
    return base


def test_emits_valid_json_line():
    line = emit_profile_line(_stats())
    obj = json.loads(line)
    assert isinstance(obj, dict)


def test_includes_ttfw_ms():
    obj = json.loads(emit_profile_line(_stats(t_ttfw=0.123)))
    assert obj["ttfw_ms"] == 123.0


def test_includes_chunk0_breakdown():
    obj = json.loads(emit_profile_line(_stats()))
    assert obj["chunk0_phonemize_ms"] == 1.0
    assert obj["chunk0_infer_ms"] == 270.0


def test_includes_generation_rtf_and_wall_rtf_separately():
    obj = json.loads(emit_profile_line(_stats()))
    # generation RTF = sum(per_chunk_infer_ms) / audio_duration_ms
    gen_ms = sum([270.0, 177.0, 283.0, 254.0])
    expected_gen = gen_ms / (15.88 * 1000.0)
    assert abs(obj["generation_rtf"] - expected_gen) < 1e-6
    # wall RTF = time_total / audio_duration_s
    assert abs(obj["wall_rtf"] - (17.05 / 15.88)) < 1e-6
    assert obj["generation_rtf"] != obj["wall_rtf"]


def test_includes_voice_and_chunk_metadata():
    obj = json.loads(emit_profile_line(_stats(voice="am_liam", lead_in_words=8, n_chunks=2)))
    assert obj["voice"] == "am_liam"
    assert obj["lead_in_words"] == 8
    assert obj["n_chunks"] == 2


def test_emits_exactly_one_line():
    line = emit_profile_line(_stats())
    assert line.count("\n") == 1, "profile line must end with exactly one newline"


def test_format_profile_batch_emits_one_line_per_prompt():
    out = format_profile_batch([_stats(t_ttfw=0.1), _stats(t_ttfw=0.2), _stats(t_ttfw=0.3)])
    # Exactly 3 lines, each its own JSON object
    assert out.count("\n") == 3, out
    objs = [json.loads(l) for l in out.splitlines() if l]
    assert len(objs) == 3
    assert [o["ttfw_ms"] for o in objs] == [100.0, 200.0, 300.0]


def test_format_profile_batch_handles_empty_input():
    out = format_profile_batch([])
    assert out == ""


def test_profile_line_includes_stream_open_ms():
    obj = json.loads(emit_profile_line(_stats()))
    assert "stream_open_ms" in obj
    # Default 0.0 (no stream-open cost in the bench path)
    assert obj["stream_open_ms"] == 0.0


def test_profile_line_round_trips_stream_open():
    obj = json.loads(emit_profile_line(_stats(stream_open_ms=4.2)))
    assert obj["stream_open_ms"] == 4.2


if __name__ == "__main__":
    # Plain-script runner: `uv run python tests/test_profile_line.py`
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ✓ {name}")
            except Exception:
                failed += 1
                print(f"  ✗ {name}")
                traceback.print_exc()
    if failed:
        sys.exit(1)
    print(f"\n  {sum(1 for n in globals() if n.startswith('test_') and callable(globals()[n]))} passed")

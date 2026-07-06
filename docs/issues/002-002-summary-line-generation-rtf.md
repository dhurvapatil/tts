---
id: 002-002
title: Fix the WAV summary line to use generation-RTF (consistent with Latency section)
triage: ready-for-agent
parent: PRD-002-warmup-and-buffering
blocked_by: none
---

## What to build

The top-of-prompt summary line in `tts.py` currently prints
`RTF 1.01 · slower than realtime ✗` using wall-RTF (time_total /
audio_duration_s). This is inconsistent with the Latency section
printed a few lines later, which correctly shows
`RTF (generation) 0.06x ✓` and `RTF (wall) 1.01x` as a separate,
un-verdicted fact.

The fix: extract a small `format_summary_line(stats: dict) -> str`
helper (matching the existing `format_rtf_line` /
`format_ttfw_breakdown` pattern) and have the WAV summary line
print generation-RTF as the headline number with the verdict,
matching the Latency section. Wall-RTF is shown as a secondary,
un-verdicted number on the same line.

End-to-end: user runs `uv run tts.py`, types a prompt, sees
`✓ af_nicole_...wav  (23.82s audio @ 24000 Hz,  RTF (gen) 0.06x ✓  ·  wall 1.01x)`
instead of the misleading `RTF 1.01 · slower than realtime ✗`.

## Acceptance criteria

- [ ] A new pure function `format_summary_line(stats) -> str` exists
  in `tts.py` near the other format helpers. It returns a single
  line of text ready to be printed.
- [ ] The new line uses `compute_generation_rtf()` for the headline
  number and the faster-than-realtime verdict.
- [ ] Wall-RTF is shown as a secondary, un-verdicted number
  (matching the `RTF (wall)` style in the Latency section).
- [ ] The WAV summary line in `tts.py` (printed at the top of each
  prompt's output) calls the new helper instead of computing
  wall-RTF inline.
- [ ] A new test (`tests/test_summary_line.py` or an addition to
  `test_latency_report.py`) asserts:
    - The returned string contains the generation-RTF value
    - The returned string contains "RTF (gen)" or equivalent label
    - The returned string does NOT contain "slower than realtime"
      when generation-RTF is < 1.0
    - The returned string DOES contain the wall-RTF value
- [ ] All existing tests still pass.
- [ ] Manually verified: running `uv run tts.py` and typing a
  prompt shows the consistent messaging (no `✗` then `✓`).

## Blocked by

None - can start immediately.

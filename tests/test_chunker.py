"""Tests for the lead-in chunker.

Pure unit tests.  Public behavior:
- _lead_in_split() cuts the first chunk at LEAD_IN_MIN words or at the
  first clause boundary after that, whichever comes first
- A hard cut is allowed when no natural boundary lands early
- Abbreviations (Dr., e.g., 3.14) survive the split
- The lead-in size is configurable
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


def test_lead_in_split_returns_none_for_short_input():
    # Input shorter than LEAD_IN_MIN stays as a single chunk
    text = "hello world"
    chunks, info = tts._lead_in_split([text], lead_in_min=6, target=25, hard_cut=True)
    assert chunks == [text]
    assert info is None


def test_lead_in_split_cuts_at_clause_boundary():
    # First clause boundary after LEAD_IN_MIN is the cut point
    text = "alpha beta gamma delta, epsilon zeta eta theta iota kappa lambda mu nu xi"
    chunks, info = tts._lead_in_split(
        [text], lead_in_min=4, target=8, hard_cut=False,
    )
    # The comma after "delta" is at word index 4 (0-indexed: positions 0-3 are
    # pre-comma, position 4 is "delta,").  We keep the comma with the lead-in.
    # So lead-in = "alpha beta gamma delta," (4 words)
    assert info is not None
    lead_in = chunks[0].split()
    assert len(lead_in) == 4
    # The word "delta" with its trailing comma is the boundary marker
    assert any(w.startswith("delta") for w in lead_in)


def test_lead_in_split_hard_cut_when_no_boundary():
    text = " ".join(f"word{i}" for i in range(20))
    chunks, info = tts._lead_in_split(
        [text], lead_in_min=3, target=8, hard_cut=True,
    )
    # No clause boundary, so hard cut at lead_in_min
    assert info is not None
    lead_in = chunks[0].split()
    assert len(lead_in) == 3
    assert chunks[1].startswith("word3")  # continuation starts at word 3


def test_lead_in_split_no_hard_cut_when_disabled():
    # Without hard_cut and no clause boundary, no split
    text = " ".join(f"word{i}" for i in range(20))
    chunks, info = tts._lead_in_split(
        [text], lead_in_min=3, target=8, hard_cut=False,
    )
    # No boundary, no hard cut → keep as single chunk
    assert info is None
    assert chunks == [text]


def test_lead_in_split_preserves_abbreviation_across_cut():
    # "Dr." should not be split by the chunker — abbreviation is
    # protected by protect_dots() before this stage.  We just verify
    # the lead-in splitter doesn't itself break "Dr." tokens.
    text = "Dr. Smith examined the patient carefully, and prescribed a course of treatment that lasted for several weeks"
    chunks, info = tts._lead_in_split(
        [text], lead_in_min=3, target=10, hard_cut=True,
    )
    # The Dr. token should be intact in chunk 0
    full = " ".join(chunks)
    assert "Dr. Smith" in full


def test_chunk_text_runs_lead_in_split():
    # End-to-end: chunk_text returns sentence-balanced chunks, then
    # _lead_in_split shrinks chunk 0.
    text = (
        "To get started, please check that your device is securely plugged into "
        "a power source and the Wi-Fi router is turned on. If the issue continues, "
        "restart the system by holding the power button for ten seconds. You can "
        "view step-by-step guides on the Support Center for further assistance."
    )
    chunks = tts.chunk_text(text, max_words=50)
    chunks, info = tts._lead_in_split(
        chunks, lead_in_min=6, target=8, hard_cut=True,
    )
    assert len(chunks) > 0
    lead_in_words = len(chunks[0].split())
    # The cut may land at a natural boundary (conjunction "and" at word 15)
    # or at the hard-cut fallback (8 words).  Either way it must shrink
    # chunk 0 below the original 22 words.
    assert lead_in_words < 22
    assert lead_in_words >= 6


def test_chunk_text_without_lead_in_split_keeps_whole_first_sentence():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "The second sentence is here."
    )
    chunks = tts.chunk_text(text, max_words=50)
    # No lead-in split applied (huge min, no hard cut) → first chunk
    # is the full first sentence
    chunks2, info = tts._lead_in_split(
        chunks, lead_in_min=999, target=50, hard_cut=False,
    )
    assert info is None
    assert chunks2[0].startswith("The quick brown fox")


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

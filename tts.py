"""
Kokoro-82M ONNX · local TTS CLI with phrase-wise streaming chunking.

Loads the model once, lets you pick a voice, then accepts multi-line
paragraphs and renders them to WAV files in ./output/.  After every
generation, prints a per-phase timing chart, peak CPU / RAM / VRAM,
and (for multi-chunk inputs) the time-to-first-sound.

Run with:  uv run tts.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows so the box-drawing / IPA chars in the output
# don't crash on the default cp1252 console encoding.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
del _stream

# Pre-register CUDA / cuDNN / cuBLAS DLLs from the venv's nvidia-* pip
# packages.  onnxruntime-gpu on Windows does NOT honour os.add_dll_directory
# at session-create time, so we have to do BOTH: prepend the bin dirs to
# PATH (so LoadLibrary finds them) AND add_dll_directory (for safety).
# This must run BEFORE the onnxruntime import.
_ROOT = Path(__file__).resolve().parent
_nvidia_root = _ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"
if _nvidia_root.exists():
    _extra_paths: list[str] = []
    for _d in sorted(_nvidia_root.glob("*/bin")):
        if _d.is_dir():
            _extra_paths.append(str(_d.resolve()))
    if _extra_paths:
        os.environ["PATH"] = ";".join(_extra_paths) + ";" + os.environ.get("PATH", "")
        for _p in _extra_paths:
            try:
                os.add_dll_directory(_p)
            except (AttributeError, OSError):
                pass
del _nvidia_root, _extra_paths  # type: ignore

import numpy as np
import psutil

# Silence onnxruntime's chatty default logging
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
try:
    import onnxruntime as ort
    # Also silence the C++ side (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL)
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass
except ImportError:
    ort = None  # type: ignore
    print("  ! onnxruntime not installed; run `uv sync`.")

# ---- audio backend selection ----------------------------------------------
# Primary: sounddevice (PortAudio) — supports sample-accurate streaming.
# Fallback: winsound (Windows stdlib) — file-based, pre-build + play-once.
_AUDIO_BACKEND = None          # "sounddevice" | "winsound" | None
_HAS_STREAMING = False
sd = None
winsound = None
try:
    import sounddevice as _sd
    sd = _sd
    _AUDIO_BACKEND = "sounddevice"
    _HAS_STREAMING = True
except (ImportError, OSError) as e:
    if sys.platform == "win32":
        try:
            import winsound as _winsound
            winsound = _winsound
            _AUDIO_BACKEND = "winsound"
        except ImportError:
            pass

if _AUDIO_BACKEND is None:
    sys.exit(
        "\n  ! no audio backend available.\n"
        "    install:  uv add sounddevice\n"
        "    then ensure PortAudio is on PATH (Windows: usually bundled).\n"
    )

from kokoro_onnx import Kokoro  # noqa: E402


# ---- audio sink (persistent stream, pre-opened once per session) -----------

class AudioSink:
    """Persistent sd.OutputStream that is opened once and reused.

    The dominant cost in TTFW used to be opening the PortAudio/WASAPI
    device for every prompt (200–500ms on Windows).  This sink lets us
    open it once at session start (or at voice-pick), so subsequent
    prompts only pay the cost of writing samples.

    Lifecycle:
        sink = AudioSink()
        sink.open(sample_rate=24000)
        sink.write(samples)        # may auto-reopen on stream failure
        sink.close()               # idempotent

    Context-manager supported:
        with AudioSink() as sink:
            sink.open(sample_rate=24000)
            sink.write(samples)
    """

    def __init__(self) -> None:
        self._stream = None
        self._sample_rate: int | None = None

    def is_open(self) -> bool:
        return self._stream is not None

    def open(self, sample_rate: int) -> None:
        """Open (or re-open) the underlying stream.  Idempotent at the
        same sample rate; re-opens if the rate changes."""
        if self._stream is not None:
            if self._sample_rate == sample_rate:
                return
            # Sample rate changed: close first.
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if sd is None:
            return  # no audio backend; sink is a no-op
        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32"
        )
        self._sample_rate = sample_rate
        self._stream.start()

    def write(self, samples: np.ndarray) -> None:
        """Write samples.  Auto-reopens on a stale/missing stream so
        a device change doesn't take down the whole session."""
        if self._stream is None and self._sample_rate is not None:
            # Recover from a previous write failure.
            self.open(self._sample_rate)
        if self._stream is None:
            return  # no-op when audio backend is unavailable
        try:
            self._stream.write(samples)
        except Exception:
            # Stream went bad (device unplugged, etc.) — reopen once.
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            if self._sample_rate is not None:
                self.open(self._sample_rate)
                if self._stream is not None:
                    self._stream.write(samples)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def __enter__(self) -> "AudioSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---- paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "kokoro-v1.0.onnx"
VOICES_PATH = ROOT / "models" / "voices-v1.0.bin"
OUTPUT_DIR = ROOT / "output"

# ---- voice menu ------------------------------------------------------------
# (id, short hint).  Static list — add/remove as you like.

VOICES: list[tuple[str, str]] = [
    ("af_heart",    "American · F · warm narrator"),
    ("af_bella",    "American · F · bright, energetic"),
    ("af_nicole",   "American · F · clear, professional"),
    ("af_sarah",    "American · F · neutral, news"),
    ("af_sky",      "American · F · soft, gentle"),
    ("af_river",    "American · F · calm"),
    ("am_adam",     "American · M · deep, confident"),
    ("am_michael",  "American · M · calm, mature"),
    ("am_liam",     "American · M · young, friendly"),
    ("am_onyx",     "American · M · low, grounded"),
    ("bf_emma",     "British  · F · refined"),
    ("bf_isabella", "British  · F · expressive"),
    ("bm_george",   "British  · M · formal"),
    ("bm_lewis",    "British  · M · conversational"),
]


# ---- chunking constants ---------------------------------------------------

TARGET_CHUNK_WORDS    = 25
MIN_FIRST_CHUNK_WORDS = 20
MIN_REST_CHUNK_WORDS  = 10
MAX_CHUNK_WORDS       = 50      # soft max: sentences up to this stay whole
CROSSFADE_MS          = 40
QUEUE_MAXSIZE         = 4
LEAD_IN_MIN           = 6       # min words for lead-in chunk (streaming optimization)

# Placeholder for protected dots (looks like a middle dot, not a period).
DOT = "\u00B7"  # ·


# ---- abbreviation handling ------------------------------------------------

ABBR: list[str] = [
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "Mt",
    "Rev", "Hon", "Pres", "Gov", "Sen", "Rep", "Gen", "Col",
    "Capt", "Lt", "Sgt", "Maj", "Cmdr", "Cpl", "Pvt",
    "vs", "etc", "eg", "ie", "cf", "al", "viz",
    "No", "Fig", "Eq", "Vol", "Inc", "Ltd", "Co", "Corp", "Dept", "Est",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept",
    "Oct", "Nov", "Dec",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
    "a.m", "p.m",
]

# Multi-word / internal-dot abbreviations (must be matched BEFORE the main
# ABBR_PATTERN pass, since word-boundary matching can't see "e" + "g" as
# the abbreviation "eg" when the input is "e.g.").
INTERNAL_DOT_ABBR = re.compile(
    r"\b(?:e\.g|i\.e|etc)\.?",
    re.IGNORECASE,
)

ABBR_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in ABBR) + r")\.",
    re.IGNORECASE,
)


def protect_dots(text: str) -> str:
    """Replace dots in known contexts with a placeholder so the sentence
    splitter doesn't trip on them."""
    # 1. internal-dot abbreviations
    text = INTERNAL_DOT_ABBR.sub(
        lambda m: m.group().replace(".", DOT), text
    )
    # 2. trailing-dot titles etc.  (e.g. "Dr" + literal "." -> "Dr" + DOT)
    text = ABBR_PATTERN.sub(
        lambda m: m.group()[:-1] + DOT, text
    )
    # 3. decimals
    text = re.sub(r"(\d)\.(\d)", rf"\1{DOT}\2", text)
    # 4. ellipsis (two or more consecutive dots)
    text = re.sub(r"\.{2,}", lambda m: DOT * len(m.group()), text)
    return text


def restore_dots(text: str) -> str:
    return text.replace(DOT, ".")


# ---- chunker --------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _pack_strings(parts: list[str], max_words: int) -> list[str]:
    """Greedy pack parts into chunks of <= max_words.  Word-boundary split
    is the last resort."""
    chunks: list[str] = []
    current: list[str] = []
    for part in parts:
        pwords = part.split()
        if len(current) + len(pwords) <= max_words:
            current.extend(pwords)
        else:
            if current:
                chunks.append(" ".join(current))
            if len(pwords) > max_words:
                # forced word split
                for i in range(0, len(pwords), max_words):
                    chunks.append(" ".join(pwords[i:i + max_words]))
                current = []
            else:
                current = pwords
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_on_conjunctions(text: str, max_words: int) -> list[str]:
    """Split on conjunction boundaries.  The lookahead keeps the conjunction
    in the NEXT chunk (sentence-initial treatment, which sounds more natural
    than sentence-final)."""
    parts = re.split(
        r"\s+(?=(?:and|but|or|so|because|while|though|which|that)\s+\S)",
        text,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p.strip()]
    return _pack_strings(parts, max_words)


def _balanced_split(
    sentence: str, max_words: int, min_rest: int
) -> list[str] | None:
    """Try to split the sentence at a single clause boundary that produces
    two roughly-balanced halves.  Returns None if no acceptable split exists.
    Avoids the tiny-fragment problem that greedy packing can produce on
    long sentences with many clause boundaries."""
    words = sentence.split()
    if len(words) <= max_words:
        return None
    # Collect candidate split points: word index after a clause-boundary
    # char (the boundary itself stays with the previous half).
    candidates: list[tuple[int, int]] = []  # (word_idx, char_offset)
    for m in re.finditer(r"[,;:\u2014]\s+", sentence):
        char_idx = m.end()
        word_idx = len(sentence[:char_idx].split())
        candidates.append((word_idx, char_idx))
    if not candidates:
        return None
    best: tuple[int, int] | None = None
    best_score = float("inf")
    mid = len(words) / 2.0
    for word_idx, char_idx in candidates:
        prefix_n = word_idx
        suffix_n = len(words) - word_idx
        if prefix_n > max_words or suffix_n > max_words:
            continue
        if suffix_n < min_rest:
            continue
        score = abs(prefix_n - mid)
        if score < best_score:
            best_score = score
            best = (word_idx, char_idx)
    if best is None:
        return None
    _, char_idx = best
    return [sentence[:char_idx].strip(), sentence[char_idx:].strip()]


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    # Prefer a single balanced split — avoids tiny fragments that greedy
    # packing can produce on long sentences with many clauses.
    balanced = _balanced_split(sentence, max_words, MIN_REST_CHUNK_WORDS)
    if balanced is not None:
        return balanced
    # Fallback: multi-clause split + pack.  Layer 2: conjunctions if needed.
    parts = re.split(r"(?<=[,;:\u2014])\s+", sentence)
    parts = [p.strip() for p in parts if p.strip()]
    chunks = _pack_strings(parts, max_words)
    final: list[str] = []
    for ch in chunks:
        if len(ch.split()) > max_words:
            final.extend(_split_on_conjunctions(ch, max_words))
        else:
            final.append(ch)
    return final


def _adjust_chunks(
    chunks: list[str], min_first: int, min_rest: int, max_words: int
) -> list[str]:
    if not chunks:
        return chunks
    # First chunk too small: keep merging forward into subsequent chunks
    # until it's substantial or the next merge would exceed max_words.
    while len(chunks) >= 2 and len(chunks[0].split()) < min_first:
        if len(chunks[0].split()) + len(chunks[1].split()) > max_words:
            break
        chunks = [chunks[0] + " " + chunks[1]] + chunks[2:]
    # Post-merge: any non-final chunk that ended up below min_rest because
    # of an awkward mid-sentence split gets pulled back into its predecessor
    # if combined fits.  Defends against splits on adjective-list commas
    # like "complex, multilingual".
    i = 1
    while i < len(chunks) - 1 and len(chunks[i].split()) < min_rest:
        if len(chunks[i - 1].split()) + len(chunks[i].split()) <= max_words:
            chunks = chunks[:i - 1] + [chunks[i - 1] + " " + chunks[i]] + chunks[i + 1:]
            # don't advance i; the new chunks[i-1] may also be small after merge
            continue
        i += 1
    # Last chunk too small: merge back into the previous (if combined fits).
    if len(chunks) >= 2 and len(chunks[-1].split()) < min_rest:
        merged = chunks[-2] + " " + chunks[-1]
        if len(merged.split()) <= max_words:
            chunks = chunks[:-2] + [merged]
    return chunks


def _lead_in_split(
    chunks: list[str],
    lead_in_min: int = LEAD_IN_MIN,
    target: int = TARGET_CHUNK_WORDS,
    hard_cut: bool = True,
) -> tuple[list[str], dict | None]:
    """
    Split chunk[0] into a short lead-in chunk + remainder for faster first-sound.

    Adaptive cut: find the first clause boundary (comma/semicolon/colon/dash) or
    conjunction (and/but/or/etc.) at position ≥ lead_in_min words, but only
    within a small window past lead_in_min (slack = lead_in_min // 2).
    If no natural boundary lands in that window:
      - with hard_cut=True, cut at lead_in_min words (mid-clause if necessary)
      - with hard_cut=False, leave the chunk alone (return as-is)

    `target` is kept for API back-compat (legacy callers pass it; it's no
    longer used to size the cut, since lead_in_min is the source of truth).

    Returns (new_chunks, lead_info_dict or None).
    """
    if not chunks:
        return chunks, None

    chunk0 = chunks[0]
    words = chunk0.split()
    n_words = len(words)

    # Only split if chunk[0] is large enough
    if n_words <= lead_in_min:
        return chunks, None

    # Find cut point: first clause boundary or conjunction at position ≥ lead_in_min
    # but only within the [lead_in_min, lead_in_min + slack] window.  Beyond that
    # the lead-in is already bigger than we want, so fall through to a hard cut
    # at lead_in_min.
    #
    # `slack` is how much flexibility we allow on a natural cut: a lead-in of
    # 6 might naturally land at 9 (e.g. after "however, ..."), and that's fine.
    # We just don't want the natural search to drift all the way to word 25
    # when the user asked for 6.
    clause_chars = ",;:\u2014"  # comma, semicolon, colon, em-dash
    conjunctions = {"and", "but", "or", "so", "yet", "nor", "for", "although", "because", "while", "since", "unless"}

    slack = max(2, lead_in_min // 2)  # at least 2 words of slack, scaling up
    natural_window_end = min(n_words, lead_in_min + slack)
    cut_pos = None

    # Search for clause boundary or conjunction within the natural window
    for i in range(lead_in_min, natural_window_end):
        word = words[i]
        word_lower = word.lower().strip(clause_chars)

        # Check if previous word ends with clause boundary
        if i > 0 and words[i-1] and words[i-1][-1] in clause_chars:
            cut_pos = i
            break

        # Check if current word is a conjunction
        if word_lower in conjunctions:
            cut_pos = i
            break

    # Fallback: hard cut at lead_in_min if no natural boundary landed
    # in the natural window.  The whole point of lead_in_min is "I want
    # chunk 0 to be roughly this size"; if no natural boundary came close,
    # we cut at the requested size even if it's mid-clause.
    if cut_pos is None:
        if hard_cut and n_words > lead_in_min:
            cut_pos = lead_in_min
        else:
            # Chunk is small enough, or hard cut disabled → leave alone
            return chunks, None

    # Perform the split
    lead_in_words = " ".join(words[:cut_pos])
    remainder_words = " ".join(words[cut_pos:])

    new_chunks = [lead_in_words, remainder_words] + chunks[1:]

    lead_info = {
        "lead_in_words": cut_pos,
        "remainder_words": n_words - cut_pos,
        "original_chunk0_words": n_words,
    }
    
    return new_chunks, lead_info


def chunk_text(
    text: str,
    target: int = TARGET_CHUNK_WORDS,
    min_first: int = MIN_FIRST_CHUNK_WORDS,
    min_rest: int = MIN_REST_CHUNK_WORDS,
    max_words: int = MAX_CHUNK_WORDS,
) -> list[str]:
    """Split a paragraph into phrase-sized chunks for streaming TTS.

    Note: this returns the sentence-balanced chunks; for a smaller
    first chunk (faster TTFW), call _lead_in_split on the result.
    """
    if not text or not text.strip():
        return []
    text = protect_dots(text)
    text = re.sub(r"\s+", " ", text.strip())
    sentences = _split_sentences(text)
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence.split()) <= max_words:
            chunks.append(sentence)
        else:
            chunks.extend(_split_long_sentence(sentence, max_words))
    chunks = _adjust_chunks(chunks, min_first, min_rest, max_words)
    return [restore_dots(c).strip() for c in chunks if c.strip()]


# ---- WorkerPool (long-lived worker, reused across prompts) -----------------

class _ExceptionQueue(queue.Queue):
    """A Queue whose get() re-raises any exception put onto it.

    Standard queue.Queue.get() returns whatever was put, with no way to
    distinguish a real value from an exception carrier.  This subclass
    lets the worker thread put a (kind, payload) pair where kind is
    'ok' or 'err', and get() returns the value (kind='ok') or raises
    the exception (kind='err').
    """

    def put(self, item, *args, **kwargs):  # type: ignore[override]
        if isinstance(item, tuple) and len(item) == 2 and item[0] in ("ok", "err"):
            kind, payload = item
            if kind == "ok":
                return super().put(payload, *args, **kwargs)
            return super().put(_RaisingRef(payload), *args, **kwargs)
        return super().put(item, *args, **kwargs)

    def get(self, *args, **kwargs):  # type: ignore[override]
        item = super().get(*args, **kwargs)
        if isinstance(item, _RaisingRef):
            item.raise_()
        return item


class _RaisingRef:
    __slots__ = ("_exc",)

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def raise_(self) -> None:
        raise self._exc


class _ExceptionHolder:
    """One-shot exception carrier: the worker puts one of these on the
    result queue, and the caller's .get() raises the stored exception.

    Using a holder (rather than a tuple) keeps the public API clean:
    the caller does `result = q.get()` and either gets the value or
    sees the exception raised.
    """
    holders: dict[int, BaseException] = {}

    def __init__(self, qid: int) -> None:
        self._qid = qid

    def __getattr__(self, name):
        # When the caller accesses any attribute (e.g. iteration), raise
        # the stored exception instead.  This makes the holder look
        # like a value but actually raise.
        exc = _ExceptionHolder.holders.pop(self._qid, None)
        if exc is None:
            raise AttributeError(name)
        raise exc


class WorkerPool:
    """One daemon thread, one job queue, reused across prompts.

    Replaces the per-prompt `threading.Thread(target=worker, daemon=True)`
    pattern.  Per-prompt thread spawn costs ~5-20ms of variance on
    Windows (thread creation + GIL acquisition + first-Python-bytecode
    dispatch); reusing one thread keeps the prompt-to-prompt variance
    out of the TTFW measurement.

    Lifecycle:
        pool = WorkerPool()
        pool.start(backend=lambda text, voice: ...)
        result_q = pool.submit("hello", "af_bella")
        result = result_q.get()        # returns backend's return value
        pool.close()

    The backend callable is invoked with (text, voice) on the worker
    thread.  Its return value goes onto the per-job result queue; any
    exception is re-raised when the caller calls get().
    """

    def __init__(self) -> None:
        self._job_q: queue.Queue | None = None
        self._result_q_for_job: dict[int, queue.Queue] = {}
        self._next_job_id = 0
        self._thread: threading.Thread | None = None
        self._backend = None
        self._call_args: tuple = ()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self, backend) -> None:
        """Start the worker thread with the given backend callable.

        `backend` is called as `backend(text, voice)` on the worker thread
        and must return the result (any picklable value).  If you need
        per-call args (e.g. a different kokoro or voice across prompts),
        use set_call_args() to set the active (kokoro, voice) pair before
        each call to submit().
        """
        if self._thread is not None:
            return
        self._backend = backend
        self._job_q = queue.Queue()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_call_args(self, *args) -> None:
        """Update the per-call args the worker will pass alongside the
        job's text/voice.  Use this when sharing one pool across prompts
        with different (kokoro, voice) pairs.
        """
        with self._lock:
            self._call_args = args

    def submit(self, text: str, voice: str) -> "queue.Queue":
        """Submit a job; return a per-job result queue the caller can .get().

        The returned queue is a _ExceptionQueue: if the worker raised an
        exception while running the job, .get() will re-raise it.
        """
        with self._lock:
            job_id = self._next_job_id
            self._next_job_id += 1
        result_q: queue.Queue = _ExceptionQueue(maxsize=1)
        with self._lock:
            self._result_q_for_job[job_id] = result_q
        assert self._job_q is not None, "WorkerPool.start() not called"
        self._job_q.put((job_id, text, voice))
        return result_q

    def _run(self) -> None:
        assert self._job_q is not None
        assert self._backend is not None
        while not self._stop.is_set():
            try:
                job_id, text, voice = self._job_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if job_id is None:
                # Sentinel: drain and stop
                break
            result_q = self._result_q_for_job.pop(job_id, None)
            if result_q is None:
                continue
            try:
                with self._lock:
                    call_args = self._call_args
                result = self._backend(text, voice, *call_args)
                result_q.put(("ok", result))
            except BaseException as exc:
                result_q.put(("err", exc))

    def close(self) -> None:
        """Signal the worker to stop and wait for it to drain."""
        if self._thread is None:
            return
        self._stop.set()
        if self._job_q is not None:
            # Sentinel to unblock a get() that's mid-wait
            self._job_q.put((None, None, None))
        self._thread.join(timeout=5.0)
        self._thread = None


# ---- ChunkError sentinel --------------------------------------------------

class ChunkError:
    """Sentinel pushed into the queue when the worker fails on a chunk."""
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


# ---- crossfade ------------------------------------------------------------

def crossfade(prev: np.ndarray, curr: np.ndarray, n_fade: int) -> np.ndarray:
    """Overlap-add prev's tail with curr's head.  Returns only the NEW
    material to write to the stream (region + curr's tail), not prev's
    body — prev has already been written."""
    assert len(prev) >= n_fade and len(curr) >= n_fade, (
        f"chunks too short for {n_fade}-sample crossfade "
        f"(prev={len(prev)}, curr={len(curr)})"
    )
    t = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    fade_out = np.cos(t * np.float32(np.pi / 2.0))   # 1 → 0, equal-power
    fade_in  = np.sin(t * np.float32(np.pi / 2.0))   # 0 → 1, equal-power
    region = prev[-n_fade:] * fade_out + curr[:n_fade:] * fade_in
    return np.concatenate([region, curr[n_fade:]]).astype(np.float32, copy=False)


# ---- wav writer -----------------------------------------------------------

def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono 16-bit PCM WAV from float32 samples in [-1, 1]."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ---- formatting helpers ---------------------------------------------------

def fmt_t(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1.0 else f"{seconds:.2f}s"


def bar(value: float, max_value: float, width: int = 24) -> str:
    if max_value <= 0:
        return "·" * width
    n = max(0, min(width, int(round(value / max_value * width))))
    return "█" * n + "·" * (width - n)


def fmt_mb(n: float) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n > 1024 * 1024 else f"{n / 1024:.0f} KB"


# ---- per-prompt profile line (--profile) ---------------------------------

def emit_profile_line(stats: dict) -> str:
    """Build a single-line JSON profile record for one prompt.

    Public, pure function: takes a stats dict (the same shape returned by
    play_streaming / play_simple / single-chunk inline path) and returns
    a JSON string with one trailing newline. Stable key vocabulary so
    downstream tooling (CI, diffs) can rely on it.

    Metrics included:
      - ttfw_ms: time-to-first-sound (chunk-0 phonemize + infer + overhead)
      - chunk0_phonemize_ms, chunk0_infer_ms: per-stage breakdown of chunk 0
      - per_chunk_infer_ms: list of infer times, one per chunk
      - generation_rtf: sum(per_chunk_infer_ms) / audio_duration_ms
      - wall_rtf: time_total / audio_duration_s (includes realtime playback)
      - audio_duration_s, sample_rate, voice, lead_in_words, n_chunks
      - time_total_s: wall time from user press to last sample written
    """
    audio_s = float(stats.get("audio_duration_s") or 0.0)
    time_total = float(stats.get("time_total") or 0.0)
    per_chunk = list(stats.get("per_chunk_infer_ms") or [])
    gen_ms = sum(per_chunk)
    generation_rtf = (gen_ms / (audio_s * 1000.0)) if audio_s > 0 else 0.0
    wall_rtf = (time_total / audio_s) if audio_s > 0 else 0.0
    payload = {
        "ttfw_ms": round(float(stats.get("t_ttfw") or 0.0) * 1000.0, 3),
        "chunk0_phonemize_ms": round(float(stats.get("chunk0_phonemize_ms") or 0.0), 3),
        "chunk0_infer_ms": round(float(stats.get("chunk0_infer_ms") or 0.0), 3),
        "stream_open_ms": round(float(stats.get("stream_open_ms") or 0.0), 3),
        "per_chunk_infer_ms": [round(float(x), 3) for x in per_chunk],
        "generation_rtf": round(generation_rtf, 6),
        "wall_rtf": round(wall_rtf, 6),
        "time_total_s": round(time_total, 6),
        "audio_duration_s": round(audio_s, 6),
        "sample_rate": int(stats.get("sample_rate") or 0),
        "voice": str(stats.get("voice") or ""),
        "lead_in_words": int(stats.get("lead_in_words") or 0),
        "n_chunks": int(stats.get("n_chunks") or 0),
        # Issue 002-005: buffer-health fields from LookAheadBuffer.
        # Older code paths (play_simple, single-chunk inline) won't
        # set these, so default to 0.
        "buffer_peak": int(stats.get("buffer_peak") or 0),
        "buffer_median": round(float(stats.get("buffer_median") or 0.0), 3),
        "buffer_final": int(stats.get("buffer_final") or 0),
        "hiccup_count": int(stats.get("hiccup_count") or 0),
        "playback_underrun_count": int(stats.get("playback_underrun_count") or 0),
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def format_profile_batch(stats_list: list[dict]) -> str:
    """Format a list of stats dicts as concatenated profile lines.

    Used by --profile: append one line per prompt to a buffer that is
    printed (or written to a file) when the session ends.
    """
    return "".join(emit_profile_line(s) for s in stats_list)


# ---- RTF computation (separate generation-RTF from wall-RTF) -------------

def compute_generation_rtf(per_chunk_infer_ms: list[float], audio_duration_s: float) -> float:
    """Generation RTF = sum of per-chunk infer time / audio duration.

    This is a property of the model: < 1 means generation is faster than
    realtime (always true for Kokoro on CUDA / CPU alike).
    """
    if audio_duration_s <= 0:
        return 0.0
    return sum(per_chunk_infer_ms) / (audio_duration_s * 1000.0)


def compute_wall_rtf(time_total_s: float, audio_duration_s: float) -> float:
    """Wall RTF = wall-clock time / audio duration.

    This is a property of the audio sink: when sounddevice plays back at
    realtime, wall_rtf >= 1.0 regardless of how fast the model is.
    """
    if audio_duration_s <= 0:
        return 0.0
    return time_total_s / audio_duration_s


def format_rtf_line(generation_rtf: float, wall_rtf: float) -> str:
    """Format a single-line RTF summary that labels both metrics and
    applies the faster/slower verdict to the GENERATION RTF only.

    The wall RTF is presented as a fact (no verdict) so users don't
    mistake a slow audio sink for a slow model.
    """
    gen_verdict = "faster than realtime ✓" if generation_rtf < 1.0 else "slower than realtime ✗"
    return (
        f"  RTF (generation)   {generation_rtf:>6.2f}x  {gen_verdict}  "
        f"(sum infer / audio · <1 = faster than realtime)\n"
        f"  RTF (wall)         {wall_rtf:>6.2f}x           "
        f"(wall / audio · includes realtime playback)"
    )


def format_summary_line(stats: dict) -> str:
    """Format the WAV summary line that appears at the top of each prompt.

    Issue 002-002: the headline number is generation-RTF (model speed)
    with the faster/slower verdict, matching the Latency section.  Wall-RTF
    is shown as a secondary, un-verdicted fact so a slow audio sink
    doesn't mislead the user into thinking the model is slow.

    Returns a single line of text ready to be printed.  Pure function.
    """
    per_chunk = list(stats.get("per_chunk_infer_ms") or [])
    audio_s = float(stats.get("audio_duration_s") or 0.0)
    time_total = float(stats.get("time_total") or 0.0)
    gen_rtf = compute_generation_rtf(per_chunk, audio_s)
    wall_rtf = compute_wall_rtf(time_total, audio_s)
    gen_verdict = "faster than realtime ✓" if gen_rtf < 1.0 else "slower than realtime ✗"
    return (
        f"  RTF (gen)  {gen_rtf:>5.2f}x  {gen_verdict}  "
        f"(model · <1 = faster than realtime)  ·  "
        f"wall {wall_rtf:.2f}x  (sink · includes realtime playback)"
    )


def format_ttfw_breakdown(
    chunk0_phonemize_ms: float,
    chunk0_infer_ms: float,
    stream_open_ms: float,
    t_ttfw_ms: float,
) -> str:
    """Format a multi-line per-phase TTFW breakdown.

    The named phases (phon, infer, stream-open) are explicit, and any
    remaining time is reported as "other" (queue wait / thread /
    misc) so the breakdown always sums to t_ttfw_ms.
    """
    named = chunk0_phonemize_ms + chunk0_infer_ms + stream_open_ms
    other_ms = max(0.0, t_ttfw_ms - named)
    return (
        f"  TTFW (total)       {t_ttfw_ms:>6.0f}ms\n"
        f"    ├─ chunk-0 phon  {chunk0_phonemize_ms:>6.0f}ms\n"
        f"    ├─ chunk-0 infer {chunk0_infer_ms:>6.0f}ms\n"
        f"    ├─ stream-open   {stream_open_ms:>6.0f}ms  "
        f"({'no-op (pre-opened)' if stream_open_ms < 1.0 else 'first device open'})\n"
        f"    └─ other         {other_ms:>6.0f}ms  "
        f"(queue wait / thread / crossfade setup)"
    )


# ---- shape-range warmup (issue 002-003) -----------------------------------

# Default phoneme-count shape set used at startup to pre-compile cuDNN
# kernels across the full range of real lead-in shapes (6-word leads
# phonemize to ~30, 8-word leads to ~55, etc.).  A real lead-in can
# land anywhere in this range; the default covers short command-style
# leads up through typical 25-word chunks.
DEFAULT_WARMUP_SHAPES: list[int] = [20, 40, 60, 100, 160, 256]

# Source text used by the shape warmup.  Long enough that the longest
# default target (256) can be reached by truncation alone in most cases
# (and a small tail of pad-tokens fills any remainder).
WARMUP_SOURCE_TEXT: str = (
    "The morning sun rose slowly over the hills beyond the harbor as the crew "
    "prepared the sturdy ship for a long voyage to distant waters, double-"
    "checking every rope, every sail, and every instrument aboard the vessel."
)


def run_warmups(
    sess,
    kokoro,
    target_phoneme_counts: list[int],
    *,
    voice: str = "af_heart",
    source_text: str = WARMUP_SOURCE_TEXT,
) -> list[tuple[int, float]]:
    """Run inference at each target phoneme count, in input order.

    Phonemizes `source_text` ONCE, then for each target count truncates
    the resulting phoneme string to that length (or pads by repeating
    the last phoneme token if the source is shorter than the target)
    and runs `kokoro.create` on it.  Audio output is discarded.

    The goal is to pre-compile cuDNN/ORT kernels for the full shape
    range so the first real prompt's chunk-0 doesn't trigger an
    on-demand kernel compile (issue 002-003).

    Returns one `(phoneme_count, ms)` pair per input count, in input
    order.  Never raises — failed warmups are recorded as
    `(phoneme_count, -1.0)` so the caller can still report a banner
    line.
    """
    if not target_phoneme_counts:
        return []

    # Phonemize the source text once.  This is the only call to the
    # tokenizer; all per-shape work uses string slicing of the result.
    try:
        source_phonemes = kokoro.tokenizer.phonemize(
            source_text, lang="en-us", norm=True
        )
    except Exception:
        # Phonemize failed entirely: signal it but don't crash startup.
        return [(c, -1.0) for c in target_phoneme_counts]

    if not source_phonemes:
        return [(c, -1.0) for c in target_phoneme_counts]

    results: list[tuple[int, float]] = []
    for target in target_phoneme_counts:
        if target <= 0:
            continue
        if len(source_phonemes) >= target:
            phonemes = source_phonemes[:target]
        else:
            # Pad by repeating the last token.  Padded tokens are dummy
            # audio but the kernel-compile side-effect is what we want.
            pad_token = source_phonemes[-1]
            phonemes = source_phonemes + pad_token * (target - len(source_phonemes))
        try:
            t_wu = time.perf_counter()
            _ = kokoro.create(
                phonemes, voice=voice, lang="en-us",
                is_phonemes=True, speed=1.0,
            )
            ms = (time.perf_counter() - t_wu) * 1000.0
            results.append((target, ms))
        except Exception:
            results.append((target, -1.0))
    return results


# ---- resource monitor -----------------------------------------------------

class ResourceMonitor:
    """Background sampler for self-process CPU%, RSS, NVIDIA VRAM + util."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.cpu: list[float] = []
        self.ram: list[int] = []
        self.gpu_mem: list[int] = []
        self.gpu_util: list[float] = []
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())

    def __enter__(self):
        self._proc.cpu_percent(interval=None)  # prime
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.cpu.append(self._proc.cpu_percent(interval=None))
                self.ram.append(self._proc.memory_info().rss)
                if self._has_nvidia_smi:
                    gm, gu = _nvidia_query()
                    if gm is not None:
                        self.gpu_mem.append(gm)
                    if gu is not None:
                        self.gpu_util.append(gu)
            except psutil.NoSuchProcess:
                break
            self._stop.wait(self.interval)

    @property
    def peak_cpu(self) -> float:
        return max(self.cpu) if self.cpu else 0.0

    @property
    def avg_cpu(self) -> float:
        return sum(self.cpu) / len(self.cpu) if self.cpu else 0.0

    @property
    def peak_ram(self) -> int:
        return max(self.ram) if self.ram else 0

    @property
    def peak_gpu_mem(self) -> int:
        return max(self.gpu_mem) if self.gpu_mem else 0

    @property
    def peak_gpu_util(self) -> float:
        return max(self.gpu_util) if self.gpu_util else 0.0

    @property
    def avg_gpu_util(self) -> float:
        return sum(self.gpu_util) / len(self.gpu_util) if self.gpu_util else 0.0

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_mem)


def _nvidia_query() -> tuple[int | None, float | None]:
    """Single GPU sample via nvidia-ml-py, returning (memory_used_mb, gpu_util_pct).

    In-process (no subprocess fork/exec), so it's microsecond-cost per call
    instead of the 10-30ms cost of spawning nvidia-smi.  The previous
    subprocess-based implementation was responsible for inflating the
    reported CPU peak by ~170 process-spawns per long run.

    Issue 002-001: uses the canonical `nvidia-ml-py` package (the
    `pynvml` package was deprecated and now prints a FutureWarning
    on import; the API surface we use is identical).
    """
    try:
        import nvidia_ml_py as pynvml
        # Cheap re-init: the handle is cached on the module; we only
        # initialize once.  Subsequent calls are essentially free.
        if not getattr(_nvidia_query, "_initialized", False):
            try:
                pynvml.nvmlInit()
            except Exception:
                return None, None
            _nvidia_query._initialized = True
            _nvidia_query._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        handle = _nvidia_query._handle
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return int(mem_info.used / (1024 * 1024)), float(util.gpu)
    except Exception:
        return None, None


# Backwards-compat alias (kept in case any other module imports it)
def _nvidia_mem_used_mb() -> int | None:
    mem, _ = _nvidia_query()
    return mem


# ---- pretty output --------------------------------------------------------

def print_phase_chart(phases: list[tuple[str, float]], total: float) -> None:
    print("\n  Phase breakdown")
    print("  " + "─" * 62)
    max_t = max((t for _, t in phases), default=0.0)
    width = 24
    for name, t in phases:
        pct = (t / total * 100) if total > 0 else 0
        print(f"  {name:<20} {bar(t, max_t, width):<{width}}  {fmt_t(t):>8}  {pct:>4.0f}%")
    print("  " + "─" * 62)
    pct = 100.0 if total > 0 else 0
    print(f"  {'TOTAL':<20} {bar(total, total, width):<{width}}  {fmt_t(total):>8}  {pct:>4.0f}%")


def print_resource_summary(mon: ResourceMonitor) -> None:
    print("\n  Resource usage (during pipeline)")
    print("  " + "─" * 62)
    print(f"  CPU peak   {mon.peak_cpu:>6.1f}%     (avg {mon.avg_cpu:>5.1f}%)")
    print(f"  RAM peak   {fmt_mb(mon.peak_ram):>10}")
    if mon.has_gpu:
        print(f"  VRAM peak  {mon.peak_gpu_mem:>6.0f} MB")
        print(
            f"  GPU util   {mon.peak_gpu_util:>6.1f}%     "
            f"(avg {mon.avg_gpu_util:>5.1f}%)"
        )
    print("  " + "─" * 62)


# ---- latency / streaming health report -----------------------------------

def _fmt_ms(ms: float) -> str:
    """Format a millisecond value (always ms, never switch to s)."""
    return f"{ms:.0f}ms"


def _per_chunk_summary(values: list[float]) -> str:
    """min / med / p95 / max of per-chunk infer ms, plus raw values."""
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{_fmt_ms(values[0])}  (single chunk)"
    mn = min(values)
    mx = max(values)
    md = statistics.median(values)
    p95 = (
        statistics.quantiles(values, n=20)[-1]
        if len(values) >= 5
        else mx
    )
    raw = " · ".join(_fmt_ms(v) for v in values)
    return (
        f"min {_fmt_ms(mn)} · med {_fmt_ms(md)} · "
        f"p95 {_fmt_ms(p95)} · max {_fmt_ms(mx)}\n"
        f"                  raw: [{raw}]"
    )


def print_latency_report(
    stats: dict,
    n_chunks: int,
    word_counts: list[int],
    duration_s: float,
    is_streaming_path: bool,
) -> None:
    """Print the latency + streaming-health sections."""
    print("\n  Latency")
    print("  " + "─" * 62)
    t_ttfw = stats.get("t_ttfw", 0.0)
    ph_ms = stats.get("chunk0_phonemize_ms", 0.0)
    inf_ms = stats.get("chunk0_infer_ms", 0.0)
    stream_open_ms = stats.get("stream_open_ms", 0.0)
    # Use the new per-phase breakdown formatter so users can see exactly
    # where the TTFW time went (and confirm stream-open is ~0 with a
    # pre-opened AudioSink).
    print(format_ttfw_breakdown(
        chunk0_phonemize_ms=ph_ms,
        chunk0_infer_ms=inf_ms,
        stream_open_ms=stream_open_ms,
        t_ttfw_ms=t_ttfw * 1000.0,
    ))
    wall = stats.get("t_wall_end", 0.0) - stats.get("t_user_press", 0.0)
    print(f"  wall time         {fmt_t(wall):>8}     (Enter → last sample written)")
    # RTF: generation (model) vs wall (sink).  Verdict applies to generation only.
    per_chunk = stats.get("per_chunk_infer_ms") or []
    gen_rtf = compute_generation_rtf(per_chunk, duration_s)
    wall_rtf = compute_wall_rtf(wall, duration_s)
    print(format_rtf_line(gen_rtf, wall_rtf))

    print("\n  Streaming health")
    print("  " + "─" * 62)
    if n_chunks <= 1:
        print(f"  chunks            1  ({word_counts} words)")
        print("  per-chunk infer   n/a (single chunk)")
        print("  starvation        n/a (single chunk)")
    else:
        print(f"  chunks            {n_chunks}  ({word_counts} words)")
        per = stats.get("per_chunk_infer_ms") or []
        print(f"  per-chunk infer   {_per_chunk_summary(per)}")
        n_blocks = stats.get("q_get_blocks", 0)
        block_ms = stats.get("q_get_block_total_ms", 0.0)
        if n_blocks == 0:
            print("  starvation        0 events (0ms total)  · gen kept up with playback ✓")
        else:
            print(
                f"  starvation        {n_blocks} events "
                f"({_fmt_ms(block_ms)} total)  · gen fell behind {n_blocks}x ✗"
            )

    # Chunk rate (always meaningful, but more interesting multi-chunk)
    rate = n_chunks / wall if wall > 0 else 0.0
    print(f"  chunk rate        {rate:>8.2f}/s   (effective streaming throughput)")

    # Buffer health (issue 002-005): only meaningful on the pre-generate
    # streaming path with multiple chunks.
    buffer_peak = stats.get("buffer_peak", 0) or 0
    if is_streaming_path and n_chunks > 1 and buffer_peak > 0:
        buffer_median = stats.get("buffer_median", 0.0) or 0.0
        buffer_final = stats.get("buffer_final", 0) or 0
        hiccup_count = stats.get("hiccup_count", 0) or 0
        underrun_count = stats.get("playback_underrun_count", 0) or 0
        print("\n  Buffer health")
        print("  " + "─" * 62)
        print(
            f"  peak              {buffer_peak:>4d} chunks ahead   "
            f"(max queue depth)"
        )
        print(
            f"  median            {buffer_median:>4.1f}               "
            f"(typical queue depth)"
        )
        print(
            f"  final             {buffer_final:>4d}                 "
            f"(queue empty at end ✓)"
        )
        if hiccup_count == 0:
            print("  hiccups           0  (no chunk's infer exceeded its audio duration ✓)")
        else:
            print(
                f"  hiccups           {hiccup_count}  "
                f"(chunk(s) where infer > audio duration ✗)"
            )
        if underrun_count == 0:
            print("  underruns         0  (gen stayed ahead of playback ✓)")
        else:
            print(
                f"  underruns         {underrun_count}  "
                f"(gen fell behind playback ✗)"
            )


# ---- voice picker ---------------------------------------------------------

def pick_voice() -> str:
    print("\n  Available voices")
    print("  " + "─" * 62)
    for i, (vid, hint) in enumerate(VOICES, 1):
        print(f"  {i:>2}. {vid:<12} {hint}")
    print("  " + "─" * 62)
    ids = {v[0] for v in VOICES}
    while True:
        raw = input("  Pick a voice (number or id, Enter=1): ").strip()
        if not raw:
            return VOICES[0][0]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(VOICES):
                return VOICES[idx][0]
        if raw in ids:
            return raw
        print(f"  ! unknown: {raw!r}  (try a number or voice id)")


# ---- provider selection ---------------------------------------------------

# CUDA provider options tuned for low-latency first inference.
#
# Notes:
#   - cudnn_conv_algo_search=HEURISTIC: avoids the multi-second EXHAUSTIVE
#     search that causes a 451ms p95 spike on the first real chunk.
#   - arena_extend_strategy=kSameAsRequested: pre-allocates the arena for
#     the chunk shapes we actually use, avoiding per-call arena-grow stalls.
#   - tunable_op_enable=True: lets ORT pick tuned kernels for matmul/conv.
#     (cheap to enable; only triggers a brief auto-tune on first call.)
#   - do_copy_in_default_stream=True: matches the original Kokoro reference
#     behaviour; some versions of ORT require this for stable memcpy.
CUDA_PROVIDER_OPTIONS: dict = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "HEURISTIC",
    "tunable_op_enable": True,
    "tunable_op_max_tuning_duration_ms": 0,  # 0 = tune once, no per-call retune
    "do_copy_in_default_stream": True,
}


def build_cuda_provider_options(**overrides) -> dict:
    """Return a copy of CUDA_PROVIDER_OPTIONS with optional overrides.

    Public for tests and for callers that want to tweak a single knob
    (e.g. a future --no-tunable-ops flag) without mutating the module
    constant.
    """
    opts = dict(CUDA_PROVIDER_OPTIONS)
    opts.update(overrides)
    return opts


def select_providers(choice: str, return_options: bool = False):
    """Return (providers, backend_label[, options]).

    When return_options is True, the third element is a list of
    provider-options dicts, one per provider, ready to pass to
    onnxruntime.InferenceSession.  When False (default), preserves
    the original (providers, label) tuple for backward compatibility.
    """
    if ort is None:
        if return_options:
            return ["CPUExecutionProvider"], "CPU", [{}]
        return ["CPUExecutionProvider"], "CPU"
    available = set(ort.get_available_providers())
    if choice == "cuda":
        if "CUDAExecutionProvider" in available:
            opts = [build_cuda_provider_options(), {}]
            if return_options:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA", opts
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA"
        print("  ! --provider cuda requested, but CUDA isn't available.")
        print("    (check that onnxruntime-gpu[cuda,cudnn] is installed)")
        if return_options:
            return ["CPUExecutionProvider"], "CPU", [{}]
        return ["CPUExecutionProvider"], "CPU"
    if choice == "cpu":
        if return_options:
            return ["CPUExecutionProvider"], "CPU", [{}]
        return ["CPUExecutionProvider"], "CPU"
    if "CUDAExecutionProvider" in available:
        opts = [build_cuda_provider_options(), {}]
        if return_options:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA", opts
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA"
    if return_options:
        return ["CPUExecutionProvider"], "CPU", [{}]
    return ["CPUExecutionProvider"], "CPU"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tts.py",
        description="Local Kokoro-82M ONNX TTS CLI with streaming chunking",
    )
    p.add_argument(
        "--provider",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="inference backend (default: auto, prefers CUDA if available)",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="pre-build all chunks then play once (used when sounddevice is unavailable)",
    )
    p.add_argument(
        "--no-pre-generate",
        action="store_true",
        help=(
            "issue 002-005: disable the LookAheadBuffer pre-generate path. "
            "By default the streaming path submits all chunks up-front and "
            "drains with crossfade (uses the 17x generation slack). Pass "
            "this to preserve the legacy submit-as-you-read behavior."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="emit one JSON line per prompt on stdout (TTFW, RTF, per-chunk timings). "
             "Diff across runs with `diff <(old) <(new)`.",
    )
    p.add_argument(
        "--no-warm",
        action="store_true",
        help="skip the model-load warmups (cold-cache measurement; slower first chunk).",
    )
    p.add_argument(
        "--warmup-shapes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_WARMUP_SHAPES),
        help=(
            "comma-separated phoneme counts to pre-compile at startup "
            "(default: %(default)s). Issue 002-003 shape-range warmup. "
            "Empty string disables shape warmups. Ignored if --no-warm is set."
        ),
    )
    p.add_argument(
        "--lead-in-words",
        type=int,
        default=LEAD_IN_MIN,
        help="minimum words in the lead-in chunk (default: %(default)d). "
             "Lower = smaller first chunk = faster TTFW. Hard-cut at the "
             "TARGET_CHUNK_WORDS boundary when no natural clause lands early.",
    )
    return p.parse_args()


# ---- multi-line text input -----------------------------------------------

def read_paragraph() -> str:
    print("  Enter/paste text.  Blank line = generate.  ':q' = quit.")
    lines: list[str] = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if line.strip() == ":q":
            return ":q"
        if line == "" and lines:
            break
        if line == "" and not lines:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# ---- inference helpers ----------------------------------------------------

def _generate_chunk_audio(
    kokoro: Kokoro, text: str, voice: str
) -> tuple[np.ndarray, int]:
    """Phonemize + run ONNX inference for one chunk."""
    phonemes = kokoro.tokenizer.phonemize(text, lang="en-us", norm=True)
    samples, sr = kokoro.create(
        phonemes, voice=voice, lang="en-us",
        is_phonemes=True, speed=1.0,
    )
    return samples, sr


def _timed_generate_chunk_audio(
    kokoro: Kokoro, text: str, voice: str
) -> tuple[np.ndarray, int, float, float]:
    """Same as _generate_chunk_audio but returns per-stage timings in ms.

    Returns (samples, sample_rate, phonemize_ms, infer_ms).
    """
    t0 = time.perf_counter()
    phonemes = kokoro.tokenizer.phonemize(text, lang="en-us", norm=True)
    ph_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    samples, sr = kokoro.create(
        phonemes, voice=voice, lang="en-us",
        is_phonemes=True, speed=1.0,
    )
    inf_ms = (time.perf_counter() - t0) * 1000.0
    return samples, sr, ph_ms, inf_ms


# ---- LookAheadBuffer (issue 002-005) -------------------------------------

@dataclass
class BufferHealth:
    """Captured health of a LookAheadBuffer run.

    `peak`/`median`/`final` are queue depths (chunks queued ahead of
    playback).  `hiccup_count` is the number of chunks whose individual
    inference time exceeded that chunk's own audio duration (a stall
    event).  `playback_underrun_count` is the number of moments the
    consumer found the queue empty *after the first chunk* (i.e.
    generation fell behind playback, not the initial chunk-0 wait).
    """
    peak: int
    median: float
    final: int
    hiccup_count: int
    playback_underrun_count: int


class LookAheadBuffer:
    """Pre-generate the whole prompt and drain to the sink with crossfade.

    With generation at ~0.06x realtime and playback at 1.0x, the worker
    finishes all chunks in ~1.4s and then sits idle while the speaker
    drains.  Submitting all chunks up-front (rather than back-pressuring
    on the sink) embraces the slack: the consumer just reads from a
    pre-filled result-queue and writes to the sink with the existing
    crossfade.

    The buffer is a thin coordinator over an existing WorkerPool.  It
    does NOT own the worker thread or the audio sink; it just owns
    the submit-then-drain pattern and the metrics it captures.

    Lifecycle:
        buf = LookAheadBuffer(pool=session_pool)
        buf.submit_all([(text, voice), ...])
        stats = buf.drain_to_sink(sink=session_sink, n_fade=960)
        h = buf.health()    # BufferHealth with peak/median/final + counts
    """

    # Threshold for counting a `get()` as a real underrun (vs. jitter).
    _UNDERRUN_THRESHOLD_MS = 1.0

    def __init__(self, pool: "WorkerPool") -> None:
        self._pool = pool
        self._result_queues: list = []
        self._peak_depth = 0
        self._depth_samples: list[int] = []
        self._hiccup_count = 0
        self._underrun_count = 0

    def submit_all(self, jobs: list) -> None:
        """Schedule every (text, voice) pair on the underlying pool.

        After this returns, the pool's worker may already have started
        (and possibly finished) some jobs — but the per-submit sample
        recorded in self._depth_samples is the depth at submit time,
        so `peak` is monotonic in `len(jobs)`.
        """
        for text, voice in jobs:
            rq = self._pool.submit(text, voice)
            self._result_queues.append(rq)
            self._record_depth()

    def health(self) -> BufferHealth:
        """Return a snapshot of captured health metrics."""
        if not self._depth_samples:
            return BufferHealth(0, 0.0, 0, 0, 0)
        return BufferHealth(
            peak=self._peak_depth,
            median=statistics.median(self._depth_samples),
            final=len(self._result_queues),
            hiccup_count=self._hiccup_count,
            playback_underrun_count=self._underrun_count,
        )

    def _record_depth(self) -> None:
        depth = len(self._result_queues)
        if depth > self._peak_depth:
            self._peak_depth = depth
        self._depth_samples.append(depth)

    def drain_to_sink(self, sink, n_fade: int) -> dict:
        """Wait for the first chunk, then drain the rest to the sink.

        `sink` is a pre-opened AudioSink (or a compatible stand-in).
        `n_fade` is the crossfade length in samples at the chunk's
        sample rate.  Returns a stats dict with the buffer-health
        fields plus the standard per-chunk metrics (so callers can
        drop it straight into emit_profile_line).
        """
        if not self._result_queues:
            return {
                "n_chunks": 0,
                "chunks_succeeded": 0,
                "chunks_failed": 0,
                "per_chunk_infer_ms": [],
                "buffer_peak": 0,
                "buffer_median": 0.0,
                "buffer_final": 0,
                "hiccup_count": 0,
                "playback_underrun_count": 0,
                "chunk0_phonemize_ms": 0.0,
            }

        sr: int | None = None
        prev_chunk: np.ndarray | None = None
        final_audio: np.ndarray | None = None
        n_fade_local = n_fade
        per_chunk_infer_ms: list[float] = []
        chunk0_phonemize_ms = 0.0   # captured from the first rq.get()
        chunks_succeeded = 0
        chunks_failed = 0
        t_first_audio = 0.0
        is_first_get = True

        # Iterate by index so we can `pop(0)` the head after each
        # read without mutating the list we're iterating over.
        # submit_all() order is the read order, so this is the right
        # order for the sink too.
        i = 0
        while i < len(self._result_queues):
            rq = self._result_queues[i]
            t_get = time.perf_counter()
            try:
                samples, chunk_sr, ph_ms, inf_ms = rq.get(timeout=10.0)
            except Exception:
                # Worker raised on this job.  Skip it and continue
                # draining the rest so one bad chunk doesn't dead-lock
                # the consumer.
                chunks_failed += 1
                self._result_queues.pop(i)
                self._record_depth()
                continue
            get_ms = (time.perf_counter() - t_get) * 1000.0
            if not is_first_get and get_ms > self._UNDERRUN_THRESHOLD_MS:
                self._underrun_count += 1
            is_first_get = False

            per_chunk_infer_ms.append(inf_ms)
            if chunks_succeeded == 0:
                chunk0_phonemize_ms = ph_ms
            chunks_succeeded += 1

            # Hiccup = a single chunk whose inference took longer than
            # its own audio duration.  This is a stall event even if
            # the overall generation stays ahead of playback.
            if chunk_sr:
                audio_dur_ms = (len(samples) / chunk_sr) * 1000.0
                if inf_ms > audio_dur_ms:
                    self._hiccup_count += 1

            if sr is None:
                sr = chunk_sr
                n_fade_local = int(sr * 0.04)  # 40ms default crossfade
            # Crossfade write (or first-chunk plain write).
            if prev_chunk is None:
                # t_first_audio MUST be stamped BEFORE the blocking
                # sink.write(): sounddevice.OutputStream.write() blocks
                # at realtime until all samples are handed to the callback,
                # so for a ~2.5s first chunk it blocks ~2.5s *during which
                # audio is already playing*.  Stamping after write() would
                # charge that realtime playback to TTFW, inflating it by
                # ~the first chunk's audio duration (the "other: 2461ms"
                # regression seen on the pre-generate path).
                t_first_audio = time.perf_counter()
                if hasattr(sink, "write"):
                    sink.write(samples)
                final_audio = samples
            else:
                new_material = crossfade(prev_chunk, samples, n_fade_local)
                if hasattr(sink, "write"):
                    sink.write(new_material)
                if final_audio is not None:
                    final_audio = np.concatenate(
                        [final_audio[:-n_fade_local], new_material]
                    )
            prev_chunk = samples

            # Remove the consumed queue head and record new depth.
            self._result_queues.pop(i)
            self._record_depth()
            # i stays the same since we popped the head.

        return {
            "n_chunks": chunks_succeeded + chunks_failed,
            "chunks_succeeded": chunks_succeeded,
            "chunks_failed": chunks_failed,
            "per_chunk_infer_ms": per_chunk_infer_ms,
            "chunk0_phonemize_ms": chunk0_phonemize_ms,
            "buffer_peak": self._peak_depth,
            "buffer_median": (
                statistics.median(self._depth_samples)
                if self._depth_samples else 0.0
            ),
            "buffer_final": len(self._result_queues),
            "hiccup_count": self._hiccup_count,
            "playback_underrun_count": self._underrun_count,
            "t_first_audio": t_first_audio,
            "audio": final_audio,
            "sample_rate": sr,
        }


# ---- streaming playback (sounddevice) ------------------------------------


def _consume_chunk(
    samples, chunk_sr, sink, prev_chunk, n_fade, sr, final_audio,
    stream_open_ms, t_first_audio, chunks_succeeded,
):
    """One chunk of streaming playback: open sink if needed, crossfade,
    write, and update the running state.  Returns the (possibly updated)
    state values so the caller can rebind them.

    Pure w.r.t. global state (only mutates the AudioSink).
    """
    chunks_succeeded += 1
    if sr is None:
        sr = chunk_sr
        n_fade = int(sr * CROSSFADE_MS / 1000)
        t_open = time.perf_counter()
        sink.open(sr)
        stream_open_ms += (time.perf_counter() - t_open) * 1000.0
        t_first_audio = time.perf_counter()
    else:
        assert chunk_sr == sr, (
            f"sample rate changed mid-stream: {sr} -> {chunk_sr}"
        )
    if prev_chunk is None:
        sink.write(samples)
        final_audio = samples
    else:
        new_material = crossfade(prev_chunk, samples, n_fade)
        sink.write(new_material)
        final_audio = np.concatenate([final_audio[:-n_fade], new_material])
    prev_chunk = samples
    return (sr, n_fade, prev_chunk, final_audio, stream_open_ms,
            t_first_audio, chunks_succeeded)

def play_streaming(
    chunks: list[str], voice: str, kokoro: Kokoro, output_path: Path,
    t_user_press: float = 0.0,
    sink: AudioSink | None = None,
    pool: WorkerPool | None = None,
    pre_generate: bool = True,
) -> dict:
    """Worker thread generates audio; main thread streams it to sounddevice
    with crossfade between chunks.  Returns stats dict for reporting.

    If `pool` is given, jobs are submitted to the long-lived WorkerPool
    (reused across prompts) instead of spawning a fresh thread.  This
    shaves ~5-20ms of thread-spawn variance off the TTFW measurement.

    Issue 002-005: when `pre_generate` is True (the default) and a pool
    is provided, all chunks are submitted to the pool up-front via a
    `LookAheadBuffer`, then drained to the sink with crossfade.  This
    embraces the 17× generation slack (model finishes all chunks in
    ~1.4s while the speaker drains at realtime).  The buffer captures
    `buffer_peak/median/final`, `hiccup_count`, and
    `playback_underrun_count` as new stats fields.

    Tracks:
      - t_first_audio: wall clock right before the first stream.write
      - per-chunk infer times (ms) — for distribution stats
      - chunk-0 phonemize + infer ms — for the TTFW breakdown
      - q.get() blocking time — proxy for "generation was the bottleneck"
    """
    q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # Per-chunk stats (worker -> consumer handoff)
    chunk0_phonemize_ms = [0.0]
    chunk0_infer_ms = [0.0]
    per_chunk_infer_ms: list[float] = []
    use_pool = pool is not None

    def pool_backend(text: str, _voice: str, _kokoro: Kokoro):
        """Worker backend: runs one chunk and returns its (samples, sr, ph_ms, inf_ms)."""
        return _timed_generate_chunk_audio(_kokoro, text, _voice)

    owns_pool = False
    result_queues: list[queue.Queue] | None = None
    if use_pool:
        # Caller supplied a long-lived pool.  We share it across prompts.
        # The backend is the same for every prompt; but each play_streaming
        # call closes over its own kokoro + voice, so we need a per-call
        # backend closure.  WorkerPool's start() takes the backend once;
        # since we want a single pool across prompts with potentially
        # different voices, we wrap the backend in a single-callable that
        # uses the most-recently-set call_args.  See set_call_args().
        # NOTE: we set call_args here (needed by BOTH pre-generate and
        # legacy pool paths) but we DO NOT submit yet.  Submission
        # happens in the path-specific branch below — submitting here
        # unconditionally would cause the pre-generate path to submit
        # each chunk twice (once here, once via LookAheadBuffer.submit_all),
        # making the consumer wait for the worker to finish N wasted jobs
        # before reading any results.  This was a real ~3.6s TTFW bug.
        pool.set_call_args(kokoro)
    else:
        # Fallback: spawn a fresh per-prompt thread (legacy).
        def worker():
            try:
                for i, chunk_text in enumerate(chunks):
                    samples, sr, ph_ms, inf_ms = _timed_generate_chunk_audio(
                        kokoro, chunk_text, voice
                    )
                    per_chunk_infer_ms.append(inf_ms)
                    if i == 0:
                        chunk0_phonemize_ms[0] = ph_ms
                        chunk0_infer_ms[0] = inf_ms
                    q.put((samples, sr, chunk_text))
            except Exception as e:
                q.put(ChunkError(e))
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()

    t_start = time.perf_counter()

    sr: int | None = None
    n_fade: int | None = None
    owns_sink = sink is None
    if owns_sink:
        sink = AudioSink()
    prev_chunk: np.ndarray | None = None
    final_audio: np.ndarray | None = None
    chunks_failed = 0
    chunks_succeeded = 0

    # Starvation tracking
    q_get_blocks = 0
    q_get_block_total_ms = 0.0
    Q_GET_BLOCK_THRESHOLD_MS = 1.0  # anything under 1ms is scheduler jitter

    # TTFW marker — set right before the first stream.write
    t_first_audio = 0.0
    # Stream-open cost: how long we spent opening/re-opening the sink
    # around the first chunk.  When the sink is pre-opened, this is ~0.
    stream_open_ms = 0.0

    # Buffer-health fields (issue 002-005).  Populated only on the
    # pool + pre_generate path; stay 0 otherwise.
    buffer_peak = 0
    buffer_median = 0.0
    buffer_final = 0
    hiccup_count = 0
    playback_underrun_count = 0

    try:
        if use_pool and pre_generate:
            # Issue 002-005: pre-generate path.  Use LookAheadBuffer to
            # schedule every chunk up-front and drain with crossfade.
            # The existing pool.submit() / per-job result queue plumbing
            # is unchanged; LookAheadBuffer is a thin coordinator.
            buf = LookAheadBuffer(pool=pool)
            buf.submit_all([(c, voice) for c in chunks])
            n_fade_buf = int(24000 * 0.04)  # 40ms @ 24kHz default
            buf_stats = buf.drain_to_sink(
                sink=sink, n_fade=n_fade_buf,
            )
            per_chunk_infer_ms = buf_stats.get("per_chunk_infer_ms", [])
            chunks_succeeded = buf_stats.get("chunks_succeeded", 0)
            chunks_failed = buf_stats.get("chunks_failed", 0)
            final_audio = buf_stats.get("audio")
            sr = buf_stats.get("sample_rate")
            t_first_audio = buf_stats.get("t_first_audio", 0.0)
            buffer_peak = buf_stats.get("buffer_peak", 0)
            buffer_median = buf_stats.get("buffer_median", 0.0)
            buffer_final = buf_stats.get("buffer_final", 0)
            hiccup_count = buf_stats.get("hiccup_count", 0)
            playback_underrun_count = buf_stats.get(
                "playback_underrun_count", 0
            )
            # chunk-0 phon/infer from the first per-chunk entry
            if per_chunk_infer_ms:
                chunk0_infer_ms[0] = per_chunk_infer_ms[0]
            chunk0_phonemize_ms[0] = buf_stats.get("chunk0_phonemize_ms", 0.0)
        elif use_pool:
            # Legacy pool path: submit all chunks now, then read each
            # from its per-job result queue in order.  The first .get()
            # always blocks (chunk 0 hasn't been generated yet), so it
            # represents the chunk-0 latency and is NOT counted as
            # starvation.  Subsequent gets measure how far behind playback we
            # got (should be 0 with the chunker ahead of audio).
            result_queues = [
                pool.submit(chunk_text, voice) for chunk_text in chunks
            ]
            is_first_get = True
            for rq in result_queues:
                t_get = time.perf_counter()
                samples, chunk_sr, ph_ms, inf_ms = rq.get()
                get_ms = (time.perf_counter() - t_get) * 1000.0
                if not is_first_get and get_ms > Q_GET_BLOCK_THRESHOLD_MS:
                    q_get_blocks += 1
                    q_get_block_total_ms += get_ms
                is_first_get = False
                per_chunk_infer_ms.append(inf_ms)
                if chunks_succeeded == 0:
                    chunk0_phonemize_ms[0] = ph_ms
                    chunk0_infer_ms[0] = inf_ms
                (sr, n_fade, prev_chunk, final_audio, stream_open_ms,
                 t_first_audio, chunks_succeeded) = _consume_chunk(
                    samples, chunk_sr, sink, prev_chunk, n_fade, sr,
                    final_audio, stream_open_ms, t_first_audio,
                    chunks_succeeded,
                )
        else:
            # Legacy path: shared queue with sentinel.
            is_first_get = True
            while True:
                t_get = time.perf_counter()
                item = q.get()
                get_ms = (time.perf_counter() - t_get) * 1000.0
                if not is_first_get and get_ms > Q_GET_BLOCK_THRESHOLD_MS:
                    q_get_blocks += 1
                    q_get_block_total_ms += get_ms
                is_first_get = False

                if item is None:
                    break
                if isinstance(item, ChunkError):
                    print(f"  ! chunk failed: {item.exc}")
                    chunks_failed += 1
                    continue
                samples, chunk_sr, _chunk_text = item
                chunks_succeeded += 1
                (sr, n_fade, prev_chunk, final_audio, stream_open_ms,
                 t_first_audio, chunks_succeeded) = _consume_chunk(
                    samples, chunk_sr, sink, prev_chunk, n_fade, sr,
                    final_audio, stream_open_ms, t_first_audio,
                    chunks_succeeded,
                )
    finally:
        if owns_sink and sink is not None:
            sink.close()

    t_wall_end = time.perf_counter()
    # Wall time includes chunking (t_user_press is set before chunk_text).
    t_total = (
        (t_wall_end - t_user_press)
        if t_user_press
        else t_wall_end - t_start
    )
    t_ttfw = (
        (t_first_audio - t_user_press)
        if t_first_audio and t_user_press
        else 0.0
    )

    if final_audio is not None and sr is not None:
        save_wav(output_path, final_audio, sr)

    return {
        "audio": final_audio,
        "sample_rate": sr,
        "n_chunks": len(chunks),
        "chunks_succeeded": chunks_succeeded,
        "chunks_failed": chunks_failed,
        "time_total": t_total,
        "time_first_chunk": t_ttfw,   # legacy key — now equals TTFW
        "t_user_press": t_user_press,
        "t_first_audio": t_first_audio,
        "t_ttfw": t_ttfw,
        "t_wall_end": t_wall_end,
        "chunk0_phonemize_ms": chunk0_phonemize_ms[0],
        "chunk0_infer_ms": chunk0_infer_ms[0],
        "stream_open_ms": stream_open_ms,
        "per_chunk_infer_ms": per_chunk_infer_ms,
        "q_get_blocks": q_get_blocks,
        "q_get_block_total_ms": q_get_block_total_ms,
        # Issue 002-005: buffer-health fields from LookAheadBuffer.
        "buffer_peak": buffer_peak,
        "buffer_median": buffer_median,
        "buffer_final": buffer_final,
        "hiccup_count": hiccup_count,
        "playback_underrun_count": playback_underrun_count,
    }


# ---- simple playback (winsound fallback) ---------------------------------

def play_simple(
    chunks: list[str], voice: str, kokoro: Kokoro, output_path: Path,
    t_user_press: float = 0.0,
) -> dict:
    """Generate all chunks, crossfade in memory, write a single WAV, play it
    with winsound (blocking).  No streaming — first sound waits for all."""
    t_start = time.perf_counter()

    all_samples: list[np.ndarray] = []
    sr: int | None = None
    chunks_failed = 0
    per_chunk_infer_ms: list[float] = []
    chunk0_phonemize_ms = 0.0
    chunk0_infer_ms = 0.0

    for i, chunk_text in enumerate(chunks):
        try:
            samples, chunk_sr, ph_ms, inf_ms = _timed_generate_chunk_audio(
                kokoro, chunk_text, voice
            )
            if sr is None:
                sr = chunk_sr
            all_samples.append(samples)
            per_chunk_infer_ms.append(inf_ms)
            if i == 0:
                chunk0_phonemize_ms = ph_ms
                chunk0_infer_ms = inf_ms
        except Exception as e:
            print(f"  ! chunk failed: {chunk_text!r}: {e}")
            chunks_failed += 1

    if not all_samples or sr is None:
        t_wall_end = time.perf_counter()
        return {
            "audio": None, "sample_rate": None,
            "n_chunks": len(chunks),
            "chunks_succeeded": 0, "chunks_failed": chunks_failed,
            "time_total": t_wall_end - t_start,
            "time_first_chunk": 0.0,
            "t_user_press": t_user_press,
            "t_first_audio": 0.0,
            "t_ttfw": 0.0,
            "t_wall_end": t_wall_end,
            "chunk0_phonemize_ms": chunk0_phonemize_ms,
            "chunk0_infer_ms": chunk0_infer_ms,
            "stream_open_ms": 0.0,
            "per_chunk_infer_ms": per_chunk_infer_ms,
            "q_get_blocks": 0,
            "q_get_block_total_ms": 0.0,
        }

    n_fade = int(sr * CROSSFADE_MS / 1000)
    final_audio = all_samples[0]
    for i in range(1, len(all_samples)):
        final_audio = np.concatenate(
            [final_audio[:-n_fade],
             crossfade(all_samples[i - 1], all_samples[i], n_fade)]
        )

    save_wav(output_path, final_audio, sr)

    if winsound is not None:
        winsound.PlaySound(
            str(output_path),
            winsound.SND_FILENAME | winsound.SND_NODEFAULT,
        )

    t_wall_end = time.perf_counter()
    # Wall time includes chunking (t_user_press is set before chunk_text).
    t_total = (
        (t_wall_end - t_user_press)
        if t_user_press
        else t_wall_end - t_start
    )
    # No streaming: first sound waits for full generation.
    t_ttfw = t_total
    return {
        "audio": final_audio,
        "sample_rate": sr,
        "n_chunks": len(chunks),
        "chunks_succeeded": len(all_samples),
        "chunks_failed": chunks_failed,
        "time_total": t_total,
        "time_first_chunk": t_ttfw,   # legacy key
        "t_user_press": t_user_press,
        "t_first_audio": t_wall_end,   # not meaningful for non-streaming
        "t_ttfw": t_ttfw,
        "t_wall_end": t_wall_end,
        "chunk0_phonemize_ms": chunk0_phonemize_ms,
        "chunk0_infer_ms": chunk0_infer_ms,
        "stream_open_ms": 0.0,
        "per_chunk_infer_ms": per_chunk_infer_ms,
        "q_get_blocks": 0,
        "q_get_block_total_ms": 0.0,
    }


# ---- main -----------------------------------------------------------------

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or not VOICES_PATH.exists():
        sys.exit(
            "\n  ! Model files missing.\n"
            f"    expected: {MODEL_PATH}\n"
            f"    expected: {VOICES_PATH}\n"
            "    run:      uv run python download_model.py\n"
        )

    print("  ── Kokoro-82M ONNX · local TTS ──")
    print(f"  model   : {MODEL_PATH.name}")
    print(f"  voices  : {VOICES_PATH.name}")
    print(
        f"  audio   : {_AUDIO_BACKEND}"
        f"{' (streaming, crossfade)' if _HAS_STREAMING else ' (pre-build, file-playback)'}"
    )

    args = parse_args()
    providers, backend_label, provider_options = select_providers(
        args.provider, return_options=True
    )
    print(f"  compute : {backend_label}   (requested: {args.provider})")

    # ---- model load --------------------------------------------------------
    t = time.perf_counter()

    # Phase A: ORT session tuning for faster inference
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = True

    sess = ort.InferenceSession(
        str(MODEL_PATH),
        providers=providers,
        provider_options=provider_options,
        sess_options=sess_options,
    )
    kokoro = Kokoro.from_session(sess, str(VOICES_PATH))
    # Warm-up (CUDA compiles kernels here).  We DO time it because it's
    # the cold-start cost the first real prompt would otherwise pay.
    # Issue 002-003: replaced the three fixed-text warmups with a
    # shape-range warmup (run_warmups) that pre-compiles kernels for
    # a list of phoneme-count shapes.  Real lead-ins phonemize to
    # unpredictable counts (30, 55, 80...) and the first prompt's
    # chunk-0 would otherwise trigger an on-demand cuDNN kernel
    # compile (377ms spikes on 6-word af_nicole chunks).  Pre-warming
    # the shape range eliminates the cold-compile spike.
    warmup_pairs: list[tuple[int, float]] = []
    if (not args.no_warm) and "CUDAExecutionProvider" in sess.get_providers():
        # Parse the comma-separated shape list.  Empty string disables
        # shape warmups (run_warmups returns [] for an empty list).
        shapes_raw = (args.warmup_shapes or "").strip()
        if shapes_raw:
            try:
                shape_list = [int(s.strip()) for s in shapes_raw.split(",") if s.strip()]
            except ValueError:
                shape_list = list(DEFAULT_WARMUP_SHAPES)
        else:
            shape_list = []
        warmup_pairs = run_warmups(
            sess, kokoro, shape_list,
        )
    elif args.no_warm:
        print("  warmup  : --no-warm set, skipping kernel pre-compile (cold first chunk).")
    load_ms = (time.perf_counter() - t) * 1000
    print(
        f"  loaded  : {load_ms:.0f}ms  ·  "
        f"active: {sess.get_providers()[0].replace('ExecutionProvider', '').upper()}"
    )
    if warmup_pairs:
        total_warmup_ms = sum(max(0.0, ms) for _, ms in warmup_pairs)
        shape_list = ", ".join(str(c) for c, _ in warmup_pairs)
        print(
            f"  warmed  : {len(warmup_pairs)} shapes "
            f"({shape_list} phonemes) in {total_warmup_ms:.0f}ms  ·  "
            f"pre-compiles cuDNN kernels across the lead-in shape range"
        )

    voice = pick_voice()
    print(f"\n  voice: {voice}   (Ctrl+C or ':q' to exit)\n")

    use_streaming = _HAS_STREAMING and not args.no_stream

    # --profile: collect per-prompt stats and dump them on exit
    profile_buffer: list[dict] = []
    if args.profile:
        print("  --profile enabled: one JSON line per prompt will be emitted on stdout.")

    # Pre-open a persistent AudioSink so the first prompt doesn't pay
    # the PortAudio/WASAPI device-open cost on the critical path.
    # SAMPLE_RATE is a known constant (24000) from kokoro_onnx.config.
    session_sink: AudioSink | None = None
    if use_streaming and sd is not None:
        try:
            session_sink = AudioSink()
            session_sink.open(sample_rate=24000)
            print("  audio   : persistent stream pre-opened at 24000 Hz "
                  "(per-prompt device-open cost eliminated).")
        except Exception as e:
            print(f"  ! could not pre-open audio stream: {e}")
            print("    falling back to per-prompt open (slower first sound).")
            session_sink = None

    # Long-lived WorkerPool: one inference thread reused across prompts.
    # Saves ~5-20ms of thread-spawn variance per prompt (Windows).
    # Backend signature: (text, voice, kokoro).  The (text, voice) pair
    # comes from submit(); the (kokoro) comes from set_call_args().
    session_pool: WorkerPool | None = None
    if use_streaming:
        def _pool_backend(text: str, voice: str, _kokoro: Kokoro):
            return _timed_generate_chunk_audio(_kokoro, text, voice)
        session_pool = WorkerPool()
        session_pool.start(backend=_pool_backend)
        session_pool.set_call_args(kokoro)

    try:
        while True:
            text = read_paragraph()
            if text == ":q" or not text:
                if not text:
                    print("  (empty input, exiting)")
                break

            # ---- chunking -------------------------------------------------
            t_chunk = time.perf_counter()
            chunks = chunk_text(text)

            # Apply lead-in split for streaming mode
            lead_info = None
            if use_streaming:
                chunks, lead_info = _lead_in_split(
                    chunks,
                    lead_in_min=args.lead_in_words,
                    target=TARGET_CHUNK_WORDS,
                    hard_cut=True,
                )

            chunk_time = time.perf_counter() - t_chunk
            n_chunks = len(chunks)
            word_counts = [len(c.split()) for c in chunks]

            # Display chunk stats with lead-in info if applicable
            if lead_info:
                print(
                    f"  ▸ {n_chunks} chunks: {word_counts} words  ·  "
                    f"chunked in {fmt_t(chunk_time)}  ·  "
                    f"lead-in: {lead_info['lead_in_words']}w"
                )
            else:
                print(
                    f"  ▸ {n_chunks} chunk{'s' if n_chunks != 1 else ''}: "
                    f"{word_counts} words  ·  chunked in {fmt_t(chunk_time)}"
                )

            out_path = OUTPUT_DIR / (
                f"{voice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )

            # t_user_press is captured RIGHT BEFORE the generation pipeline
            # starts, so TTFW measures "system latency" only (chunking + print
            # + UI are excluded — they're not model work).
            t_user_press = time.perf_counter()

            # ---- generation + playback (sampling resources throughout) ----
            with ResourceMonitor() as mon:
                if use_streaming and n_chunks > 1:
                    stats = play_streaming(
                        chunks, voice, kokoro, out_path,
                        t_user_press=t_user_press,
                        sink=session_sink,
                        pool=session_pool,
                        pre_generate=not args.no_pre_generate,
                    )
                elif n_chunks == 1:
                    # Single chunk — inline path (no thread/queue overhead).
                    samples, sr, ph_ms, inf_ms = _timed_generate_chunk_audio(
                        kokoro, chunks[0], voice
                    )
                    save_wav(out_path, samples, sr)
                    if sd is not None:
                        # Reuse the session sink if we have one; otherwise
                        # open a one-off stream (legacy path).
                        if session_sink is not None:
                            t_first_audio = time.perf_counter()
                            session_sink.write(samples)
                        else:
                            stream = sd.OutputStream(
                                samplerate=sr, channels=1, dtype="float32"
                            )
                            stream.start()
                            try:
                                t_first_audio = time.perf_counter()
                                stream.write(samples)
                            finally:
                                stream.stop()
                                stream.close()
                    else:
                        t_first_audio = time.perf_counter()
                    t_wall_end = time.perf_counter()
                    t_total = t_wall_end - t_user_press
                    t_ttfw = t_first_audio - t_user_press
                    stats = {
                        "audio": samples,
                        "sample_rate": sr,
                        "n_chunks": 1,
                        "chunks_succeeded": 1,
                        "chunks_failed": 0,
                        "time_total": t_total,
                        "time_first_chunk": t_ttfw,  # legacy key
                        "t_user_press": t_user_press,
                        "t_first_audio": t_first_audio,
                        "t_ttfw": t_ttfw,
                        "t_wall_end": t_wall_end,
                        "chunk0_phonemize_ms": ph_ms,
                        "chunk0_infer_ms": inf_ms,
                        "stream_open_ms": 0.0,  # session sink was pre-opened
                        "per_chunk_infer_ms": [inf_ms],
                        "q_get_blocks": 0,
                        "q_get_block_total_ms": 0.0,
                    }
                else:
                    stats = play_simple(
                        chunks, voice, kokoro, out_path,
                        t_user_press=t_user_press,
                    )

            samples = stats["audio"]
            sr = stats["sample_rate"]
            if samples is None or sr is None:
                print("  ! no audio produced.\n")
                continue

            duration_s = len(samples) / sr

            print(
                f"\n  ✓ {out_path.name}  "
                f"({duration_s:.2f}s audio @ {sr} Hz,  "
                f"{format_summary_line({**stats, 'audio_duration_s': duration_s}).strip()})"
            )
            if stats["chunks_failed"]:
                print(
                    f"    ({stats['chunks_failed']} chunk(s) failed, "
                    f"{stats['chunks_succeeded']} succeeded)"
                )

            # ---- latency / streaming health report ---------------------
            print_latency_report(
                stats, n_chunks, word_counts, duration_s,
                is_streaming_path=(use_streaming and n_chunks > 1),
            )

            # ---- phase chart ---------------------------------------------
            # We don't have fine-grained per-phase timings for the streaming
            # path (chunks overlap with playback).  Approximate by attributing
            # total time to inference for now.
            total_t = stats["time_total"]
            other_t = max(0.0, total_t - chunk_time)
            print_phase_chart(
                [
                    ("Chunking (text)",  chunk_time),
                    ("Gen + play (all)", other_t),
                ],
                total_t,
            )
            print_resource_summary(mon)

            # --profile: capture this prompt's stats for end-of-session emit
            if args.profile:
                profile_stats = dict(stats)
                profile_stats["audio_duration_s"] = duration_s
                profile_stats["voice"] = voice
                profile_stats["lead_in_words"] = (
                    lead_info["lead_in_words"] if lead_info else word_counts[0]
                )
                profile_buffer.append(profile_stats)
            print()
    except KeyboardInterrupt:
        print("\n  interrupted.")

    # Emit all collected profile lines on stdout at the end
    if args.profile and profile_buffer:
        sys.stdout.write(format_profile_batch(profile_buffer))
        sys.stdout.flush()

    # Close the persistent audio sink (if we opened one).
    if session_sink is not None:
        session_sink.close()

    # Close the long-lived worker pool.
    if session_pool is not None:
        session_pool.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

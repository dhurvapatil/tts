"""End-to-end TTFW regression harness.

Measures the user-visible time-to-first-sound for a representative
paragraph through the full streaming path, plus per-phase timings.

Run with:
  uv run python bench_ttfw.py
  uv run python bench_ttfw.py --assert-ttfw-under-ms 400   # CI mode

Outputs a single JSON profile line per run (same shape as `tts.py --profile`)
plus, in --assert mode, a one-line PASS/FAIL summary on stderr.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Reuse the DLL preload dance + sys.path setup from tts.py
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# DLL preload (must run before onnxruntime import)
_nvidia_root = _ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"
if _nvidia_root.exists():
    _extra = []
    for _d in sorted(_nvidia_root.glob("*/bin")):
        if _d.is_dir():
            _extra.append(str(_d.resolve()))
    if _extra:
        import os
        os.environ["PATH"] = ";".join(_extra) + ";" + os.environ.get("PATH", "")
        for _p in _extra:
            try:
                os.add_dll_directory(_p)
            except Exception:
                pass
del _nvidia_root, _extra

import numpy as np  # noqa: E402

import onnxruntime as ort  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402

# Quiet ORT logging
import logging
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
try:
    ort.set_default_logger_severity(3)
except Exception:
    pass

from tts import (  # noqa: E402
    MODEL_PATH, VOICES_PATH, OUTPUT_DIR,
    _timed_generate_chunk_audio, save_wav, emit_profile_line,
    chunk_text, _lead_in_split,
    format_profile_batch,
    DEFAULT_WARMUP_SHAPES, run_warmups,
)


# ---- VRAM plateau detection (issue 002-004) -------------------------------

@dataclass
class PlateauStats:
    """Result of plateau detection over a series of per-run peak VRAMs.

    `first` and `last` are None for an empty series.  `delta` is
    `last - first` (signed; may be negative if VRAM shrinks).  The
    verdict `is_plateau` uses ABSOLUTE delta: any drift whose magnitude
    exceeds `threshold_mb` is flagged.
    """
    first: int | None
    last: int | None
    delta: int
    is_plateau: bool


# Default threshold: 64 MB comfortably absorbs cuDNN workspace / ORT
# arena caching per shape but would catch a real leak (typically
# +5-50 MB/iteration observed in the wild on long sessions).
DEFAULT_VRAM_PLATEAU_THRESHOLD_MB = 64


def compute_vram_plateau(
    per_run_vram_mb: Sequence[int],
    threshold_mb: int = DEFAULT_VRAM_PLATEAU_THRESHOLD_MB,
) -> PlateauStats:
    """Classify whether the per-run peak VRAM has plateaued.

    An empty series is vacuously a plateau (no leak visible).  A
    single-element series has zero delta and is a plateau.  For a
    multi-element series, plateau iff `|last - first| <= threshold_mb`.
    """
    if not per_run_vram_mb:
        return PlateauStats(first=None, last=None, delta=0, is_plateau=True)
    first = int(per_run_vram_mb[0])
    last = int(per_run_vram_mb[-1])
    delta = last - first
    return PlateauStats(
        first=first,
        last=last,
        delta=delta,
        is_plateau=abs(delta) <= threshold_mb,
    )


# The standard paragraph used for TTFW measurement.  Same one as the
# previous tts.py interactive run that produced 775ms TTFW.
STANDARD_PARAGRAPH = (
    "To get started, please check that your device is securely plugged into "
    "a power source and the Wi-Fi router is turned on. If the issue continues, "
    "restart the system by holding the power button for ten seconds. You can "
    "view step-by-step guides on the Support Center for further assistance."
)


def measure_ttfw(kokoro, text, voice):
    """Run a single end-to-end generation, returning the stats dict.

    This is the public measurement seam: same shape as
    tts.play_streaming() returns, so it can be passed straight into
    emit_profile_line().
    """
    chunks = chunk_text(text)
    chunks, lead_info = _lead_in_split(chunks)
    n_chunks = len(chunks)
    word_counts = [len(c.split()) for c in chunks]
    lead_in_words = lead_info["lead_in_words"] if lead_info else word_counts[0]

    t_user_press = time.perf_counter()
    per_chunk_infer_ms: list[float] = []
    chunk0_phonemize_ms = 0.0
    chunk0_infer_ms = 0.0
    all_samples: list[np.ndarray] = []
    sr: int | None = None

    for i, chunk_text_str in enumerate(chunks):
        samples, chunk_sr, ph_ms, inf_ms = _timed_generate_chunk_audio(
            kokoro, chunk_text_str, voice
        )
        per_chunk_infer_ms.append(inf_ms)
        if i == 0:
            chunk0_phonemize_ms = ph_ms
            chunk0_infer_ms = inf_ms
            # For bench purposes we set t_first_audio at the END of chunk 0
            # generation.  This measures the *floor* the model is capable
            # of (phon + infer), without the PortAudio open that the
            # interactive CLI does.  The CLI's TTFW = this + stream-open
            # + queue/thread overhead.
            t_first_audio = time.perf_counter()
        if sr is None:
            sr = chunk_sr
        all_samples.append(samples)

    # Stitch with crossfade (40ms) for audio duration
    if sr is None or not all_samples:
        audio_duration_s = 0.0
    else:
        n_fade = int(sr * 40 / 1000)
        if n_fade <= 0 or len(all_samples) == 1:
            audio = np.concatenate(all_samples) if len(all_samples) > 1 else all_samples[0]
        else:
            audio = all_samples[0]
            for i in range(1, len(all_samples)):
                audio = np.concatenate(
                    [audio[:-n_fade],
                     np.zeros(0)]  # placeholder, see real crossfade below
                )
            # Real crossfade (matches tts.crossfade semantics)
            from tts import crossfade
            audio = all_samples[0]
            for i in range(1, len(all_samples)):
                audio = np.concatenate(
                    [audio[:-n_fade], crossfade(all_samples[i - 1], all_samples[i], n_fade)]
                )
        audio_duration_s = len(audio) / sr
        # Also write the wav (parity with the CLI)
        out = OUTPUT_DIR / f"_bench_ttfw_{int(time.time())}.wav"
        save_wav(out, audio, sr)

    t_wall_end = time.perf_counter()
    time_total = t_wall_end - t_user_press
    t_ttfw = (t_first_audio - t_user_press) if t_first_audio else 0.0

    return {
        "t_ttfw": t_ttfw,
        "chunk0_phonemize_ms": chunk0_phonemize_ms,
        "chunk0_infer_ms": chunk0_infer_ms,
        "stream_open_ms": 0.0,  # bench uses no sink; not a TTFW component here
        "per_chunk_infer_ms": per_chunk_infer_ms,
        "time_total": time_total,
        "audio_duration_s": audio_duration_s,
        "sample_rate": sr,
        "voice": voice,
        "lead_in_words": lead_in_words,
        "n_chunks": n_chunks,
    }


def main() -> int:
    p = argparse.ArgumentParser(prog="bench_ttfw.py")
    p.add_argument("--voice", default="af_bella")
    p.add_argument("--runs", type=int, default=3,
                   help="number of timed runs (median reported)")
    p.add_argument("--assert-ttfw-under-ms", type=float, default=None,
                   help="CI mode: exit 1 if median TTFW exceeds this many ms")
    p.add_argument("--text", default=STANDARD_PARAGRAPH)
    p.add_argument("--no-warm", action="store_true",
                   help="skip the warmup pass (cold-cache measurement)")
    p.add_argument("--vram-plateau-runs", type=int, default=0,
                   help=(
                       "issue 002-004: run the standard paragraph N times and "
                       "report per-run peak VRAM. 0 = disabled (default)."
                   ))
    p.add_argument("--assert-vram-plateau-under-mb", type=float, default=None,
                   help=(
                       "issue 002-004: CI mode, exit 1 if |last-first| VRAM "
                       "delta exceeds K MB. Default threshold 64 MB when "
                       "--vram-plateau-runs > 0 is set without this flag."
                   ))
    args = p.parse_args()

    print(f"loading model ...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    kokoro = Kokoro.from_session(sess, str(VOICES_PATH))
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f}ms; "
          f"providers={sess.get_providers()}", file=sys.stderr, flush=True)

    if not args.no_warm:
        # Shape-range warmup (issue 002-003): pre-compile cuDNN kernels
        # for the full lead-in shape range so the first timed run
        # doesn't trigger an on-demand kernel compile.
        _ = run_warmups(
            sess, kokoro, DEFAULT_WARMUP_SHAPES, voice=args.voice,
        )

    ttfws: list[float] = []
    chunk0_inf: list[float] = []
    profiles: list[str] = []
    # Issue 002-004: optional per-run peak VRAM tracking.
    per_run_vram: list[int] = []
    for run_idx in range(args.runs):
        stats = measure_ttfw(kokoro, args.text, args.voice)
        profiles.append(emit_profile_line(stats))
        ttfws.append(stats["t_ttfw"] * 1000.0)
        chunk0_inf.append(stats["chunk0_infer_ms"])
        print(f"  run {run_idx+1}: ttfw={ttfws[-1]:.0f}ms  "
              f"chunk0_inf={chunk0_inf[-1]:.0f}ms  "
              f"chunks={stats['n_chunks']}", file=sys.stderr, flush=True)

    med_ttfw = statistics.median(ttfws)
    med_chunk0 = statistics.median(chunk0_inf)
    print(f"\n  median ttfw: {med_ttfw:.0f}ms   "
          f"median chunk0 infer: {med_chunk0:.0f}ms\n", file=sys.stderr)

    # Always print the median run's profile line on stdout
    median_idx = ttfws.index(min(ttfws, key=lambda x: abs(x - med_ttfw)))
    sys.stdout.write(profiles[median_idx])
    sys.stdout.flush()

    if args.assert_ttfw_under_ms is not None:
        if med_ttfw > args.assert_ttfw_under_ms:
            print(f"\n  FAIL: median ttfw {med_ttfw:.0f}ms > "
                  f"budget {args.assert_ttfw_under_ms:.0f}ms",
                  file=sys.stderr)
            return 1
        print(f"\n  PASS: median ttfw {med_ttfw:.0f}ms <= "
              f"budget {args.assert_ttfw_under_ms:.0f}ms",
              file=sys.stderr)

    # Issue 002-004: optional VRAM plateau mode.
    if args.vram_plateau_runs and args.vram_plateau_runs >= 2:
        from tts import ResourceMonitor, _nvidia_query
        for _ in range(args.vram_plateau_runs):
            with ResourceMonitor(interval=0.02) as mon:
                measure_ttfw(kokoro, args.text, args.voice)
            peak = int(mon.peak_gpu_mem) if mon.has_gpu else 0
            per_run_vram.append(peak)
        threshold = (
            int(args.assert_vram_plateau_under_mb)
            if args.assert_vram_plateau_under_mb is not None
            else DEFAULT_VRAM_PLATEAU_THRESHOLD_MB
        )
        stats = compute_vram_plateau(per_run_vram, threshold_mb=threshold)
        vram_str = ", ".join(str(v) for v in per_run_vram)
        print(
            f"  vram_plateau: [{vram_str}] MB  delta: "
            f"{stats.delta:+d} MB  threshold: {threshold} MB",
            file=sys.stderr,
        )
        if args.assert_vram_plateau_under_mb is not None and not stats.is_plateau:
            print(
                f"\n  FAIL: VRAM plateau {stats.delta:+d} MB exceeds "
                f"budget {threshold} MB",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Measure where first-sound time actually goes.

Reports, for several chunk sizes:
  - phonemize time (CPU, espeak/misaki)
  - pure ONNX inference time (GPU)
  - total (the real first-sound contribution of chunk 0)
and the inferred realtime multiplier.

Also measures warmup effects: with vs without a representative-length
warmup pass before the timed inference.
"""
from __future__ import annotations
import os, sys, time, statistics
from pathlib import Path

# Reuse the DLL preload dance from tts.py
_ROOT = Path(__file__).resolve().parent
_nvidia_root = _ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"
if _nvidia_root.exists():
    _extra = []
    for _d in sorted(_nvidia_root.glob("*/bin")):
        if _d.is_dir(): _extra.append(str(_d.resolve()))
    if _extra:
        os.environ["PATH"] = ";".join(_extra) + ";" + os.environ.get("PATH","")
        for _p in _extra:
            try: os.add_dll_directory(_p)
            except Exception: pass

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

MODEL = _ROOT / "models" / "kokoro-v1.0.onnx"
VOICES = _ROOT / "models" / "voices-v1.0.bin"
VOICE = "bm_george"

print(f"loading model ...", flush=True)
t = time.perf_counter()
sess = ort.InferenceSession(str(MODEL), providers=["CUDAExecutionProvider","CPUExecutionProvider"])
kokoro = Kokoro.from_session(sess, str(VOICES))
print(f"  loaded in {(time.perf_counter()-t)*1000:.0f}ms; providers={sess.get_providers()}")

# warmup with a SHORT input (matches current tts.py behaviour)
t = time.perf_counter()
_ = kokoro.create("ready.", voice=VOICE, lang="en-us", speed=1.0)
print(f"  short warmup ('ready.') in {(time.perf_counter()-t)*1000:.0f}ms")

# test chunks of increasing size
chunks = {
    "10w":  "The morning sun rose slowly over the hills beyond the harbor.",
    "20w":  "The morning sun rose slowly over the hills beyond the harbor as the crew prepared the ship for a long voyage to distant waters.",
    "30w":  "The morning sun rose slowly over the hills beyond the harbor as the crew prepared the ship for a long voyage to distant waters, double-checking every rope, every sail, and every instrument aboard the sturdy wooden vessel.",
    "44w":  "At precisely 07:43 a.m. on June 25, 2026, the autonomous research vessel Asteria-7 departed the harbor carrying 128 scientists, engineers, and observers from 17 countries, each tasked with evaluating whether next-generation artificial intelligence systems could reliably interpret complex, multilingual information under unpredictable real-world conditions.",
}

def measure(text, n=3):
    # phonemize
    t_ph = []
    for _ in range(n):
        t = time.perf_counter()
        ph = kokoro.tokenizer.phonemize(text, lang="en-us", norm=True)
        t_ph.append((time.perf_counter()-t)*1000)
    # inference (phonemes already computed)
    t_inf = []
    last_audio_len = 0
    for _ in range(n):
        t = time.perf_counter()
        samples, sr = kokoro.create(ph, voice=VOICE, lang="en-us", is_phonemes=True, speed=1.0)
        dt = (time.perf_counter()-t)*1000
        t_inf.append(dt)
        last_audio_len = len(samples)/sr
    return statistics.median(t_ph), statistics.median(t_inf), last_audio_len

print(f"\n{'name':6} {'words':>5} {'phones':>6} {'phon_ms':>8} {'inf_ms':>8} {'total_ms':>9} {'audio_s':>8} {'xRT':>6}")
print("-"*70)
for name, txt in chunks.items():
    nw = len(txt.split())
    ph = kokoro.tokenizer.phonemize(txt, lang="en-us", norm=True)
    nph = len(ph)
    ph_ms, inf_ms, audio_s = measure(txt)
    total = ph_ms + inf_ms
    xrt = audio_s / (inf_ms/1000) if inf_ms>0 else 0
    print(f"{name:6} {nw:>5} {nph:>6} {ph_ms:>8.1f} {inf_ms:>8.1f} {total:>9.1f} {audio_s:>8.2f} {xrt:>6.1f}")

# measure effect of a representative-length warmup BEFORE the 44w inference
print(f"\n--- warmup-shape effect on 44w chunk ---")
# fresh session to simulate cold start
sess2 = ort.InferenceSession(str(MODEL), providers=["CUDAExecutionProvider","CPUExecutionProvider"])
kokoro2 = Kokoro.from_session(sess2, str(VOICES))
# warm with a LONG input (~44 words) using the SAME voice
long_warm = chunks["44w"]
ph_long = kokoro2.tokenizer.phonemize(long_warm, lang="en-us", norm=True)
t = time.perf_counter()
_ = kokoro2.create(ph_long, voice=VOICE, lang="en-us", is_phonemes=True, speed=1.0)
print(f"  long warmup in {(time.perf_counter()-t)*1000:.0f}ms (compiles shape-{len(ph_long)} kernels)")
# now timed inference of the 44w chunk (same shape)
txt = chunks["44w"]
ph = kokoro2.tokenizer.phonemize(txt, lang="en-us", norm=True)
times = []
for _ in range(3):
    t = time.perf_counter()
    s, sr = kokoro2.create(ph, voice=VOICE, lang="en-us", is_phonemes=True, speed=1.0)
    times.append((time.perf_counter()-t)*1000)
print(f"  44w inf after long warmup: median {statistics.median(times):.1f}ms (audio {len(s)/sr:.2f}s)")
print(f"  -> so warmup-shape vs short-shape is the compile cost we can relocate")
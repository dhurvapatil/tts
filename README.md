# Kokoro-82M ONNX · local TTS CLI

A small terminal app that runs [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
locally via ONNX Runtime. Pick a voice, type/paste a paragraph, get a WAV
file in `output/`. After every generation you get a per-phase timing chart,
peak CPU/RAM, peak VRAM, and the **time-to-first-sound**.

Long inputs are **phrase-wise chunked** and **streamed** — you hear chunk 1
in **~200ms** while the rest generates (CUDA, 4060, 6-word lead-in). Chunks
are crossfaded so boundaries are smooth, not clicky.

Supports **CUDA** (NVIDIA, default if detected) and **CPU** inference.

---

## 1. Prerequisites

* **Python 3.11+**
* **espeak-ng** — required for phonemization
  * **Easiest (Windows):** download `espeak-ng-*-x64.msi` from
    <https://github.com/espeak-ng/espeak-ng/releases>, run it, open a *new*
    terminal, and verify:
    ```powershell
    espeak-ng --version
    ```
  * Or with winget:
    ```powershell
    winget install -e --id eSpeak-NG
    ```
* **uv** (recommended) — <https://docs.astral.sh/uv/getting-started/installation/>
* **For GPU:** an NVIDIA GPU + a recent driver (≥ 525 is safe). The
  `onnxruntime-gpu[cuda,cudnn]` wheel pulls in matching CUDA 12 + cuDNN 9
  pip packages automatically — **no system CUDA toolkit install needed.**

## 2. Project setup

```powershell
uv sync                              # creates .venv, installs deps + GPU + audio
uv run python download_model.py      # grabs the ONNX model + voices (~85 MB)
```

> If a download fails, you can also grab the same two files from the
> [HuggingFace mirror](https://huggingface.co/hexgrad/Kokoro-82M) and drop
> them into `models/`.

**CPU-only install?** Edit `pyproject.toml` and swap
`"onnxruntime-gpu[cuda,cudnn]>=1.26.0,<1.27.0"` for `"onnxruntime>=1.18.0"`,
then `uv sync`.

Plain `pip` equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install kokoro-onnx "onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27" numpy psutil sounddevice
python download_model.py
```

## 3. Run

```powershell
uv run tts.py                            # streaming + CUDA (auto-detect)
uv run tts.py --provider cpu             # force CPU
uv run tts.py --provider cuda            # force CUDA
uv run tts.py --no-stream                # pre-build all then play (winsound path)
uv run tts.py --lead-in-words 4          # smaller first chunk (faster TTFW)
uv run tts.py --no-warm                  # skip model-load warmups (cold-cache bench)
uv run tts.py --profile                  # one JSON line per prompt on stdout
```

Flow:

1. Pick a voice from the numbered list (or type its id, e.g. `af_heart`).
2. Paste your text. **Enter twice** (a blank line) to generate, or type
   `:q` / `Ctrl+C` to quit.
3. Audio starts playing immediately (or as soon as chunk 1 is ready for
   long inputs). A single final WAV is written to `output/`.

## 4. Streaming & chunking

Long inputs are split into **phrase-sized chunks** for low-latency playback:

- **Target:** 25 words/chunk
- **Lead-in (chunk 0):** 6 words by default (configurable via
  `--lead-in-words N`). A small lead-in shrinks the first chunk's infer
  time and gets the first sound to you faster. The chunker tries a
  natural clause boundary within a small window past `N` words; if
  none lands there, it hard-cuts at `N` words. The crossfade (below)
  smooths the join.
- **First chunk (legacy / non-streaming):** at least 20 words (merges
  forward if the first sentence is short) — this is the no-stream path.
- **Max:** 50 words (Kokoro quality starts to degrade past this)
- **Split points:** sentence boundaries first, then clauses (`, ; : —`),
  then conjunctions (with the conjunction kept in the *next* chunk for
  sentence-initial prosody)
- **Abbreviation protection:** `Dr.`, `Mr.`, `3.14`, `e.g.`, `i.e.`, URLs,
  ellipsis, etc. are recognized so the sentence splitter doesn't trip on them

Chunks are **crossfaded** at their boundaries (40 ms equal-power overlap)
so transitions are smooth, not clicky. The first chunk plays the moment
it's generated — the rest streams in while you listen.

## 5. Example output (CUDA, streaming)

```
  ── Kokoro-82M ONNX · local TTS ──
  model   : kokoro-v1.0.onnx
  voices  : voices-v1.0.bin
  audio   : sounddevice (streaming, crossfade)
  compute : CUDA   (requested: auto)
  loaded  : 2788ms  ·  active: CUDA
  voice: am_liam   (Ctrl+C or ':q' to exit)

  > The art of writing well is a lifelong pursuit that rewards
  > patience and practice above all else. Every great writer...
  >
  ▸ 8 chunks: [33, 21, 14, 17, 17, 14, 21, 10] words  ·  chunked in 0ms

  ✓ am_liam_20260625_153742.wav  (41.75s audio @ 24000 Hz,  overall 1.03x realtime)
    first-sound: 662ms  (after chunk 1 of 8)

  Phase breakdown
  ──────────────────────────────────────────────────────────────
  Chunking (text)      ··                       0ms     0%
  Gen + play (all)     ██████████████████████  43.23s   100%
  ──────────────────────────────────────────────────────────────
  TOTAL                ██████████████████████  43.23s   100%

  Resource usage (during pipeline)
  ──────────────────────────────────────────────────────────────
  CPU peak    120.2%     (avg  12.2%)
  RAM peak    1121.6 MB
  VRAM peak    1137 MB
  ──────────────────────────────────────────────────────────────
```

**Numbers to look at:**
- `TTFW (total)         187ms` — you wait <200ms to start hearing a 16s clip
  (CUDA, 4060, 6-word lead-in, persistent AudioSink)
- `generation 0.06× ✓` — the model itself is ~16× faster than realtime
  (this is the real "model speed" number; ignore wall-RTF for a single
  prompt — it's dominated by realtime audio playback)
- `chunk-0 infer 188ms` — the floor of TTFW is the lead-in chunk's
  inference; smaller `--lead-in-words` = faster
- `stream-open 0ms` — the persistent AudioSink was pre-opened at session
  start; this is the cost that used to be ~500ms per prompt
- `CPU peak 110%` / `avg 9%` — GPU is doing the work; CPU is mostly idle
  (lower than before the `pynvml` swap — we no longer spawn
  `nvidia-smi` every 100ms)
- `VRAM peak 1139 MB` — for a 16s clip; rises with chunk count

For a single short sentence (one chunk), generation runs at **0.05–0.10x
realtime** (the model is ~10–20× faster than the audio it produces).

## 6. Regression harness

`bench_ttfw.py` measures TTFW end-to-end on a standard paragraph and
prints one JSON profile line on stdout (same shape as `tts.py --profile`).
Use it as a regression guard after any change to the pipeline:

```powershell
uv run python bench_ttfw.py --runs 3                       # 3 timed runs, median reported
uv run python bench_ttfw.py --runs 5 --assert-ttfw-under-ms 400   # CI mode: exit 1 if budget exceeded
uv run python bench_ttfw.py --no-warm --runs 3             # cold-cache measurement
uv run python bench_ttfw.py --voice am_michael             # a different voice
```

The profile line is also written to stdout in `--profile` mode, so you
can diff two runs:

```powershell
uv run tts.py --profile < prompt.txt > before.json
# (make a change)
uv run tts.py --profile < prompt.txt > after.json
diff before.json after.json
```



## 7. `--no-stream` fallback

If `sounddevice` fails to import (e.g. PortAudio missing on Linux), the
script will tell you. Pass `--no-stream` to force the pre-build path:

- Generates all chunks, crossfades them, writes one WAV, plays it once
- No streaming benefit — you wait for the full generation
- Uses `winsound` on Windows (always available)
- Smaller first-sound time only for very short inputs

## 8. Files

| path                | purpose                                |
|---------------------|----------------------------------------|
| `tts.py`            | the CLI                                |
| `download_model.py` | one-shot fetcher for the model files   |
| `bench_ttfw.py`     | end-to-end TTFW regression harness     |
| `bench_firstsound.py` | legacy per-shape phon/inf breakdown  |
| `tests/`            | unit tests (run with `uv run python tests/test_*.py`) |
| `models/`           | ONNX model + voices table (gitignored) |
| `output/`           | generated wavs (gitignored)            |
| `pyproject.toml`    | uv-managed dependencies                |

## 9. Performance notes

* **CUDA vs CPU** for an 82 M-param model: ~2–3× faster on a 4060, but
  even CPU runs under real-time. The biggest GPU win shows up for longer
  inputs / streaming (the chunker keeps generation ahead of playback).
* **First inference is slower** — CUDA compiles / caches kernels on the
  first call. The script does three warm-ups at startup (short, lead-in,
  long) so the first real request is already warm in *all* the chunk
  shapes you'll hit. Pass `--no-warm` for cold-cache measurements.
* **VRAM** is reported via `pynvml` every 50 ms (in-process, no
  `nvidia-smi` subprocess spawn). Falls back to CPU/RAM-only on
  non-NVIDIA systems.
* **TTFW optimizations applied:** HEURISTIC cuDNN search (kills the
  451ms p95 spike on cold first-call), lead-in-shaped warmup, 6-word
  lead-in chunk by default (configurable via `--lead-in-words N`),
  pre-opened persistent `AudioSink` (eliminates per-prompt PortAudio
  open), and a long-lived `WorkerPool` (saves per-prompt thread-spawn
  variance). See `docs/PRD-001-first-word-latency.md` for the full
  rationale and per-change measurement.
* **Streaming caveat:** if the model is slower than real-time on your
  hardware (it isn't on a 4060, but on a really old CPU it could be), the
  user hears silence between chunks as the worker catches up. Use
  `--no-stream` to avoid the gap.

## 10. Troubleshooting

* **`OSError: espeak-ng not found`** — install it (step 1) and open a new
  terminal so `PATH` refreshes.
* **`sounddevice` import error** — install PortAudio:
  * Windows: usually bundled with the `sounddevice` wheel; if not, the
    `conda-forge` `portaudio` package works
  * Linux: `sudo apt install libportaudio2`
  * macOS: `brew install portaudio`
  Then `uv pip install --force-reinstall sounddevice`.
* **CUDA provider not in the list** — check
  `uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"`.
  If `CUDAExecutionProvider` is missing, the nvidia pip packages didn't
  resolve. Try `uv pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27"`.
* **`DLL initialization routine failed` / `cublasLt64_12.dll is missing`**
  — your onnxruntime-gpu and nvidia-cu* versions are out of sync. Re-run
  `uv sync` to re-resolve.
* **Model load error / shape mismatch** — the file may be an older
  `v0_19` build. Rename to `kokoro-v1.0.onnx` / `voices-v1.0.bin`, or
  fetch v1.0 from the link in step 2.
* **Chunking splits mid-sentence / prosody feels off** — the chunker is
  tuned for English prose. Tweak the constants at the top of `tts.py`:
  `MAX_CHUNK_WORDS` (raise to keep more sentences together),
  `CROSSFADE_MS` (raise to soften boundaries further).

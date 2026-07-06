# PRD-001: First-Word Latency Optimization for Kokoro TTS CLI

Status: ready-for-agent
Owner: tts.py
Targets: `uv run tts.py` interactive TTFW on CUDA + streaming

## Problem Statement

Users of `tts.py` measure perceived quality of a streaming TTS system by **time-to-first-sound (TTFW)** — the wall-clock delay between pressing Enter and hearing the first word of audio. Today, on the bundled test paragraph (4 chunks, 15s of audio, `af_bella` voice, CUDA), the CLI reports:

- TTFW: **775–959 ms**
  - phonemize: 1 ms
  - inference (chunk 0, 15 words): 177–451 ms (p95 451 ms — a one-time cuDNN algo-search spike on the first real chunk)
  - **overhead: 504–507 ms** (65% of TTFW; labeled "chunking/print/thread/stream")
- RTF: 1.07–1.09× — this is `wall_time / audio_duration`, which conflates generation speed with realtime playback. Generation is ~16× faster than realtime (1s of GPU work produces 16s of audio).
- GPU utilization: peak 54–67%, **average 1.8–2.9%** — GPU sits idle waiting for the speaker to drain.

The dominant cost of "how fast does the user hear the first word" is the 500ms of overhead that sits *between* pressing Enter and the first sample reaching the audio device. Inside that overhead: PortAudio/WASAPI `OutputStream` open + start happens **after** the first chunk's inference completes, instead of overlapping with it or being pre-opened at session start. A secondary cost is the 270ms chunk-0 inference itself, where 451ms p95 spikes on cold first-call and the lead-in chunk is still 15 words.

The user's goal is: **TTFW ≤ 400ms** for typical paragraphs (target: 250ms) on the existing CUDA hardware, with no regression in audio quality, no extra dependencies, and the existing `bench_firstsound.py` style of measurement as the regression guard.

## Solution

A two-track optimization targeting the two components of TTFW: **shrink the overhead**, and **shrink chunk-0 inference**. The product surface changes minimally — same CLI, same streaming behavior, same voice list, same output WAV. Internally:

1. **Eliminate the serial PortAudio open** by pre-opening one persistent `sd.OutputStream` at session start (or at voice pick), kept warm across prompts. SAMPLE_RATE is a known constant from the kokoro-onnx library, so the device can be opened before any text is typed. Writes just go to the already-open stream.

2. **Overlap what remains** by opening the stream concurrently with the worker's first chunk inference in a fallback path (used when a persistent stream can't be reused, e.g. device change).

3. **Stabilize and shrink chunk-0 inference** by:
   - Passing CUDA provider options that disable exhaustive cuDNN algo search (`cudnn_conv_algo_search: HEURISTIC`) to kill the 451ms p95 spike.
   - Tightening the lead-in cut so chunk 0 is 6–8 words (~120ms) instead of 15 words (~270ms), with a hard word cut as a fallback when no clause boundary lands early. Slightly less natural first-phrase prosody is an acceptable trade, smoothed by the existing 40ms crossfade.

4. **Land CUDA Graphs (deferred to a follow-up PRD)** as a 2–4× inference throughput lever once the cheaper fixes are in. This requires a fixed-shape padded path and shape-padding tokens to MAX_PHONEME_LENGTH, so it's gated on a clean prototype.

5. **Reframe the displayed RTF** to mean **generation RTF** (`total_inference_time / audio_duration`) rather than wall-RTF. The current wall-RTF is a property of the audio sink, not the model, and is misleading. Wall-clock still appears as a separate line for transparency.

The measurement and regression harness is `bench_firstsound.py` (and a new `bench_ttfw.py` for the full streaming path). Every change ships with a bench delta in the PR.

## User Stories

1. As a CLI user, I want to hear the first word of my paragraph in under 400ms after pressing Enter, so that interactive TTS feels responsive.

2. As a CLI user, I want consistent first-sound latency on every prompt, so that a slow first prompt doesn't make me think the tool is broken.

3. As a CLI user, I want the displayed "realtime" metric to reflect how fast the model generates, not how fast my speaker drains, so that I can tell when a slowdown is model-bound.

4. As a CLI user on CUDA, I want CUDA provider options tuned for low-latency inference (no exhaustive cuDNN search, arena sized for chunks, etc.) so that my hardware is used well.

5. As a CLI user, I want the first chunk to be short (6–10 words) so that the first chunk's inference finishes quickly and the audio device starts sooner.

6. As a CLI user, I want the crossfade to smooth over the slightly harder lead-in cut so that the slightly less natural phrase boundary isn't audible.

7. As a CLI user, I want the audio device to be opened once at startup (not per prompt) so that I don't pay a per-prompt WASAPI open cost.

8. As a CLI user, I want the device-open cost to overlap with the first chunk's inference in the fallback path so that the cost is hidden behind the infer even when a persistent stream isn't viable.

9. As a CLI user, I want chunk-0 phonemes to be pre-warmed at the right shape during model load so that the first real prompt doesn't trigger an on-demand cuDNN compile.

10. As a CLI user, I want chunking to consider lead-in size as a tunable knob (CLI flag or constant) so that I can trade prosody for TTFW without editing source.

11. As a CLI user, I want the inferred path to be CUDA-Graph-ready (fixed-shape padded inference) so that a follow-up optimization can plug in without rewriting the call site.

12. As a CLI user, I want the output WAV to be the same format and quality as today (24kHz, mono, 16-bit PCM, 40ms crossfade between chunks).

13. As a CLI user, I want the per-prompt resource and timing report to still print after every generation, with the new metrics slotted in alongside the old ones.

14. As a CLI user, I want a `--lead-in-words N` flag so that I can experiment with lead-in size from the command line.

15. As a CLI user, I want a `--no-warm` escape hatch so that I can disable the model-load warmups when benchmarking pure inference (e.g. cold cache behavior).

16. As a CLI user, I want fallback to `--no-stream` / winsound to still work, and the optimizations to gracefully no-op on that path.

17. As a CLI user on CPU, I want the same TTFW improvements to apply (pre-open stream, shorter lead-in, latency-report reframing) even though the absolute numbers will be larger.

18. As a CLI user, I want the TTFW breakdown (phonemize / infer / overhead) to remain printed, with the "overhead" component dropping dramatically so I can see the win in the existing report.

19. As a developer / agent picking up the next task, I want the inferences in this PRD to be a follow-up, with the prototype that proved the fixed-shape padded path callable from `kokoro.create()` cleanly.

20. As a developer, I want `bench_ttfw.py` to exist as a regression harness that measures TTFW end-to-end (not just phonemize vs infer) so that future changes can't silently regress the user-visible metric.

21. As a developer, I want the `ResourceMonitor` to use pynvml (or an equivalent in-process API) instead of spawning `nvidia-smi` as a subprocess every 100ms, so that the CPU samples aren't polluted by ~170 process spawns per run and so the resource overhead matches what the system is actually doing.

22. As a developer, I want CUDA provider options to be a single named function / dict so that tuning lives in one place and is testable, not scattered.

23. As a developer, I want the streaming pipeline's per-phase timings to be recorded individually (chunk-0 phonemize, chunk-0 infer, queue wait, stream-open, first-write) so that future regressions in any one of them are pinpointable from the post-run report.

24. As a developer, I want the lead-in splitter to be unit-testable in isolation (text → chunks + lead-in info) so that tuning `LEAD_IN_MIN` and friends doesn't require end-to-end runs.

25. As a developer, I want the OutputStream lifecycle to be encapsulated in a small class (open once, write many, close on exit) so that "forgot to close" / "opened twice" bugs are structurally impossible.

26. As a developer, I want the chunker to expose a knob for "minimum acceptable chunk-0 size" separate from the natural-lead-in size, so that TTFW is bounded from below.

27. As a developer, I want the CLI to surface a `--profile` flag that writes a per-prompt JSON line (chunk sizes, per-phase timings, RTF gen-only, RTF wall, TTFW) so that I can diff TTFW across runs in a machine-readable way.

## Implementation Decisions

### Module / function changes

- **`select_providers()` in `tts.py`** — extend to return a `(providers, backend_label, provider_options_dict)` triple. Move the provider-options dict into a module-level constant `CUDA_PROVIDER_OPTIONS` so it's testable and discoverable. Provider options for CUDA will include at minimum: `device_id`, `arena_extend_strategy: "kSameAsRequested"`, `cudnn_conv_algo_search: "HEURISTIC"`, and a tunable-op enable. No `enable_cuda_graph` yet — that's gated on the fixed-shape padded path in the follow-up.

- **`main()`** — at model load time, after the long-warmup, do an additional **representative-shape warmup that matches the lead-in size** (6–8 words, ~30 phonemes) so the first real prompt's chunk 0 doesn't trigger an on-demand kernel compile. The existing short and long warmups stay; the new one is third. Also: open the `sd.OutputStream` once at voice-pick time, kept open across prompts, and disposed on exit.

- **A new `AudioSink` class** (probably in a new `audio.py` or inline in `tts.py`) — encapsulates the persistent `sd.OutputStream`: `open(sample_rate)`, `write(samples)`, `drain()`, `close()`. Single ownership, single lifecycle. Falls back to the current per-prompt open path when the device can't be pre-opened (e.g. device changes, import-time failure).

- **`chunk_text()` / `_lead_in_split()`** — `LEAD_IN_MIN` lowered to ~5–6. Add a `LEAD_IN_HARD_CUT = True` mode that allows splitting mid-clause for the lead-in only. The first chunk's target size becomes a first-class parameter, exposed as `--lead-in-words N`.

- **`play_streaming()`** — the TTFW measurement points are refined: the existing `t_first_audio` (right before the first `stream.write`) stays, and the per-phase report (chunk-0 phonemize ms, chunk-0 infer ms, queue wait ms, stream-open ms, first-write-to-t_first_audio ms) becomes structured. If the persistent stream is in use, the "stream-open" component collapses to ~0.

- **`print_latency_report()`** — add a **generation RTF** line computed as `sum(per_chunk_infer_ms) / (audio_duration_ms * 1000)`, separate from the existing wall RTF. Keep both; clearly label which is which. Drop the misleading "slower than realtime ✗" verdict from the wall-RTF line.

- **`ResourceMonitor`** — replace `subprocess.run(["nvidia-smi", ...])` with `pynvml` (added as a dep). Lower the default sample interval to 50ms. Add an option to disable (for `--profile`-style runs where the sampler itself is the noise).

- **Long-lived worker thread** — replace the per-prompt `threading.Thread(target=worker, daemon=True).start()` with a single long-lived worker that processes jobs from a fresh `queue.Queue` per prompt. Saves ~5–20ms of thread-spawn per prompt and reduces variance.

- **`_timed_generate_chunk_audio()`** — unchanged. It's already the right per-call measurement seam.

### Cross-cutting decisions

- **Pinned CUDA wheel version stays.** The `onnxruntime-gpu[cuda,cudnn]>=1.26.0,<1.27.0` pin in `pyproject.toml` does not change. The new `pynvml` is added as a runtime dep; on non-NVIDIA systems, the monitor falls back to CPU/RAM only as it does today.

- **No model changes.** This PRD is entirely on the host side: ORT session options, pipeline structure, chunker. The fixed-shape padded path that would enable CUDA Graphs is a **follow-up PRD** because it requires changes inside the infer call (or a wrapper) and needs a small prototype to validate output equivalence. The follow-up will inline the relevant bits (token padding to MAX_PHONEME_LENGTH, attention mask, output slicing) — not full file contents — once prototyped.

- **CLI surface:**
  - `--lead-in-words N` (int) — minimum words in chunk 0 before the natural cut.
  - `--no-warm` — skip all model-load warmups.
  - `--profile` — emit a machine-readable per-prompt JSON line for diffing.
  - Existing `--provider`, `--no-stream` unchanged.

- **Sample rate is treated as a known constant** (`SAMPLE_RATE = 24000` from `kokoro_onnx.config`) at the seam where the persistent stream is opened, so the device is openable before any text arrives.

- **Fallbacks:** `--no-stream` and winsound paths are unchanged. The optimizations are no-ops on those paths; the per-phase report will still print, with stream-open collapsed.

### Decision from prototype (inline)

From `_bench2.py` — the measured relationship between phoneme count and infer time (≈3ms/phoneme on this hardware) implies that a 6-word lead-in (≈35 phonemes) infers in ~120ms, a 15-word chunk in ~270ms, and 400ms TTFW requires ≤135 phonemes (≈17 words). This sets the **upper bound for the lead-in target** at ~10 words and the **TTFW floor** at ~120ms (post-overhead) on CUDA.

## Testing Decisions

A good test in this codebase measures **external, observable behavior** (TTFW as seen by the user, generation RTF, audio duration, sample count) and not internal call counts. All tests are runnable as plain `uv run python <file>` with no pytest dependency, matching the existing style.

### What gets tested

- **End-to-end TTFW regression (`bench_ttfw.py` — new).** Generates the standard test paragraph (the one in `bench_firstsound.py` "44w" + the bundled paragraph from the CLI run) and reports: TTFW, chunk-0 phonemize ms, chunk-0 infer ms, stream-open ms, generation RTF, wall RTF. Has a `--assert-ttfw-under-ms N` mode for CI use. Asserts TTFW ≤ 400ms on the test hardware (with a generous margin) as the regression guard.

- **Lead-in chunking (`test_chunker.py` — new, lightweight).** Pure unit: feed the chunker text samples (short, long, mid-clause, all-conjunctions) and assert the lead-in chunk is ≤ `LEAD_IN_MIN + slack` words and that the remainder still parses. Asserts abbreviation protection (Dr., e.g., 3.14) is preserved across the split.

- **CUDA provider options smoke (`bench_firstsound.py` — extended).** Re-runs the existing phon/inf per-shape matrix with the new provider options; asserts the p95 of the 30w and 44w chunks drops to ≤ 1.5× the median (i.e. no 451ms-style spike).

- **AudioSink lifecycle (manual + a smoke check).** Open → write → close, opened-once-across-prompts, no double-open. The CLI run itself is the manual test; the smoke is a small script that drives AudioSink directly.

- **No regression in audio quality.** Output WAV file is byte-stable in sample count ± 1 sample per chunk boundary (crossfade math unchanged). The existing crossfade tests (manual `audiomoth` / ear check) remain the quality gate; we don't add new automated audio-quality tests in this PRD.

### Prior art in the codebase

- `bench_firstsound.py` — already isolates phonemize vs infer, uses median over N runs, and the warmup-shape analysis. The new tests reuse its pattern and copy its preamble (DLL preload, session creation, warmup) so behavior matches.
- `_bench2.py` — the chunk-size experiment that produced the 3ms/phone observation. The new `bench_ttfw.py` extends the same paragraph over the full streaming path.
- `tts.py` itself has no unit tests; the bench-script style is the established convention. We follow that convention rather than introducing pytest.

### What we explicitly do NOT test

- We do not test internal call counts (e.g. "create() called once per chunk") — that's an implementation detail.
- We do not test that CUDA Graphs are or aren't being used in this PRD (they're deferred). When the follow-up PRD lands, the test will be an end-to-end TTFW delta, not a "graph was captured" check.
- We do not test the winsound fallback's audio output (out of scope; unchanged path).

## Out of Scope

- **CUDA Graphs (fixed-shape padded inference)** — the biggest inference-throughput lever, but requires a prototype to validate output equivalence and a small change to the infer call (or a wrapper). Deferred to a follow-up PRD. Mentioned in user story 19 so it's not lost.
- **Pre-phonemize all chunks in parallel** with a thread pool — only worth doing once GPU is being saturated (today GPU util is 2%). Deferred until the CUDA Graphs work is in.
- **Voice-style pre-fetching / warmup of every voice** — only the active voice is in the hot path; other voices are not loaded into the session.
- **Batched inference across chunks** — Kokoro is a streaming model; batching helps throughput but not TTFW (batching's first output still waits for the batch to fill).
- **Changing the model's quantization / precision** — out of scope; would change audio quality and is a model-side change.
- **A new audio backend** (e.g. switching from sounddevice/PortAudio to a lower-overhead one) — keep sounddevice; the win from pre-opening is enough.
- **Network / HTTP API** — this PRD is local-CLI only.
- **GUI / web front-end** — out of scope.

## Further Notes

- **Why pre-open the stream and not just "open in a background thread"?** Background-thread open still pays the same cost, just on a different thread. Pre-opening is strictly better because the cost is paid at session start, before the user even types. The only reason to fall back to a per-prompt open is device-change or import-time failure, both rare.

- **Why is RTF being reframed now?** The current `RTF 1.07 ✗` is a property of the audio sink, not the model. Generation RTF (sum of infer times / audio duration) is ~0.06 — the model is ~16× faster than realtime. Showing wall-RTF as the headline "realtime" metric misleads users into thinking the model is too slow. The fix is to keep both numbers but make it clear which is which; the `✗` verdict moves off the wall line.

- **The follow-up CUDA Graphs PRD** will need: (a) a prototype that pads tokens to a fixed `MAX_PHONEME_LENGTH` shape with a mask, runs infer, slices output, and confirms audio matches the un-padded run; (b) a small wrapper around `kokoro.create()` that exposes the padded path; (c) integration with `play_streaming()` so the worker calls the padded path. This PRD does not include the prototype, but user story 19 makes the intent explicit so the next agent has a clear next step.

- **The `pynvml` swap** is small and gets rid of ~170 process spawns per run, which currently inflate the CPU peak and add noise to the resource report. It's included in this PRD because it's a clean, low-risk change that improves measurement quality while we're already touching the pipeline.

- **Risk: persistent `sd.OutputStream` on Windows.** WASAPI device open can be sticky — if the user unplugs/replugs a device, the stream may need to be re-opened. The AudioSink abstraction should detect write failures and fall back to a fresh open per prompt. This is handled, not ignored, in the design.

- **Risk: hard-cut lead-in prosody.** A 6-word hard cut in the middle of a clause can sound unnatural. The 40ms crossfade softens the join, but the lead-in's *content* still sounds slightly clipped. We mitigate by: (a) trying natural boundaries first within `LEAD_IN_MIN..LEAD_IN_MIN + 6` words, (b) only hard-cutting if no boundary is found, (c) exposing `--lead-in-words` so users who care more about prosody can raise it back up.

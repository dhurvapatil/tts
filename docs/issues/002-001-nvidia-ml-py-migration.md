---
id: 002-001
title: Migrate pynvml to nvidia-ml-py to remove the deprecation warning
triage: ready-for-agent
parent: PRD-002-warmup-and-buffering
blocked_by: none
---

## What to build

Replace the deprecated `pynvml` import with `nvidia-ml-py` (the new
canonical name) so the `FutureWarning: The pynvml package is deprecated`
message no longer pollutes every CLI run. The API surface we use is
identical: `nvmlInit`, `nvmlDeviceGetHandleByIndex`,
`nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetUtilizationRates`. The lock
file already has `nvidia-ml-py` as a transitive dependency of
`onnxruntime-gpu[cuda,cudnn]`, so this is a direct rename plus a
`pyproject.toml` change to make the dependency explicit.

The behavioral change is **zero** — the same GPU samples reach the
same report. The user-visible change is a clean startup with no
warning line under the "audio:" line.

## Acceptance criteria

- [ ] `_nvidia_query` (and any other pynvml user) imports from
  `nvidia_ml_py` instead of `pynvml`, with no other behavioral change.
- [ ] `pyproject.toml` `dependencies` lists `nvidia-ml-py>=0.1` and
  no longer lists `pynvml`.
- [ ] `uv sync` resolves without conflict; `pynvml` is no longer
  installed in the venv after the change.
- [ ] Running `uv run python tts.py` (or `bench_ttfw.py`) produces
  no `FutureWarning` from the pynvml import.
- [ ] The existing `tests/test_resource_monitor.py` still passes
  (with the assertion that `_nvidia_query`'s source does not call
  `subprocess` and does reference pynvml/nvml, the latter check
  updated to accept `nvidia_ml_py` as a valid import).
- [ ] A new test asserts that `nvidia_ml_py` is the imported module
  (i.e. the rename actually happened, not just a parallel import).
- [ ] `uv run python bench_ttfw.py --runs 1` still produces a
  sensible VRAM and GPU-util reading.

## Blocked by

None - can start immediately.

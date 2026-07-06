"""Tests for CUDA provider options builder.

Pure unit tests: assert the public function returns the right shape
and includes the critical knobs that affect first-inference latency
(cuDNN algo search, arena strategy, tunable ops).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


def test_cuda_options_is_dict():
    opts = tts.build_cuda_provider_options()
    assert isinstance(opts, dict)
    assert len(opts) > 0


def test_cuda_options_uses_heuristic_conv_algo_search():
    # Default must be HEURISTIC, not EXHAUSTIVE — the latter is the
    # cause of the 451ms p95 spike observed on cold first-call.
    opts = tts.build_cuda_provider_options()
    assert opts.get("cudnn_conv_algo_search") == "HEURISTIC"


def test_cuda_options_disables_arena_growth():
    # kSameAsRequested avoids the per-call arena-grow latency.
    opts = tts.build_cuda_provider_options()
    assert opts.get("arena_extend_strategy") == "kSameAsRequested"


def test_cuda_options_enables_tunable_ops():
    opts = tts.build_cuda_provider_options()
    assert opts.get("tunable_op_enable") is True


def test_cuda_options_pins_device_id():
    opts = tts.build_cuda_provider_options()
    assert opts.get("device_id") == 0


def test_select_providers_returns_options():
    providers, label, options = tts.select_providers("auto", return_options=True)
    assert label in ("CUDA", "CPU")
    if "CUDAExecutionProvider" in providers:
        # Options should be a dict (or list of dicts, depending on shape)
        assert options is not None


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

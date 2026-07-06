"""Tests for the ResourceMonitor.

Pure unit tests where possible.  The monitor's CPU/RAM reading uses
psutil (in-process) and the GPU reading uses pynvml when available,
falling back to a no-op.  The tests verify:
  - pynvml path is preferred over nvidia-smi subprocess
  - the monitor does not spawn external processes on each sample
  - the monitor exposes a stop()/join() lifecycle
  - the pynvml query is wrapped in a try/except so a missing driver
    or broken pynvml doesn't kill the monitor
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


def test_resource_monitor_uses_nvidia_ml_py():
    # The module-level function should return a (mem, util) pair
    # without spawning a subprocess.  The cheap way to assert that
    # is to check the implementation: there must be NO call to
    # subprocess.run within _nvidia_query.
    # Issue 002-001: must use the canonical nvidia-ml-py package
    # (the legacy pynvml PyPI package is deprecated and prints a
    # FutureWarning on import).  The module name stays `pynvml`
    # because nvidia-ml-py ships a backward-compat shim; what changes
    # is which PyPI package is in pyproject.toml and on disk.
    import inspect
    import re
    src = inspect.getsource(tts._nvidia_query)
    assert not re.search(r"subprocess\.(run|Popen|call|check)", src), (
        f"_nvidia_query must not spawn subprocesses; got:\n{src}"
    )
    assert "nvml" in src.lower(), (
        f"_nvidia_query should still call nvml API; got:\n{src}"
    )


def test_resource_monitor_does_not_import_legacy_pynvml():
    """Issue 002-001: the legacy pynvml import must be gone from
    _nvidia_query (it raises a FutureWarning on import)."""
    import inspect
    src = inspect.getsource(tts._nvidia_query)
    # Forbid a top-level "import pynvml" or "from pynvml" line.
    assert not re.search(r"^\s*(import pynvml|from pynvml)\b", src, re.MULTILINE), (
        f"_nvidia_query must not import the deprecated pynvml package; got:\n{src}"
    )


def test_pyproject_uses_nvidia_ml_py_not_legacy_pynvml():
    """Issue 002-001: the canonical nvidia-ml-py package must be in
    pyproject.toml; the deprecated pynvml package must not be listed."""
    from pathlib import Path
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "nvidia-ml-py" in text, (
        f"pyproject.toml must declare nvidia-ml-py (issue 002-001); got:\n{text}"
    )
    # The legacy `pynvml` package must not be a direct dependency.
    # (A bare word match is fine here: pyproject is a small TOML file.)
    assert re.search(r'^\s*"pynvml"\s*=', text, re.MULTILINE) is None, (
        f"pyproject.toml must not declare the legacy pynvml package (issue 002-001); got:\n{text}"
    )


def test_resource_monitor_lifecycle():
    mon = tts.ResourceMonitor(interval=0.05)
    with mon:
        # Give it a few samples
        import time
        time.sleep(0.15)
    # After exit, no thread is running and peak values are populated
    assert mon.peak_cpu >= 0.0
    assert mon.peak_ram > 0


def test_resource_monitor_pynvml_unavailable_does_not_raise():
    """If pynvml raises on init, the monitor should still work and
    report no GPU (has_gpu() == False).  We simulate this by patching
    _nvidia_query to always raise."""

    import tts as _tts
    orig = _tts._nvidia_query

    def boom():
        raise RuntimeError("simulated pynvml failure")

    _tts._nvidia_query = boom
    try:
        mon = tts.ResourceMonitor(interval=0.05)
        with mon:
            import time
            time.sleep(0.1)
        assert mon.has_gpu is False
    finally:
        _tts._nvidia_query = orig


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

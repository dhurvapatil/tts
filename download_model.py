"""Download Kokoro-82M ONNX model + voice table into ./models/.

Uses only the standard library so it works before `uv sync` is run.
Re-running is safe; existing files are skipped.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

# Pinned to the model-files-v1.0 release of the kokoro-onnx project.
URLS: dict[str, str] = {
    "kokoro-v1.0.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/"
        "download/model-files-v1.0/kokoro-v1.0.onnx"
    ),
    "voices-v1.0.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/"
        "download/model-files-v1.0/voices-v1.0.bin"
    ),
}


def download(url: str, dest: Path) -> None:
    print(f"  ↓ {dest.name}  ({url.rsplit('/', 1)[-1]})")
    req = urllib.request.Request(url, headers={"User-Agent": "kokoro-tts-cli/0.1"})
    with urllib.request.urlopen(req) as r, dest.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        chunk = 1024 * 256
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if total:
                pct = got * 100 / total
                bar = "█" * int(pct // 4) + "·" * (25 - int(pct // 4))
                print(f"    {bar} {pct:5.1f}%  {got/1024/1024:6.1f} MB", end="\r")
        if total:
            print()
    print(f"  ✓ {dest.name}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"  target: {MODELS}\n")
    for name, url in URLS.items():
        out = MODELS / name
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"  · {name} already present, skipping")
            continue
        try:
            download(url, out)
        except Exception as e:
            print(f"\n  ! download failed for {name}: {e}", file=sys.stderr)
            if out.exists():
                out.unlink()
            return 1
    print("\n  done.  next:  uv run tts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

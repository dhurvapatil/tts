import os, time, statistics
from pathlib import Path
_R = Path(__file__).resolve().parent
_n = _R / ".venv" / "Lib" / "site-packages" / "nvidia"
if _n.exists():
    _ex = [str(d.resolve()) for d in sorted(_n.glob("*/bin")) if d.is_dir()]
    if _ex:
        os.environ["PATH"] = ";".join(_ex) + ";" + os.environ.get("PATH", "")
        for _p in _ex:
            try: os.add_dll_directory(_p)
            except Exception: pass
import onnxruntime as ort
from kokoro_onnx import Kokoro
sess = ort.InferenceSession(str(_R / "models" / "kokoro-v1.0.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
k = Kokoro.from_session(sess, str(_R / "models" / "voices-v1.0.bin"))
VOICE = "bm_george"
_ = k.create("ready.", voice=VOICE, lang="en-us", speed=1.0)
cuts = [
    ("At precisely 07:43 a.m. on June 25, 2026, the autonomous research vessel Asteria-7 departed the harbor carrying 128 scientists,", 19),
    ("At precisely 07:43 a.m. on June 25, 2026, the autonomous research vessel Asteria-7 departed the harbor carrying 128 scientists, engineers, and observers from 17 countries,", 25),
    ("At precisely 07:43 a.m. on June 25, 2026, the autonomous research vessel Asteria-7 departed the harbor carrying 128 scientists, engineers, and observers from 17 countries, each tasked with evaluating whether next-generation artificial intelligence systems", 33),
]
print(f"{'label':10} {'w':>3} {'ph':>4} {'inf_ms':>7} {'audio_s':>8}")
for txt, nw in cuts:
    ph = k.tokenizer.phonemize(txt, lang="en-us", norm=True)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        s, sr = k.create(ph, voice=VOICE, lang="en-us", is_phonemes=True, speed=1.0)
        ts.append((time.perf_counter() - t) * 1000)
    print(f"{str(nw)+'w':10} {nw:>3} {len(ph):>4} {statistics.median(ts):>7.1f} {len(s)/sr:>8.2f}")
print("PHYSICS: 400ms first-sound needs <=135 phones ~= 17-18 words at measured 3ms/phone")
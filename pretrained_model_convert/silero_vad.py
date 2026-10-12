"""
Silero VAD v5 (silero_vad.onnx) wrapper -- C#-portable logic.

Calling convention that actually works for this ONNX (the important part):
  * 16 kHz, process in 512-sample hops.
  * The model input is NOT 512 samples -- it is  context(64) + chunk(512) = 576.
    The last 64 samples of the previous chunk are prepended as context; the first
    chunk uses 64 zeros. Feeding bare 512 samples silently produces ~0 output.
  * state: (2, 1, 128) float32, zeros at start, fed back each call.
  * sr: int64 scalar 16000.

All of this is a plain loop over fixed-size buffers -> trivial to port to C#.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
SR = 16000
WIN = 512          # hop / new samples per step
CONTEXT = 64       # samples prepended from previous step


class SileroVAD:
    def __init__(self, model_path="silero_vad.onnx", providers=None, sess_options=None):
        p = Path(model_path)
        if not p.is_absolute():
            p = HERE / p
        self.sess = ort.InferenceSession(str(p), sess_options=sess_options,
                                         providers=providers or ["CPUExecutionProvider"])
        self._sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)

    def probs(self, wav: np.ndarray) -> np.ndarray:
        """mono 16 kHz waveform -> per-frame speech probability (one per 512 samples)."""
        self.reset()
        wav = np.asarray(wav, dtype=np.float32)
        out = []
        for i in range(0, len(wav), WIN):
            chunk = wav[i:i + WIN]
            if len(chunk) < WIN:
                chunk = np.pad(chunk, (0, WIN - len(chunk)))
            inp = np.concatenate([self.context, chunk]).astype(np.float32)[None]  # (1,576)
            p, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": self._sr})
            out.append(float(p.reshape(-1)[0]))
            self.context = chunk[-CONTEXT:]
        return np.array(out, dtype=np.float32)


def get_speech_timestamps(probs, threshold=0.5, min_speech_ms=250,
                          min_silence_ms=100, speech_pad_ms=30,
                          total_samples=None, sr=SR, win=WIN):
    """Frame probabilities -> list of [start, end] sample indices (hysteresis smoothing)."""
    neg = threshold - 0.15
    min_speech = sr * min_speech_ms / 1000
    min_silence = sr * min_silence_ms / 1000
    pad = int(sr * speech_pad_ms / 1000)
    if total_samples is None:
        total_samples = len(probs) * win

    segs, triggered, start, temp_end = [], False, 0, 0
    for i, p in enumerate(probs):
        cur = i * win
        if p >= threshold:
            temp_end = 0
            if not triggered:
                triggered, start = True, cur
        elif p < neg and triggered:
            if temp_end == 0:
                temp_end = cur
            if cur - temp_end >= min_silence:
                if temp_end - start >= min_speech:
                    segs.append([start, temp_end])
                triggered, temp_end = False, 0
    if triggered and total_samples - start >= min_speech:
        segs.append([start, total_samples])

    # pad, clamp, merge overlaps
    out = []
    for s, e in segs:
        s, e = max(0, s - pad), min(total_samples, e + pad)
        if out and s <= out[-1][1]:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def extract_speech(wav: np.ndarray, vad: "SileroVAD", **kw) -> np.ndarray:
    """Return only the speech samples of wav, concatenated. Empty if no speech."""
    pr = vad.probs(wav)
    ts = get_speech_timestamps(pr, total_samples=len(wav), **kw)
    if not ts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([wav[s:e] for s, e in ts]).astype(np.float32)


if __name__ == "__main__":
    import soundfile as sf
    from scipy.signal import resample_poly
    vad = SileroVAD()
    files = {f.name.split("_")[0]: f for f in (HERE.parent / "samples_inmu3").glob("*_*.mp3")}
    for k in sorted(files, key=int):
        w, sr = sf.read(files[k], always_2d=True)
        w = w.mean(1).astype(np.float32)
        if sr != SR:
            g = np.gcd(sr, SR); w = resample_poly(w, SR // g, sr // g).astype(np.float32)
        sp = extract_speech(w, vad)
        print(f"{k}: total={len(w)/SR:4.1f}s  speech={len(sp)/SR:4.1f}s  "
              f"{'<<< NO SPEECH' if len(sp) == 0 else ''}")

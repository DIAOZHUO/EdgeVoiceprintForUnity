"""
Voice-print verification test (mirrors the Quest enroll/verify use-case), now
with Silero VAD gating.

  enroll:  ref[spk] = mean of embed(speech_of(clip)) over the first --enroll
           clips of each speaker (numeric file order)
  verify, per remaining clip:
     1. Silero VAD over the clip.
        - max speech prob < VAD_REJECT  -> "?"  (non-speech, reject)
        - else keep only the speech samples (fall back to full clip if < 1.5 s)
     2. cosine(embed(speech), ref[spk]) for every speaker -> nearest speaker.

Runs twice (VAD off / on) to show the effect.

Dataset: samples_inmu3/ (3 speakers: 野獣, MUR, KMR). Labels come from the file
names "<n>_<speaker>「...」.mp3"; a name without 「 (the door sound) is non-speech N.

  python -m test_result.voiceprint_on_clip                 # enroll first 3 clips per speaker
  python -m test_result.voiceprint_on_clip --enroll 1 --model pretrained_model_convert/ecapa_tdnn_int8.onnx
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

from pretrained_model_convert.fbank_cs import compute_fbank, SAMPLE_RATE
from pretrained_model_convert.silero_vad import SileroVAD, get_speech_timestamps

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "pretrained_model_convert" / "ecapa_tdnn.onnx"
VAD_REJECT = 0.3     # max frame prob below this -> non-speech
MIN_SPEECH_S = 1.5   # if VAD speech shorter than this, embed the full clip (don't over-trim)

def load_mono_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wav


def load_dataset(sample_dir: Path):
    """-> {key: path}, {key: label}; keys are the leading clip numbers."""
    files, labels = {}, {}
    for f in sample_dir.glob("*.mp3"):
        m = re.match(r"\s*(\d+)", f.name)
        if not m:
            continue                                  # e.g. all.mp3
        key = m.group(1)
        files[key] = f
        spk = re.match(r"\s*\d+[_ ]?([^「]+)「", f.name)
        labels[key] = spk.group(1).strip() if spk else "N"
    return files, labels


def run_clip_test(sess, vad, wavs, labels, n_enroll=3, use_vad=True, timings=None, log=print):
    """Enroll the first n_enroll clips of each speaker, then identify every other clip.

    wavs/labels: {key: mono 16 kHz wav}, {key: speaker or "N"}. If timings is a dict, the
    per-call fbank / ECAPA times (ms) of the test clips are appended to timings["fbank_ms"]
    and timings["ecapa_ms"].
    -> {"speakers", "enroll", "rows": [(key, {spk: cos} | None, pred, truth)],
        "ok": {spk: n}, "n": {spk: n}, "noise_ok", "noise_n"}
    """
    order = sorted(wavs, key=int)
    speakers = list(dict.fromkeys(labels[k] for k in order if labels[k] != "N"))
    enroll = {s: [k for k in order if labels[k] == s][:n_enroll] for s in speakers}
    test = [k for k in order if not any(k in v for v in enroll.values())]

    def embed(wav, timed=False):
        t0 = time.perf_counter()
        feat = compute_fbank(wav)[np.newaxis].astype(np.float32)
        t1 = time.perf_counter()
        e = sess.run(None, {"waveform_feature": feat})[0].reshape(-1)
        t2 = time.perf_counter()
        if timed and timings is not None:
            timings.setdefault("fbank_ms", []).append((t1 - t0) * 1000)
            timings.setdefault("ecapa_ms", []).append((t2 - t1) * 1000)
        return e / (np.linalg.norm(e) + 1e-9)

    def vad_gate(wav):
        """-> (is_speech, speech_wav). Rejects non-speech by max prob."""
        pr = vad.probs(wav)
        if pr.max() < VAD_REJECT:
            return False, None
        ts = get_speech_timestamps(pr, total_samples=len(wav))
        sp = np.concatenate([wav[s:e] for s, e in ts]) if ts else np.zeros(0, np.float32)
        if len(sp) < MIN_SPEECH_S * SAMPLE_RATE:     # too little -> use full clip
            sp = wav
        return True, sp.astype(np.float32)

    refs = {}
    for s in speakers:
        v = 0
        for k in enroll[s]:
            wav = wavs[k]
            if use_vad:
                is_sp, sp = vad_gate(wav)
                if is_sp:
                    wav = sp
                else:
                    log(f"  ! enroll clip {k} ({s}) rejected by VAD as non-speech; using the full clip")
            v = v + embed(wav)
        refs[s] = v / np.linalg.norm(v)

    res = {"speakers": speakers, "enroll": enroll, "rows": [],
           "ok": {s: 0 for s in speakers}, "n": {s: 0 for s in speakers}, "noise_ok": 0, "noise_n": 0}
    for key in test:
        truth, wav = labels[key], wavs[key]
        if use_vad:
            is_sp, sp = vad_gate(wav)
            if not is_sp:
                res["rows"].append((key, None, "?", truth))
                if truth == "N":
                    res["noise_ok"] += 1; res["noise_n"] += 1
                else:
                    res["n"][truth] += 1
                continue
            wav = sp
        e = embed(wav, timed=True)
        cos = {s: float(e @ refs[s]) for s in speakers}
        pred = max(cos, key=cos.get)
        if truth == "N":
            res["noise_n"] += 1
        else:
            res["n"][truth] += 1; res["ok"][truth] += pred == truth
        res["rows"].append((key, cos, pred, truth))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples_inmu3", help="clip folder (relative to the repo root)")
    ap.add_argument("--model", default=str(MODEL), help="ECAPA onnx (fp32 default; Unity ships ecapa_tdnn_int8.onnx)")
    ap.add_argument("--enroll", type=int, default=3, help="first N clips of each speaker are enrolled")
    args = ap.parse_args()

    sess = ort.InferenceSession(str(ROOT / args.model), providers=["CPUExecutionProvider"])
    vad = SileroVAD()
    files, labels = load_dataset(ROOT / args.samples)
    wavs = {k: load_mono_16k(files[k]) for k in sorted(files, key=int)}

    header = None
    for use_vad in (False, True):
        res = run_clip_test(sess, vad, wavs, labels, args.enroll, use_vad)
        speakers, enroll = res["speakers"], res["enroll"]
        if header is None:
            header = (f"model {Path(args.model).name} | {args.samples}: {len(wavs)} clips | "
                      + ", ".join(f"{s} enroll {enroll[s]} ({sum(len(wavs[k]) for k in enroll[s]) / SAMPLE_RATE:.1f}s)"
                                  for s in speakers))
            print(header)
        print(f"\n===== VAD {'ON' if use_vad else 'OFF'} =====")
        print(f"{'clip':<6}" + "".join(f"{s:>8}" for s in speakers) + f"{'pred':>8}{'truth':>8}  ok")
        for key, cos, pred, truth in res["rows"]:
            _report(key, speakers, cos, pred, truth)
        ok, n = res["ok"], res["n"]
        tot_ok, tot_n = sum(ok.values()), sum(n.values())
        print(f"  speaker accuracy: {tot_ok}/{tot_n} = {tot_ok / tot_n:.0%}   ("
              + ", ".join(f"{s} {ok[s]}/{n[s]}" for s in speakers)
              + (f"; non-speech rejected {res['noise_ok']}/{res['noise_n']}" if res["noise_n"] else "") + ")")


def _report(key, speakers, cos, pred, truth):
    cells = "".join(f"{cos[s]:>8.3f}" if cos else f"{'--':>8}" for s in speakers)
    mark = "✓" if pred == truth else ("·" if truth == "N" and pred == "?" else "✗")
    print(f"{key:<6}{cells}{pred:>8}{truth:>8}  {mark}")


if __name__ == "__main__":
    main()

"""
Speaker identification over a whole conversation (<samples>/all.mp3), using the
first --enroll clips of each speaker in <samples>/ as the enrolled library.

VAD-segment diarization (one ECAPA pass per speech segment, no overlap):
    enroll:  ref[spk] = mean embedding of the speaker's first N clips (VAD-gated,
             same as voiceprint_on_clip.py)
    all.mp3: Silero VAD -> speech segments (get_speech_timestamps); segments
             longer than MAX_SEG_S are cut into equal non-overlapping pieces
             (each in (MAX_SEG_S/2, MAX_SEG_S]) so a speaker change inside one
             region is still caught; ECAPA-embed each piece; cosine vs refs.

Every speech sample goes through ECAPA exactly once. On Quest this maps to: run
ECAPA when VAD closes a segment, or every MAX_SEG_S while a segment is still open.

Scoring: each labeled clip is located inside all.mp3 by normalized
cross-correlation; clips that match (ncc >= --min-ncc) give the ground truth
timeline. Accuracy is the share of VAD-speech 10 ms frames inside those clips
that get the right speaker; enrollment clips are excluded (they are part of
all.mp3), and speech outside located clips is reported as unlabeled.

  python -m test_result.voiceprint_diarize_all                 # samples_inmu3/, first 3 clips per speaker
  python -m test_result.voiceprint_diarize_all --enroll 1 --turns
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import fftconvolve

from pretrained_model_convert.fbank_cs import compute_fbank, SAMPLE_RATE as SR
from pretrained_model_convert.silero_vad import SileroVAD, get_speech_timestamps
from test_result.voiceprint_on_clip import (
    ROOT, MODEL, VAD_REJECT, MIN_SPEECH_S, load_mono_16k, load_dataset,
)

MAX_SEG_S = 2.0                  # longer VAD segments are split into pieces <= this
FR = SR // 100                   # 10 ms scoring frames


def embed(wav, sess):
    feat = compute_fbank(wav)[np.newaxis].astype(np.float32)
    e = sess.run(None, {"waveform_feature": feat})[0].reshape(-1)
    return e / (np.linalg.norm(e) + 1e-9)


def enroll_refs(wavs, enroll, sess, vad):
    """{speaker: [keys]} -> {speaker: unit mean embedding}, VAD-gated like voiceprint_on_clip."""
    refs = {}
    for spk, keys in enroll.items():
        v = 0
        for k in keys:
            w = wavs[k]
            pr = vad.probs(w)
            if pr.max() >= VAD_REJECT:
                ts = get_speech_timestamps(pr, total_samples=len(w))
                sp = np.concatenate([w[s:e] for s, e in ts]) if ts else w[:0]
                if len(sp) >= MIN_SPEECH_S * SR:
                    w = sp
            v = v + embed(w, sess)
        refs[spk] = v / np.linalg.norm(v)
    return refs


def speech_pieces(probs, total_samples, max_seg_s=MAX_SEG_S):
    """VAD probs -> [(start, end)] sample ranges, long segments split evenly."""
    pieces, max_seg = [], int(max_seg_s * SR)
    for s, e in get_speech_timestamps(probs, total_samples=total_samples):
        n = max(1, math.ceil((e - s) / max_seg))
        cuts = np.linspace(s, e, n + 1).astype(int)
        pieces += list(zip(cuts[:-1], cuts[1:]))
    return pieces


def diarize(wav, refs, probs, sess, max_seg_s=MAX_SEG_S):
    """-> [(start_s, end_s, label, {spk: cos})], one ECAPA call per piece."""
    out = []
    for s, e in speech_pieces(probs, len(wav), max_seg_s):
        emb = embed(wav[s:e], sess)
        cos = {k: float(emb @ r) for k, r in refs.items()}
        out.append((s / SR, e / SR, max(cos, key=cos.get), cos))
    return out


def locate_clips(wav, clips):
    """{key: clip wav} -> {key: (start_s, end_s, ncc)} best match of each clip inside wav."""
    e2 = np.concatenate([[0.0], np.cumsum(wav.astype(np.float64) ** 2)])
    out = {}
    for k, c in clips.items():
        if len(c) > len(wav):
            continue
        corr = fftconvolve(wav, c[::-1].astype(np.float64), mode="valid")
        energy = np.sqrt(np.maximum(e2[len(c):] - e2[:-len(c)], 0)) * np.linalg.norm(c) + 1e-9
        ncc = corr / energy
        i = int(ncc.argmax())
        out[k] = (i / SR, (i + len(c)) / SR, float(ncc[i]))
    return out


def ground_truth(wav, probs, wavs, labels, enroll, min_ncc=0.6):
    """Locate every labelled clip inside wav -> per-10 ms-frame ground truth.

    Clips matched with ncc >= min_ncc label their frames; enrollment clips are part of wav and
    are excluded from scoring; only VAD speech frames (probs) are scored.
    -> {"truth": frame labels ("" = unlabelled), "scored": frames to score, "speech": VAD speech
        frames, "noise": non-speech clip frames detected as speech, "found": {key: (s, e, ncc)},
        "missing": [keys]}
    """
    order = sorted(wavs, key=int)
    speakers = [s for s in enroll]
    loc = locate_clips(wav, {k: wavs[k] for k in order})
    found = {k: v for k, v in loc.items() if v[2] >= min_ncc}
    nf = len(wav) // FR
    truth = np.full(nf, "", dtype=object)
    excluded = np.zeros(nf, bool)
    enrolled = {k for v in enroll.values() for k in v}
    for k, (s, e, _) in found.items():
        truth[int(s * 100):int(e * 100)] = labels[k]
        if k in enrolled:
            excluded[int(s * 100):int(e * 100)] = True
    speech = np.zeros(nf, bool)
    for s, e in get_speech_timestamps(probs, total_samples=len(wav)):
        speech[s // FR:e // FR] = True
    return {"truth": truth, "speech": speech,
            "scored": speech & ~excluded & np.isin(truth, speakers),
            "noise": speech & (truth == "N") & ~excluded,
            "found": found, "missing": [k for k in order if k not in found]}


def score_track(track, gt, speakers):
    """Frame accuracy of a diarize() track against ground_truth().
    -> {"scored_s", "accuracy" (None if nothing scored),
        "per_speaker": {spk: {"seconds", "accuracy", "predicted_s": {spk: seconds}}}}
    """
    truth, scored = gt["truth"], gt["scored"]
    pred = np.full(len(truth), "", dtype=object)
    for s, e, lab, _ in track:
        pred[int(s * 100):int(e * 100)] = lab
    out = {"scored_s": scored.sum() / 100,
           "accuracy": float((pred[scored] == truth[scored]).mean()) if scored.sum() else None,
           "per_speaker": {}}
    for t in speakers:
        m = scored & (truth == t)
        if m.sum():
            out["per_speaker"][t] = {"seconds": m.sum() / 100, "accuracy": float((pred[m] == t).mean()),
                                     "predicted_s": {p: (pred[m] == p).sum() / 100 for p in speakers}}
    return out


def fmt(t):
    return f"{int(t)//60:02d}:{t%60:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples_inmu3", help="clip folder with all.mp3 (relative to the repo root)")
    ap.add_argument("--model", default=str(MODEL), help="ECAPA onnx (fp32 default; Unity ships ecapa_tdnn_int8.onnx)")
    ap.add_argument("--enroll", type=int, default=3, help="first N clips of each speaker are enrolled")
    ap.add_argument("--max-seg", type=float, default=MAX_SEG_S, help="split VAD segments into pieces <= this (s)")
    ap.add_argument("--min-ncc", type=float, default=0.6, help="clip-location confidence needed for ground truth")
    ap.add_argument("--turns", action="store_true", help="print the merged speaker turns")
    args = ap.parse_args()

    sess = ort.InferenceSession(str(ROOT / args.model), providers=["CPUExecutionProvider"])
    vad = SileroVAD()
    files, labels = load_dataset(ROOT / args.samples)
    order = sorted(files, key=int)
    wavs = {k: load_mono_16k(files[k]) for k in order}
    speakers = list(dict.fromkeys(labels[k] for k in order if labels[k] != "N"))
    enroll = {s: [k for k in order if labels[k] == s][:args.enroll] for s in speakers}
    refs = enroll_refs(wavs, enroll, sess, vad)

    wav = load_mono_16k(ROOT / args.samples / "all.mp3")
    t0 = time.perf_counter()
    probs = vad.probs(wav)
    t_vad = time.perf_counter() - t0
    track = diarize(wav, refs, probs, sess, args.max_seg)
    t_all = time.perf_counter() - t0

    speech_s = sum(e - s for s, e, *_ in track)
    print(f"{args.samples}/all.mp3 = {len(wav)/SR:.1f}s | model {Path(args.model).name} | enroll "
          + ", ".join(f"{s} {enroll[s]}" for s in speakers))
    print(f"{len(track)} ECAPA calls over {speech_s:.1f}s speech (max piece {args.max_seg}s); "
          f"wall {t_all:.2f}s (VAD {t_vad:.2f}s) = {len(wav)/SR/t_all:.0f}x real time")

    if args.turns:
        turns = []                     # merge contiguous same-speaker pieces (gap < 0.3 s) into turns
        for s, e, lab, _ in track:
            if turns and turns[-1][2] == lab and s - turns[-1][1] < 0.3:
                turns[-1][1] = e
            else:
                turns.append([s, e, lab])
        print("=== speaker turns ===")
        for s, e, lab in turns:
            print(f"  {fmt(s)}–{fmt(e)}  ({e-s:4.1f}s)  {lab}")
    tot = {s: sum(e - b for b, e, lab, _ in track if lab == s) for s in speakers}
    print("talk time: " + "   ".join(f"{s}={t:.1f}s" for s, t in tot.items()))

    # ---- ground truth from clip locations ------------------------------------------
    gt = ground_truth(wav, probs, wavs, labels, enroll, args.min_ncc)
    res = score_track(track, gt, speakers)
    print(f"\nground truth: {len(gt['found'])}/{len(order)} clips located (ncc >= {args.min_ncc}); "
          f"not in all.mp3 or not matched: {', '.join(gt['missing']) if gt['missing'] else '-'}")
    print(f"scored speech {res['scored_s']:.1f}s (enroll clips excluded); "
          f"unlabeled speech {(gt['speech'] & (gt['truth'] == '')).sum()/100:.1f}s")
    if res["accuracy"] is not None:
        print(f"frame accuracy: {res['accuracy']:.1%}")
        w = max(len(s) for s in speakers) + 2
        head = "truth / pred"
        print(f"  {head:<{w + 8}}" + "".join(f"{s:>{w + 2}}" for s in speakers) + "   acc")
        for t, r in res["per_speaker"].items():
            row = "".join(f"{r['predicted_s'][p]:>{w + 1}.1f}s" for p in speakers)
            print(f"  {t:<{w}} ({r['seconds']:4.1f}s){row}   {r['accuracy']:.0%}")
        if gt["noise"].sum():
            print(f"  non-speech clips detected as speech: {gt['noise'].sum()/100:.1f}s")

if __name__ == "__main__":
    main()

"""
Reproducible voice-print benchmark: regenerates "Voiceprint Benchmark.html" (+ a JSON file
with the raw numbers) from the models and clips in this repo.

    python -m test_result.benchmark
    python -m test_result.benchmark --samples samples_inmu3 --enroll 3 --threads 1

What is measured (onnxruntime CPUExecutionProvider, --threads intra-op threads, default 1):
  * Speaker accuracy  -- test_result/voiceprint_on_clip.py logic: enroll the first --enroll
                         clips of each speaker, identify every other clip; ECAPA fp32 / int8,
                         VAD off / on.
  * Latency per call  -- fbank and ECAPA (fp32, int8) on the clips that are identified with
                         VAD on; mean / median / p90 in ms, after a warm-up call.
  * Silero VAD speed  -- Silero VAD v5 over <samples>/all.mp3, x real time.
  * Diarization       -- test_result/voiceprint_diarize_all.py over <samples>/all.mp3
                         (VAD + one ECAPA call per <= 2 s piece): wall time, x real time and
                         frame accuracy against the clips located inside all.mp3 (enrollment
                         clips excluded, only VAD speech frames scored).

Pipeline: Silero VAD v5 -> fbank (pretrained_model_convert/fbank_cs.py) -> ECAPA-TDNN.
"""
import argparse
import datetime
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pretrained_model_convert.fbank_cs import compute_fbank, SAMPLE_RATE as SR
from pretrained_model_convert.silero_vad import SileroVAD
from test_result.voiceprint_on_clip import ROOT, load_dataset, load_mono_16k, run_clip_test
from test_result.voiceprint_diarize_all import MAX_SEG_S, diarize, enroll_refs, ground_truth, score_track

HERE = Path(__file__).resolve().parent
MODELS = {
    "fp32": ROOT / "pretrained_model_convert" / "ecapa_tdnn.onnx",
    "int8": ROOT / "pretrained_model_convert" / "ecapa_tdnn_int8.onnx",
}


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def session_options(threads: int) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def stats_ms(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "calls": int(a.size)}


def warm_up(sess, vad):
    feat = compute_fbank(np.random.default_rng(0).standard_normal(2 * SR).astype(np.float32) * 0.1)
    sess.run(None, {"waveform_feature": feat[None]})
    vad.probs(np.zeros(SR, np.float32))


def run(args) -> dict:
    so = session_options(args.threads)
    vad = SileroVAD(sess_options=so)
    files, labels = load_dataset(ROOT / args.samples)
    wavs = {k: load_mono_16k(files[k]) for k in sorted(files, key=int)}
    audio = load_mono_16k(ROOT / args.samples / "all.mp3")
    audio_s = len(audio) / SR

    out = {"accuracy": {}, "latency_ms": {}, "diarization": {}}
    gt = None
    for name, path in MODELS.items():
        sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        warm_up(sess, vad)
        for use_vad in (False, True):
            timings = {} if use_vad else None
            res = run_clip_test(sess, vad, wavs, labels, args.enroll, use_vad, timings, log=lambda *_: None)
            ok, n = sum(res["ok"].values()), sum(res["n"].values())
            out["accuracy"][f"{name}|{'on' if use_vad else 'off'}"] = {"ok": ok, "n": n, "pct": 100 * ok / n}
            if use_vad:
                out["latency_ms"][f"ecapa_{name}"] = stats_ms(timings["ecapa_ms"])
                out["latency_ms"].setdefault("fbank", stats_ms(timings["fbank_ms"]))
                out["speakers"], out["enroll"] = res["speakers"], res["enroll"]
                out["speech_clips"] = n

        # whole-file diarization, timed from raw audio to labelled pieces
        refs = enroll_refs(wavs, out["enroll"], sess, vad)
        t0 = time.perf_counter()
        probs = vad.probs(audio)
        t_vad = time.perf_counter() - t0
        track = diarize(audio, refs, probs, sess, MAX_SEG_S)
        wall = time.perf_counter() - t0
        if gt is None:   # model independent: built once, outside the timed section
            gt = ground_truth(audio, probs, wavs, labels, out["enroll"])
            out["diarization_gt"] = {"located": len(gt["found"]), "clips": len(wavs),
                                     "scored_s": float(gt["scored"].sum() / 100)}
        out["diarization"][name] = {"wall_s": wall, "vad_s": t_vad, "ecapa_calls": len(track),
                                    "x_realtime": audio_s / wall,
                                    "accuracy": score_track(track, gt, out["speakers"])}

    t0 = time.perf_counter()
    vad.probs(audio)
    vad_s = time.perf_counter() - t0
    out["vad"] = {"seconds": vad_s, "x_realtime": audio_s / vad_s}

    out["env"] = {
        "cpu": cpu_name(), "onnxruntime": ort.__version__, "threads": args.threads,
        "python": platform.python_version(), "samples": args.samples, "audio_s": audio_s,
        "clips": len(wavs), "max_piece_s": MAX_SEG_S,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "command": f"python -m test_result.benchmark --samples {args.samples} --enroll {args.enroll} "
                   f"--threads {args.threads}",
    }
    return out


def print_summary(r: dict) -> None:
    e = r["env"]
    print(f"{e['cpu']} | onnxruntime {e['onnxruntime']} | {e['threads']} thread(s) | {e['samples']}")
    print(f"enroll: " + ", ".join(f"{s} {k}" for s, k in r["enroll"].items()) + f" | {r['speech_clips']} speech clips")
    print("\nspeaker accuracy (voiceprint_on_clip):")
    for key, a in r["accuracy"].items():
        model, vad = key.split("|")
        print(f"  ECAPA {model:4s} VAD {vad:3s}: {a['pct']:5.1f}%  ({a['ok']}/{a['n']})")
    print("\nlatency per call (ms):")
    for key, s in r["latency_ms"].items():
        print(f"  {key:11s} mean {s['mean']:6.1f}  median {s['median']:6.1f}  p90 {s['p90']:6.1f}  ({s['calls']} calls)")
    print(f"\nSilero VAD v5: {e['audio_s']:.1f} s audio in {r['vad']['seconds']:.2f} s = {r['vad']['x_realtime']:.0f}x real time")
    print("diarization (voiceprint_diarize_all):")
    g = r["diarization_gt"]
    for model, d in r["diarization"].items():
        a = d["accuracy"]
        print(f"  ECAPA {model:4s}: {d['wall_s']:.2f} s (VAD {d['vad_s']:.2f} s, {d['ecapa_calls']} ECAPA calls) "
              f"= {d['x_realtime']:.0f}x real time | frame accuracy {100 * a['accuracy']:.1f}% ("
              + ", ".join(f"{s} {100 * x['accuracy']:.0f}%" for s, x in a["per_speaker"].items()) + ")")
    print(f"  ground truth: {g['located']}/{g['clips']} clips located in all.mp3, {g['scored_s']:.1f} s speech scored")


def write_html(r: dict, path: Path) -> None:
    path.write_text(HTML.replace("__DATA__", json.dumps(r, ensure_ascii=False)), encoding="utf-8")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voiceprint Benchmark</title>
<style>
  :root{
    color-scheme: light;
    --plane:#f9f9f7; --surface:#fcfcfb;
    --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
    --on:#2a78d6; --off:#eb6834; --mag:#256abf;
    --tip-bg:#0b0b0b; --tip-fg:#fcfcfb; --radius:14px;
  }
  @media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
    --on:#3987e5; --off:#d95926; --mag:#3987e5; --tip-bg:#fcfcfb; --tip-fg:#0b0b0b;
  }}
  :root[data-theme="dark"]{
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
    --on:#3987e5; --off:#d95926; --mag:#3987e5; --tip-bg:#fcfcfb; --tip-fg:#0b0b0b;
  }
  *{box-sizing:border-box}
  body{margin:0; background:var(--plane); color:var(--ink);
    font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;}
  .wrap{max-width:1040px; margin:0 auto; padding:28px 16px 48px;}
  header h1{font-size:clamp(1.5rem,4vw,2rem); margin:0 0 6px; letter-spacing:-.02em;}
  header p{margin:0; color:var(--ink-2); max-width:70ch;}
  .meta{margin-top:10px; display:flex; flex-wrap:wrap; gap:6px 8px; font-size:.78rem; color:var(--muted);}
  .meta span{border:1px solid var(--ring); border-radius:999px; padding:3px 10px; white-space:nowrap;}
  h2{font-size:.82rem; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-2); margin:0 0 2px; font-weight:650;}
  .sub{font-size:.78rem; color:var(--muted); margin:0 0 14px;}
  .kpis{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:26px 0 34px;}
  .kpi{background:var(--surface); border:1px solid var(--ring); border-radius:var(--radius); padding:16px 16px 14px;}
  .kpi .v{font-size:clamp(1.5rem,3.4vw,2.1rem); font-weight:700; letter-spacing:-.02em; line-height:1.05;}
  .kpi .v small{font-size:.5em; font-weight:650; color:var(--ink-2); margin-left:2px;}
  .kpi .k{font-size:.8rem; color:var(--ink-2); margin-top:6px;}
  .kpi .note{font-size:.72rem; color:var(--muted); margin-top:3px;}
  .grid2{display:grid; grid-template-columns:1fr 1fr; gap:20px;}
  .card{background:var(--surface); border:1px solid var(--ring); border-radius:var(--radius); padding:18px 18px 14px; min-width:0;}
  .legend{display:flex; gap:16px; flex-wrap:wrap; font-size:.8rem; color:var(--ink-2); margin:2px 0 8px;}
  .legend b{display:inline-flex; align-items:center; gap:6px; font-weight:500;}
  .sw{width:11px; height:11px; border-radius:3px; display:inline-block;}
  svg{display:block; width:100%; height:auto; overflow:visible;}
  .ax{fill:var(--muted); font-size:11px;}
  .axv{fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums;}
  .gl{stroke:var(--grid); stroke-width:1;} .base{stroke:var(--axis); stroke-width:1;}
  .val{fill:var(--ink); font-size:11.5px; font-weight:650; font-variant-numeric:tabular-nums;}
  .foot{margin-top:20px; font-size:.76rem; color:var(--muted);}
  .foot code{font-size:.95em;}
  #tip{position:fixed; z-index:10; pointer-events:none; opacity:0; transition:opacity .08s;
    background:var(--tip-bg); color:var(--tip-fg); font-size:.76rem; line-height:1.35;
    padding:7px 9px; border-radius:8px; max-width:240px; box-shadow:0 4px 16px rgba(0,0,0,.22);}
  @media (max-width:720px){ .kpis{grid-template-columns:repeat(2,1fr);} .grid2{grid-template-columns:1fr;} }
  @media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style></head>
<body>
<div class="wrap">
  <header>
    <h1>Voiceprint Benchmark</h1>
    <p>Inference speed and accuracy of the voice-print pipeline (Silero VAD v5 → fbank → ECAPA-TDNN) on
      onnxruntime CPU, measured by <code>test_result/benchmark.py</code> with the logic of
      <code>voiceprint_on_clip.py</code> (per-clip identification) and <code>voiceprint_diarize_all.py</code>
      (whole-file diarization of all.mp3).</p>
    <div class="meta" id="meta"></div>
  </header>
  <section class="kpis" id="kpis"></section>
  <div class="grid2">
    <section class="card">
      <h2>Accuracy · speaker identification</h2>
      <p class="sub" id="acc-sub"></p>
      <div class="legend">
        <b><i class="sw" style="background:var(--off)"></i>VAD off</b>
        <b><i class="sw" style="background:var(--on)"></i>VAD on</b>
      </div>
      <svg id="acc" viewBox="0 0 460 300" role="img" aria-label="Speaker identification accuracy"></svg>
    </section>
    <section class="card">
      <h2>Speed · latency per call</h2>
      <p class="sub" id="lat-sub"></p>
      <svg id="lat" viewBox="0 0 460 300" role="img" aria-label="Latency per call"></svg>
    </section>
  </div>
  <p class="foot" id="foot"></p>
</div>
<div id="tip" role="status" aria-live="polite"></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
function el(n, a) { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; }
function txt(parent, a, s) { const t = el("text", a); t.textContent = s; parent.appendChild(t); return t; }
const tip = $("#tip");
function hook(node, html) {
  const move = ev => { const p = 12; let x = ev.clientX + p, y = ev.clientY + p; const r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - p; if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - p;
    tip.style.left = x + "px"; tip.style.top = y + "px"; };
  node.addEventListener("mouseenter", ev => { tip.innerHTML = html; tip.style.opacity = 1; move(ev); });
  node.addEventListener("mousemove", move);
  node.addEventListener("mouseleave", () => tip.style.opacity = 0);
}
const E = D.env, cpuShort = E.cpu.replace(/^Apple /, "");
const nEnroll = Object.values(D.enroll)[0].length;

// ---------- header pills ----------
[`${E.cpu} · CPUExecutionProvider`, `onnxruntime ${E.onnxruntime}`, `${E.threads} thread${E.threads > 1 ? "s" : ""}`,
 "16 kHz mono", `${E.samples}: ${D.speakers.join(" / ")}`, `enroll: first ${nEnroll} clips per speaker`,
 `${D.speech_clips} speech clips identified`].forEach(s => { const sp = document.createElement("span"); sp.textContent = s; $("#meta").appendChild(sp); });

// ---------- KPIs ----------
const best = Object.entries(D.accuracy).reduce((a, b) => b[1].pct > a[1].pct ? b : a);
const tied = Object.entries(D.accuracy).filter(([, a]) => a.ok === best[1].ok).map(([k]) => k.split("|"));
const bestWhere = tied.length === Object.keys(D.accuracy).length ? "all 4 settings"
  : tied.map(([m, v]) => `${m} + VAD ${v}`).join(", ");
const dz = D.diarization.fp32, ef = D.latency_ms.ecapa_fp32;
[[`${dz.x_realtime.toFixed(0)}<small>× real time</small>`, "Diarization throughput (fp32)", `${E.audio_s.toFixed(1)} s audio → ${dz.wall_s.toFixed(2)} s`],
 [`${best[1].pct.toFixed(0)}<small>%</small>`, "Best speaker accuracy (per clip)", `${bestWhere} · ${best[1].ok}/${best[1].n}`],
 [`~${ef.mean.toFixed(0)}<small>ms</small>`, "ECAPA per embedding (fp32)", `median ${ef.median.toFixed(0)} · p90 ${ef.p90.toFixed(0)} ms`],
 [`${D.vad.x_realtime.toFixed(0)}<small>×</small>`, "Silero VAD v5 speed", "relative to real-time audio"],
].forEach(([v, k, note]) => {
  const d = document.createElement("div"); d.className = "kpi";
  d.innerHTML = `<div class="v">${v}</div><div class="k">${k}</div><div class="note">${note}</div>`;
  $("#kpis").appendChild(d);
});

// ---------- accuracy grouped bars ----------
const G = D.diarization_gt;
$("#acc-sub").textContent = `per clip: voiceprint_on_clip.py, share of ${D.speech_clips} clips · ` +
  `diarization: voiceprint_diarize_all.py, share of ${G.scored_s.toFixed(1)} s speech frames in all.mp3`;
(function () {
  const W = 460, H = 300, L = 40, R = 8, T = 16, B = 58, pw = W - L - R, ph = H - T - B, svg = $("#acc");
  const y = v => T + ph - (v / 100) * ph;
  for (let t = 0; t <= 100; t += 25) {
    svg.appendChild(el("line", {x1: L, y1: y(t), x2: L + pw, y2: y(t), class: "gl"}));
    txt(svg, {x: L - 6, y: y(t) + 4, class: "axv", "text-anchor": "end"}, t + "%");
  }
  svg.appendChild(el("line", {x1: L, y1: y(0), x2: L + pw, y2: y(0), class: "base"}));
  const clipBar = (m, v) => { const a = D.accuracy[`${m}|${v}`];
    return {v: a.pct, c: v === "on" ? "var(--on)" : "var(--off)",
            tip: `<b>ECAPA ${m} · per clip</b><br>VAD ${v}: <b>${a.pct.toFixed(1)}%</b> (${a.ok}/${a.n})`}; };
  const diaBar = m => { const a = D.diarization[m].accuracy;
    const per = Object.entries(a.per_speaker).map(([s, x]) => `${s}: ${(100 * x.accuracy).toFixed(0)}% of ${x.seconds.toFixed(1)} s`).join("<br>");
    return {v: 100 * a.accuracy, c: "var(--on)",
            tip: `<b>ECAPA ${m} · diarization</b><br>VAD on: <b>${(100 * a.accuracy).toFixed(1)}%</b> of ${a.scored_s.toFixed(1)} s<br>${per}`}; };
  const groups = [["fp32", "per clip", [clipBar("fp32", "off"), clipBar("fp32", "on")]],
                  ["int8", "per clip", [clipBar("int8", "off"), clipBar("int8", "on")]],
                  ["fp32", "diarization", [diaBar("fp32")]],
                  ["int8", "diarization", [diaBar("int8")]]];
  const gw = pw / groups.length, bw = 30, gap = 6;
  svg.appendChild(el("line", {x1: L + 2 * gw, y1: T, x2: L + 2 * gw, y2: y(0), class: "gl", "stroke-dasharray": "3 3"}));
  groups.forEach(([m, kind, bars], i) => {
    const cx = L + gw * i + gw / 2, tw = bars.length * bw + (bars.length - 1) * gap;
    bars.forEach((b, j) => {
      const x = cx - tw / 2 + j * (bw + gap);
      const rect = el("rect", {x, y: y(b.v), width: bw, height: (b.v / 100) * ph, rx: 4, fill: b.c});
      svg.appendChild(rect);
      txt(svg, {x: x + bw / 2, y: y(b.v) - 7, class: "val", "text-anchor": "middle"}, b.v.toFixed(0) + "%");
      hook(rect, b.tip);
    });
    txt(svg, {x: cx, y: H - 40, class: "ax", "text-anchor": "middle"}, `ECAPA ${m}`);
    txt(svg, {x: cx, y: H - 27, class: "ax", "text-anchor": "middle"}, kind);
  });
  txt(svg, {x: L, y: H - 8, class: "ax"}, `higher is better · diarization truth: ${G.located}/${G.clips} clips located`);
})();

// ---------- latency bars ----------
$("#lat-sub").textContent = `${cpuShort} CPU · mean per call (ms), lower is faster`;
(function () {
  const W = 460, H = 300, L = 44, R = 16, T = 16, B = 52, pw = W - L - R, ph = H - T - B, svg = $("#lat");
  const data = [["fbank", D.latency_ms.fbank, "feature front-end"],
                ["ECAPA int8", D.latency_ms.ecapa_int8, "quantized embedding"],
                ["ECAPA fp32", D.latency_ms.ecapa_fp32, "fp32 embedding"]];
  const max = Math.max(10, Math.ceil(Math.max(...data.map(d => d[1].mean)) / 10) * 10);
  const y = v => T + ph - (v / max) * ph;
  const step = max <= 50 ? 10 : max <= 100 ? 20 : 50;
  for (let t = 0; t <= max; t += step) {
    svg.appendChild(el("line", {x1: L, y1: y(t), x2: L + pw, y2: y(t), class: "gl"}));
    txt(svg, {x: L - 6, y: y(t) + 4, class: "axv", "text-anchor": "end"}, t);
  }
  svg.appendChild(el("line", {x1: L, y1: y(0), x2: L + pw, y2: y(0), class: "base"}));
  const bw = 56, sx = pw / data.length;
  data.forEach(([name, s, note], i) => {
    const cx = L + sx * i + sx / 2, h = (s.mean / max) * ph;
    const rect = el("rect", {x: cx - bw / 2, y: y(s.mean), width: bw, height: h, rx: 4, fill: "var(--mag)"});
    svg.appendChild(rect);
    txt(svg, {x: cx, y: y(s.mean) - 7, class: "val", "text-anchor": "middle"}, s.mean.toFixed(1));
    txt(svg, {x: cx, y: H - 32, class: "ax", "text-anchor": "middle"}, name);
    hook(rect, `<b>${name}</b> (${note})<br>mean <b>${s.mean.toFixed(1)} ms</b> · median ${s.median.toFixed(1)} · p90 ${s.p90.toFixed(1)}<br>${s.calls} calls`);
  });
  txt(svg, {x: L, y: H - 12, class: "ax"}, `lower is faster · ms per call (${cpuShort} CPU)`);
  txt(svg, {x: L - 30, y: T + ph / 2, class: "ax", "text-anchor": "middle", transform: `rotate(-90 ${L - 30} ${T + ph / 2})`}, "ms");
})();

$("#foot").innerHTML = `Generated ${E.date} with <code>${E.command}</code> (Python ${E.python}). ` +
  `Diarization: ${D.diarization.fp32.ecapa_calls} ECAPA calls over ${E.audio_s.toFixed(1)} s (pieces ≤ ${E.max_piece_s} s); ` +
  `accuracy is scored on the ${G.scored_s.toFixed(1)} s of speech inside the ${G.located} clips located in all.mp3 ` +
  `(enrollment clips excluded), so it is indicative only; ` +
  `int8 ${D.diarization.int8.wall_s.toFixed(2)} s = ${D.diarization.int8.x_realtime.toFixed(0)}× real time.`;
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="samples_inmu3", help="clip folder with all.mp3 (relative to the repo root)")
    ap.add_argument("--enroll", type=int, default=3, help="first N clips of each speaker are enrolled")
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime intra-op threads")
    ap.add_argument("--out", default=str(HERE / "Voiceprint Benchmark.html"), help="HTML report path")
    ap.add_argument("--json", default=str(HERE / "benchmark_result.json"), help="raw results path")
    args = ap.parse_args()

    result = run(args)
    print_summary(result)
    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(result, Path(args.out))
    print(f"\nwrote {args.out}\n      {args.json}")


if __name__ == "__main__":
    main()

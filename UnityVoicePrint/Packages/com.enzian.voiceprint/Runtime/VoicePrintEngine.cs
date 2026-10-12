using System;
using System.Collections.Generic;
using UnityEngine;

namespace VoicePrint
{
    [Serializable]
    public class VoicePrintSettings
    {
        [Tooltip("Run Silero VAD before ECAPA: reject non-speech clips and trim silence.")]
        public bool useVad = true;

        [Tooltip("Clip is non-speech if its max VAD frame probability is below this.")]
        [Range(0f, 1f)] public float vadRejectProb = 0.3f;

        [Tooltip("Only trim to the VAD speech segments if they total at least this many seconds; " +
                 "otherwise embed the full clip (over-trimming short clips hurts ECAPA).")]
        public float minTrimmedSpeechSeconds = 1.5f;

        [Tooltip("Cosine similarity needed to accept the best enrolled speaker. Mainly rejects speakers who " +
                 "are not enrolled; raise it if strangers get accepted.")]
        [Range(-1f, 1f)] public float acceptThreshold = 0.2f;

        [Tooltip("Best score must beat the runner-up by at least this much, else Unknown. With 2 s pieces, " +
                 "0.05 halves wrong-speaker labels on conversational audio (14% -> 7%).")]
        [Range(0f, 1f)] public float minMargin = 0.05f;

        [Tooltip("IdentifySegment: split each segment into VAD speech regions, then equal pieces of at most " +
                 "this many seconds, and identify every piece (a segment can hold several speakers). " +
                 "0 = identify the whole segment at once.")]
        public float maxPieceSeconds = 2f;

        [Tooltip("IdentifySegment: blend each piece's scores with its neighbours in the same segment, weighted " +
                 "by length (0 = off, 1 = a neighbour counts as much as the piece). Off by default: in fast " +
                 "turn-taking neighbours are often another speaker, and blending raised wrong labels (14% -> 23%).")]
        [Range(0f, 1f)] public float neighbourSmoothing = 0f;
    }

    /// <summary>One piece of a segment identified by VoicePrintEngine.IdentifySegment.</summary>
    public struct SpeechPiece
    {
        public int Start, End;          // sample range inside the segment
        public float[] Audio;           // the piece itself (VAD speech, mono 16 kHz)
        public Dictionary<string, float> RawScores;   // cosine per speaker before smoothing (null if too short)
        public IdentifyResult Result;   // decision on the smoothed scores; IsSpeech false if too short to embed
    }

    /// <summary>Result of turning one clip into an embedding.</summary>
    public struct EmbedResult
    {
        public bool IsSpeech;           // false -> VAD rejected the clip (Embedding is null)
        public float MaxVadProb;        // 1 when VAD is off
        public int UsedSamples;         // samples actually fed to ECAPA
        public float[] Embedding;       // unit length, 192-d
        public float[] SpeechAudio;     // VAD speech segments concatenated (clean voice, mono 16 kHz);
                                        // the input itself when VAD is off or found no segment
    }

    public struct IdentifyResult
    {
        public bool IsSpeech;
        public string Speaker;          // best enrolled speaker, null if library empty / non-speech
        public float Score;             // cosine to Speaker
        public string RunnerUp;
        public float RunnerUpScore;
        public bool Accepted;           // Score >= acceptThreshold && margin >= minMargin
        public Dictionary<string, float> Scores;
        public EmbedResult Embed;

        public float Margin => RunnerUp == null ? Score : Score - RunnerUpScore;

        public override string ToString() => !IsSpeech
            ? "non-speech"
            : Speaker == null
                ? "no enrolled speakers"
                : $"{(Accepted ? Speaker : "unknown")} (best {Speaker} {Score:F3}" +
                  (RunnerUp != null ? $", next {RunnerUp} {RunnerUpScore:F3})" : ")");
    }

    /// <summary>
    /// The voice-print pipeline, Unity-agnostic and thread-safe:
    ///   [Silero VAD gate/trim] -> LogMelFbank -> ECAPA -> cosine vs enrolled speakers.
    /// Mirrors test_result/voiceprint_on_clip.py. All audio is mono 16 kHz float[]
    /// (see AudioUtil.ToMono16k for AudioClips / other rates).
    /// Inference is CPU heavy (~100 ms per second of audio) -- call from a worker thread.
    /// </summary>
    public sealed class VoicePrintEngine : IDisposable
    {
        public VoicePrintSettings Settings { get; set; }
        public SpeakerLibrary Library { get; } = new SpeakerLibrary();

        readonly SpeakerEncoder encoder;
        readonly SileroVadScorer vad;

        /// <param name="ecapaModel">ecapa_tdnn_int8.bytes</param>
        /// <param name="vadModel">silero_vad.bytes (null disables VAD regardless of settings)</param>
        /// <param name="embMean">optional emb_mean (192 floats) for embedding centering</param>
        public VoicePrintEngine(byte[] ecapaModel, byte[] vadModel, VoicePrintSettings settings = null,
                                float[] embMean = null, int intraOpThreads = 2)
        {
            Settings = settings ?? new VoicePrintSettings();
            encoder = new SpeakerEncoder(ecapaModel, embMean, intraOpThreads);
            if (vadModel != null && vadModel.Length > 0) vad = new SileroVadScorer(vadModel);
        }

        public bool HasVad => vad != null;

        /// <summary>VAD gate (optional) + ECAPA embedding of a mono 16 kHz clip.</summary>
        public EmbedResult Embed(float[] wav16k)
        {
            var r = new EmbedResult { IsSpeech = true, MaxVadProb = 1f, SpeechAudio = wav16k };
            float[] speech = wav16k;

            if (Settings.useVad && vad != null)
            {
                var probs = vad.Probs(wav16k);
                float max = 0f;
                foreach (var p in probs) max = Math.Max(max, p);
                r.MaxVadProb = max;
                if (max < Settings.vadRejectProb)
                {
                    r.IsSpeech = false;
                    return r;
                }
                var trimmed = Concat(wav16k, SileroVadScorer.GetSpeechTimestamps(probs, wav16k.Length));
                if (trimmed.Length > 0) r.SpeechAudio = trimmed;
                // ECAPA only gets the trimmed audio if enough is left (over-trimming short clips hurts it)
                if (trimmed.Length >= Settings.minTrimmedSpeechSeconds * LogMelFbank.SampleRate)
                    speech = trimmed;
            }

            if (speech.Length < SpeakerEncoder.MinSamples)
            {
                r.IsSpeech = false;
                return r;
            }
            r.UsedSamples = speech.Length;
            r.Embedding = encoder.Embed(speech);
            return r;
        }

        /// <summary>
        /// Add a clip to a speaker's voice print (several clips are averaged).
        /// Returns the embed result; IsSpeech == false means nothing was enrolled.
        /// Enroll with >= 3 s of clean single-speaker audio for a reliable print.
        /// </summary>
        public EmbedResult Enroll(string speaker, float[] wav16k)
        {
            var r = Embed(wav16k);
            if (r.IsSpeech) Library.Add(speaker, r.Embedding);
            return r;
        }

        public IdentifyResult Identify(float[] wav16k) => Score(Embed(wav16k));

        /// <summary>Score an existing embedding against the library.</summary>
        public IdentifyResult Score(EmbedResult e)
        {
            if (!e.IsSpeech)
                return new IdentifyResult { IsSpeech = false, Embed = e, Scores = new Dictionary<string, float>() };
            var scores = new Dictionary<string, float>();
            foreach (var (name, print) in Library.Snapshot()) scores[name] = VectorMath.Dot(e.Embedding, print);
            return Decide(scores, e);
        }

        /// <summary>Best / runner-up / accept decision on a score per speaker.</summary>
        IdentifyResult Decide(Dictionary<string, float> scores, EmbedResult e)
        {
            var r = new IdentifyResult { IsSpeech = true, Embed = e, Scores = scores };
            float best = float.NegativeInfinity, second = float.NegativeInfinity;
            foreach (var kv in scores)
            {
                float s = kv.Value;
                if (s > best)
                {
                    second = best; r.RunnerUp = r.Speaker; r.RunnerUpScore = r.Score;
                    best = s; r.Speaker = kv.Key; r.Score = s;
                }
                else if (s > second)
                {
                    second = s; r.RunnerUp = kv.Key; r.RunnerUpScore = s;
                }
            }
            r.Accepted = r.Speaker != null && r.Score >= Settings.acceptThreshold && r.Margin >= Settings.minMargin;
            return r;
        }

        /// <summary>
        /// Identify a segment that may contain several speakers (e.g. one AudioRecorder segment):
        /// VAD speech regions -> equal pieces of at most maxPieceSeconds -> ECAPA per piece ->
        /// scores blended with the neighbouring pieces (neighbourSmoothing) -> decision per piece.
        /// Same splitting as test_result/voiceprint_diarize_all.py. Returns pieces in time order;
        /// an empty list means the segment is non-speech.
        /// </summary>
        /// <param name="vadProbs">optional per-512-sample speech probabilities for wav16k from a streaming
        /// VAD (UnityVAD.VADSegment.ChunkProbs). Prefer them: a fresh VAD run on the segment starts with a
        /// cold state and misses its first words (~12% of speech on samples_inmu3).</param>
        public List<SpeechPiece> IdentifySegment(float[] wav16k, float[] vadProbs = null)
        {
            var pieces = new List<SpeechPiece>();
            List<(int start, int end)> regions;
            float maxProb = 1f;
            bool haveProbs = vadProbs != null &&
                             vadProbs.Length == (wav16k.Length + SileroVadScorer.Win - 1) / SileroVadScorer.Win;
            if (Settings.useVad && (vad != null || haveProbs))
            {
                var probs = haveProbs ? vadProbs : vad.Probs(wav16k);
                maxProb = 0f;
                foreach (var p in probs) maxProb = Math.Max(maxProb, p);
                if (maxProb < Settings.vadRejectProb) return pieces;
                regions = SileroVadScorer.GetSpeechTimestamps(probs, wav16k.Length);
                if (regions.Count == 0) regions.Add((0, wav16k.Length));
            }
            else regions = new List<(int, int)> { (0, wav16k.Length) };

            int maxPiece = (int)(Settings.maxPieceSeconds * LogMelFbank.SampleRate);
            var prints = Library.Snapshot();
            foreach (var (rs, re) in regions)
            {
                int n = maxPiece > 0 ? Math.Max(1, (int)Math.Ceiling((re - rs) / (double)maxPiece)) : 1;
                for (int i = 0; i < n; i++)
                {
                    int ps = rs + (int)((long)(re - rs) * i / n), pe = rs + (int)((long)(re - rs) * (i + 1) / n);
                    var audio = new float[pe - ps];
                    Array.Copy(wav16k, ps, audio, 0, audio.Length);
                    var piece = new SpeechPiece { Start = ps, End = pe, Audio = audio };
                    if (audio.Length >= SpeakerEncoder.MinSamples)
                    {
                        var emb = encoder.Embed(audio);
                        piece.RawScores = new Dictionary<string, float>();
                        foreach (var (name, print) in prints) piece.RawScores[name] = VectorMath.Dot(emb, print);
                        piece.Result.Embed = new EmbedResult
                        {
                            IsSpeech = true, MaxVadProb = maxProb, UsedSamples = audio.Length,
                            Embedding = emb, SpeechAudio = audio,
                        };
                    }
                    pieces.Add(piece);
                }
            }

            // neighbour smoothing over the embeddable pieces, weighted by piece length
            var idx = new List<int>();
            for (int i = 0; i < pieces.Count; i++) if (pieces[i].RawScores != null) idx.Add(i);
            float a = Settings.neighbourSmoothing;
            for (int k = 0; k < idx.Count; k++)
            {
                var p = pieces[idx[k]];
                var smoothed = new Dictionary<string, float>();
                foreach (var name in p.RawScores.Keys)
                {
                    double num = p.RawScores[name] * p.Audio.Length, den = p.Audio.Length;
                    foreach (int nb in new[] { k - 1, k + 1 })
                    {
                        if (nb < 0 || nb >= idx.Count || a <= 0f) continue;
                        var q = pieces[idx[nb]];
                        num += a * q.RawScores[name] * q.Audio.Length;
                        den += a * q.Audio.Length;
                    }
                    smoothed[name] = (float)(num / den);
                }
                p.Result = Decide(smoothed, p.Result.Embed);
                pieces[idx[k]] = p;
            }
            return pieces;
        }

        static float[] Concat(float[] wav, List<(int start, int end)> segs)
        {
            int n = 0;
            foreach (var (s, e) in segs) n += e - s;
            var outBuf = new float[n];
            int o = 0;
            foreach (var (s, e) in segs)
            {
                Array.Copy(wav, s, outBuf, o, e - s);
                o += e - s;
            }
            return outBuf;
        }

        public void Dispose()
        {
            encoder.Dispose();
            vad?.Dispose();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VoicePrint
{
    /// <summary>
    /// Offline Silero VAD v5 over a whole clip (port of pretrained_model_convert/silero_vad.py).
    /// Used to gate/trim a clip before ECAPA. Unlike UnityVAD.VADInference (streaming, event
    /// based), this returns the per-frame probabilities. Each call keeps its own state, so
    /// Probs() is safe to call from several threads on one instance.
    ///
    /// v5 calling convention: input = context(64) + chunk(512) = 576 samples, state (2,1,128)
    /// fed back each step, sr int64. Feeding a bare 512-sample chunk silently returns ~0.
    /// </summary>
    public sealed class SileroVadScorer : IDisposable
    {
        public const int Win = 512;
        public const int Context = 64;
        const int StateSize = 2 * 1 * 128;

        readonly InferenceSession session;
        static readonly int[] InputDims = { 1, Context + Win };
        static readonly int[] StateDims = { 2, 1, 128 };
        static readonly int[] SrDims = { 1 };

        public SileroVadScorer(byte[] model)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                InterOpNumThreads = 1,
                IntraOpNumThreads = 1,
            };
            session = new InferenceSession(model, options);
        }

        /// <summary>mono 16 kHz waveform -> speech probability per 512-sample frame.</summary>
        public float[] Probs(float[] wav)
        {
            int n = (wav.Length + Win - 1) / Win;
            var probs = new float[n];
            var state = new float[StateSize];
            var buf = new float[Context + Win];
            var sr = new DenseTensor<long>(new long[] { LogMelFbank.SampleRate }, SrDims);

            for (int i = 0; i < n; i++)
            {
                // buf[0..64) already holds the previous chunk's tail (zeros on the first step)
                int len = Math.Min(Win, wav.Length - i * Win);
                Array.Copy(wav, i * Win, buf, Context, len);
                if (len < Win) Array.Clear(buf, Context + len, Win - len);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(buf, InputDims)),
                    NamedOnnxValue.CreateFromTensor("state", new DenseTensor<float>(state, StateDims)),
                    NamedOnnxValue.CreateFromTensor("sr", sr),
                };
                using (var results = session.Run(inputs))
                {
                    probs[i] = results.First(r => r.Name == "output").AsEnumerable<float>().First();
                    state = results.First(r => r.Name == "stateN").AsEnumerable<float>().ToArray();
                }
                Array.Copy(buf, Win, buf, 0, Context);
            }
            return probs;
        }

        /// <summary>
        /// Frame probabilities -> [start, end) sample ranges with hysteresis, min speech/silence,
        /// padding and overlap merge. Same defaults as silero_vad.get_speech_timestamps.
        /// </summary>
        public static List<(int start, int end)> GetSpeechTimestamps(
            float[] probs, int totalSamples, float threshold = 0.5f, int minSpeechMs = 250,
            int minSilenceMs = 100, int speechPadMs = 30)
        {
            const int sr = LogMelFbank.SampleRate;
            float neg = threshold - 0.15f;
            double minSpeech = sr * minSpeechMs / 1000.0;
            double minSilence = sr * minSilenceMs / 1000.0;
            int pad = sr * speechPadMs / 1000;

            var segs = new List<(int, int)>();
            bool triggered = false;
            int start = 0, tempEnd = 0;
            for (int i = 0; i < probs.Length; i++)
            {
                int cur = i * Win;
                float p = probs[i];
                if (p >= threshold)
                {
                    tempEnd = 0;
                    if (!triggered) { triggered = true; start = cur; }
                }
                else if (p < neg && triggered)
                {
                    if (tempEnd == 0) tempEnd = cur;
                    if (cur - tempEnd >= minSilence)
                    {
                        if (tempEnd - start >= minSpeech) segs.Add((start, tempEnd));
                        triggered = false;
                        tempEnd = 0;
                    }
                }
            }
            if (triggered && totalSamples - start >= minSpeech) segs.Add((start, totalSamples));

            var merged = new List<(int start, int end)>();
            foreach (var (s0, e0) in segs)
            {
                int s = Math.Max(0, s0 - pad), e = Math.Min(totalSamples, e0 + pad);
                if (merged.Count > 0 && s <= merged[merged.Count - 1].end)
                    merged[merged.Count - 1] = (merged[merged.Count - 1].start, e);
                else
                    merged.Add((s, e));
            }
            return merged;
        }

        public void Dispose() => session.Dispose();
    }
}

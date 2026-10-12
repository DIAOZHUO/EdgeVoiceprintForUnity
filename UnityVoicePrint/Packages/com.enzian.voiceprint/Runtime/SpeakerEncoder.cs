using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VoicePrint
{
    /// <summary>
    /// ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) on ONNX Runtime.
    ///   wav -> LogMelFbank -> "waveform_feature" (1, frames, 80) -> "embedding" (1, 1, 192)
    /// The graph already contains SpeechBrain's per-utterance input mean/var norm and returns
    /// the RAW embedding (what verify_batch scores). Embed() L2-normalizes it, optionally after
    /// subtracting emb_mean (SpeechBrain's mean_var_norm_emb); keep that choice identical for
    /// enrollment and verification. Run() is thread-safe in ONNX Runtime.
    /// </summary>
    public sealed class SpeakerEncoder : IDisposable
    {
        public const int EmbeddingDim = 192;

        /// Below ~0.25 s the TDNN receptive field is mostly padding; the embedding is noise.
        public const int MinSamples = LogMelFbank.SampleRate / 4;

        readonly InferenceSession session;
        readonly string inputName;
        readonly float[] embMean;

        /// <param name="model">single-file ONNX bytes (ecapa_tdnn_int8.bytes)</param>
        /// <param name="embMean">optional 192-d centering vector; null = no centering</param>
        public SpeakerEncoder(byte[] model, float[] embMean = null, int intraOpThreads = 2)
        {
            if (embMean != null && embMean.Length != EmbeddingDim)
                throw new ArgumentException($"emb_mean must have {EmbeddingDim} values, got {embMean.Length}");
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                InterOpNumThreads = 1,
                IntraOpNumThreads = intraOpThreads,
            };
            session = new InferenceSession(model, options);
            inputName = session.InputMetadata.Keys.First();
            this.embMean = embMean;
        }

        /// <summary>log-mel fbank (frames, 80) row-major -> raw 192-d embedding.</summary>
        public float[] EmbedFeatures(float[] fbank)
        {
            int frames = fbank.Length / LogMelFbank.NMels;
            var input = new DenseTensor<float>(fbank, new[] { 1, frames, LogMelFbank.NMels });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, input) };
            using (var results = session.Run(inputs))
                return results.First().AsEnumerable<float>().ToArray();
        }

        /// <summary>16 kHz mono wav -> unit-length 192-d embedding (cosine = dot product).</summary>
        public float[] Embed(float[] wav16k)
        {
            if (wav16k == null || wav16k.Length < MinSamples)
                throw new ArgumentException($"need at least {MinSamples} samples (0.25 s @16 kHz)");
            var emb = EmbedFeatures(LogMelFbank.Compute(wav16k));
            if (embMean != null)
                for (int i = 0; i < emb.Length; i++) emb[i] -= embMean[i];
            return VectorMath.Normalize(emb);
        }

        public void Dispose() => session.Dispose();
    }

    public static class VectorMath
    {
        public static float Dot(float[] a, float[] b)
        {
            float s = 0f;
            for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
            return s;
        }

        /// <summary>In-place L2 normalization; returns the same array.</summary>
        public static float[] Normalize(float[] v)
        {
            float inv = 1f / ((float)Math.Sqrt(Dot(v, v)) + 1e-9f);
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
            return v;
        }

        public static float[] FromBytes(byte[] bytes)
        {
            var f = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
            return f;
        }
    }
}

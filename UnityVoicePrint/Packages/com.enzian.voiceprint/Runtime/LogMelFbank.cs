using System;
using System.Threading.Tasks;

namespace VoicePrint
{
    /// <summary>
    /// SpeechBrain Fbank front-end for spkrec-ecapa-voxceleb, ported from
    /// pretrained_model_convert/fbank_cs.py (numerically matches it to float precision).
    ///
    ///   16 kHz mono wav -> center-pad 200 -> frames (400 / hop 160)
    ///   -> power spectrum via window-folded DFT matrices (no FFT library)
    ///   -> 80-band mel filterbank -> 10*log10 with an 80 dB floor below the global max.
    ///
    /// Output layout is row-major (numFrames, 80), i.e. exactly the ECAPA input
    /// "waveform_feature" (1, numFrames, 80) without the batch dim.
    /// The constant matrices are built once; Compute() is thread-safe.
    /// </summary>
    public static class LogMelFbank
    {
        public const int SampleRate = 16000;
        public const int NFft = 400;            // 25 ms
        public const int Hop = 160;             // 10 ms
        public const int NFreq = NFft / 2 + 1;  // 201
        public const int NMels = 80;
        const int Pad = NFft / 2;
        const float AMin = 1e-10f;
        const float TopDb = 80f;

        // (NFreq, NFft) row-major: each frequency row is contiguous so the inner loop
        // is a straight dot product with the frame. Hamming window is folded in.
        static readonly float[] Wc;
        static readonly float[] Ws;

        // Triangular mel filters stored sparsely: band m covers freq bins
        // [melStart[m], melStart[m] + melWeights[m].Length).
        static readonly int[] melStart = new int[NMels];
        static readonly float[][] melWeights = new float[NMels][];

        static LogMelFbank()
        {
            Wc = new float[NFreq * NFft];
            Ws = new float[NFreq * NFft];
            for (int k = 0; k < NFreq; k++)
            {
                for (int n = 0; n < NFft; n++)
                {
                    double w = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * n / NFft); // periodic Hamming
                    double ang = 2.0 * Math.PI * n * k / NFft;
                    Wc[k * NFft + n] = (float)(w * Math.Cos(ang));
                    Ws[k * NFft + n] = (float)(w * Math.Sin(ang));
                }
            }
            BuildMel();
        }

        static double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
        static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

        static void BuildMel()
        {
            double fMax = SampleRate / 2.0;
            double mMin = HzToMel(0.0), mMax = HzToMel(fMax);
            var hz = new double[NMels + 2];
            for (int i = 0; i < hz.Length; i++)
                hz[i] = MelToHz(mMin + (mMax - mMin) * i / (NMels + 1));

            var col = new float[NFreq];
            for (int m = 0; m < NMels; m++)
            {
                double band = hz[m + 1] - hz[m];
                double center = hz[m + 1];
                int first = -1, last = -1;
                for (int f = 0; f < NFreq; f++)
                {
                    double freq = fMax * f / (NFreq - 1);
                    double slope = (freq - center) / band;
                    double v = Math.Max(0.0, Math.Min(slope + 1.0, -slope + 1.0));
                    col[f] = (float)v;
                    if (col[f] > 0f)
                    {
                        if (first < 0) first = f;
                        last = f;
                    }
                }
                if (first < 0) { first = 0; last = -1; }
                melStart[m] = first;
                melWeights[m] = new float[last - first + 1];
                Array.Copy(col, first, melWeights[m], 0, melWeights[m].Length);
            }
        }

        public static int NumFrames(int numSamples) => 1 + numSamples / Hop;

        /// <summary>16 kHz mono waveform -> log-mel fbank, row-major (NumFrames, 80).</summary>
        public static float[] Compute(float[] wav, int offset = 0, int count = -1)
        {
            if (count < 0) count = wav.Length - offset;
            int numFrames = NumFrames(count);
            var output = new float[numFrames * NMels];

            // Frame f covers padded[f*Hop .. f*Hop+NFft), padded = [0]*Pad + wav + [0]*Pad.
            void DoFrame(int f, float[] frame, float[] power)
            {
                int start = f * Hop - Pad;
                for (int n = 0; n < NFft; n++)
                {
                    int i = start + n;
                    frame[n] = (i >= 0 && i < count) ? wav[offset + i] : 0f;
                }
                for (int k = 0; k < NFreq; k++)
                {
                    int row = k * NFft;
                    float re = 0f, im = 0f;
                    for (int n = 0; n < NFft; n++)
                    {
                        float x = frame[n];
                        re += x * Wc[row + n];
                        im += x * Ws[row + n];
                    }
                    power[k] = re * re + im * im;
                }
                int o = f * NMels;
                for (int m = 0; m < NMels; m++)
                {
                    var wts = melWeights[m];
                    int s = melStart[m];
                    float acc = 0f;
                    for (int j = 0; j < wts.Length; j++) acc += power[s + j] * wts[j];
                    output[o + m] = 10f * (float)Math.Log10(Math.Max(acc, AMin));
                }
            }

            if (numFrames >= 64)
            {
                Parallel.For(0, numFrames,
                    () => (new float[NFft], new float[NFreq]),
                    (f, _, buf) => { DoFrame(f, buf.Item1, buf.Item2); return buf; },
                    _ => { });
            }
            else
            {
                var frame = new float[NFft];
                var power = new float[NFreq];
                for (int f = 0; f < numFrames; f++) DoFrame(f, frame, power);
            }

            // top_db: clamp everything to (global max - 80 dB)
            float max = float.NegativeInfinity;
            for (int i = 0; i < output.Length; i++) if (output[i] > max) max = output[i];
            float floor = max - TopDb;
            for (int i = 0; i < output.Length; i++) if (output[i] < floor) output[i] = floor;
            return output;
        }
    }
}

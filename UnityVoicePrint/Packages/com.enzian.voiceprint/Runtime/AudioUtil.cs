using System;
using System.Threading.Tasks;
using UnityEngine;

namespace VoicePrint
{
    public static class AudioUtil
    {
        /// <summary>
        /// AudioClip -> mono 16 kHz float[] (downmix + resample). Main thread only (GetData).
        /// The clip must be readable: Load Type "Decompress On Load", or a Microphone clip.
        /// </summary>
        public static float[] ToMono16k(AudioClip clip)
        {
            var data = new float[clip.samples * clip.channels];
            if (!clip.GetData(data, 0))
                throw new InvalidOperationException($"AudioClip '{clip.name}' is not readable (use Decompress On Load)");
            return ToMono16k(data, clip.channels, clip.frequency);
        }

        /// <summary>
        /// Async ToMono16k: call from the main thread. Loads the audio data first if needed
        /// (Preload Audio Data off / Load In Background), then downmixes and resamples on a
        /// worker thread.
        /// </summary>
        public static async Task<float[]> ToMono16kAsync(AudioClip clip)
        {
            if (clip.loadState == AudioDataLoadState.Unloaded) clip.LoadAudioData();
            while (clip.loadState == AudioDataLoadState.Loading) await Task.Yield();
            if (clip.loadState != AudioDataLoadState.Loaded)
                throw new InvalidOperationException($"AudioClip '{clip.name}' failed to load");

            var data = new float[clip.samples * clip.channels];
            if (!clip.GetData(data, 0))
                throw new InvalidOperationException($"AudioClip '{clip.name}' is not readable (use Decompress On Load)");
            int channels = clip.channels, rate = clip.frequency;
            return await Task.Run(() => ToMono16k(data, channels, rate));
        }

        public static float[] ToMono16k(float[] interleaved, int channels, int sampleRate)
        {
            var mono = interleaved;
            if (channels > 1)
            {
                mono = new float[interleaved.Length / channels];
                for (int i = 0; i < mono.Length; i++)
                {
                    float s = 0f;
                    for (int c = 0; c < channels; c++) s += interleaved[i * channels + c];
                    mono[i] = s / channels;
                }
            }
            return Resample(mono, sampleRate, LogMelFbank.SampleRate);
        }

        /// <summary>
        /// Band-limited resampling with a Hann-windowed sinc (cutoff at the lower Nyquist).
        /// Close to scipy.signal.resample_poly used on the Python side; not bit-exact.
        /// </summary>
        public static float[] Resample(float[] x, int srcRate, int dstRate, int halfTaps = 16)
        {
            if (srcRate == dstRate) return x;
            long outLen = (long)x.Length * dstRate / srcRate;
            var y = new float[outLen];
            double step = (double)srcRate / dstRate;
            double cutoff = Math.Min(1.0, (double)dstRate / srcRate);   // relative to src Nyquist
            double halfWidth = halfTaps / cutoff;                        // in source samples

            for (long i = 0; i < outLen; i++)
            {
                double t = i * step;
                int lo = (int)Math.Ceiling(t - halfWidth), hi = (int)Math.Floor(t + halfWidth);
                double acc = 0.0, wsum = 0.0;
                for (int j = Math.Max(lo, 0); j <= Math.Min(hi, x.Length - 1); j++)
                {
                    double d = j - t;
                    double a = Math.PI * d * cutoff;
                    double sinc = Math.Abs(a) < 1e-9 ? 1.0 : Math.Sin(a) / a;
                    double win = 0.5 + 0.5 * Math.Cos(Math.PI * d / halfWidth);
                    double w = sinc * win;
                    acc += x[j] * w;
                    wsum += w;
                }
                y[i] = (float)(wsum > 0 ? acc / wsum : 0.0);
            }
            return y;
        }
    }
}

"""
C#-equivalent fbank frontend (reference for the Unity/Quest port).

Uses ONLY operations that map 1:1 to C#/Burst with no FFT library (replaces the
old np.fft.rfft-based fbank.py; same output to ~1e-4 dB):

    1. frame the (zero-padded) signal           -> (num_frames, n_fft)
    2. power spectrum via precomputed DFT mats   -> real = F @ Wc,  imag = F @ Ws
                                                    power = real^2 + imag^2
       (the Hamming window is folded into Wc/Ws, so no separate windowing step)
    3. mel filterbank matmul                     -> power @ MEL
    4. log / amplitude-to-dB                     -> 10*log10(clamp) with top_db floor

Everything is a constant matrix built once + a couple of matmuls per utterance,
which is exactly what the C# MonoBehaviour will do (NativeArray + Burst, or a
tiny Sentis preprocessing model).

Numerically identical (to float precision) to SpeechBrain's Fbank for this model.
"""
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400          # 25 ms window @ 16 kHz
HOP = 160            # 10 ms hop
N_FREQ = N_FFT // 2 + 1   # 201
N_MELS = 80
PAD = N_FFT // 2     # center=True, constant padding


def _hamming_periodic(n: int) -> np.ndarray:
    k = np.arange(n)
    return 0.54 - 0.46 * np.cos(2 * np.pi * k / n)   # periodic (matches torch default)


def _hz_to_mel(hz): return 2595.0 * np.log10(1 + hz / 700.0)
def _mel_to_hz(mel): return 700.0 * (10 ** (mel / 2595.0) - 1)


def build_mel_matrix(n_fft=N_FFT, sr=SAMPLE_RATE, n_mels=N_MELS, f_min=0.0, f_max=None):
    """(N_FREQ, N_MELS) triangular mel filterbank, matching SpeechBrain."""
    if f_max is None:
        f_max = sr / 2
    mel_pts = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz = _mel_to_hz(mel_pts)
    band = hz[1:] - hz[:-1]
    f_central = hz[1:-1]
    freqs = np.linspace(0, sr / 2, N_FREQ)
    mel = np.zeros((N_FREQ, n_mels), dtype=np.float32)
    for m in range(n_mels):
        slope = (freqs - f_central[m]) / band[m]
        mel[:, m] = np.maximum(0.0, np.minimum(slope + 1.0, -slope + 1.0))
    return mel


def build_dft_matrices(n_fft=N_FFT, n_freq=N_FREQ):
    """Window-folded real/imag DFT matrices, shape (n_fft, n_freq) each.

    real[k] = sum_n (w[n]*x[n]) * cos(2*pi*k*n/N)
    imag[k] = -sum_n (w[n]*x[n]) * sin(2*pi*k*n/N)
    """
    w = _hamming_periodic(n_fft)                        # (n_fft,)
    n = np.arange(n_fft)[:, None]                       # (n_fft,1)
    k = np.arange(n_freq)[None, :]                      # (1,n_freq)
    ang = 2 * np.pi * n * k / n_fft                     # (n_fft,n_freq)
    Wc = (w[:, None] * np.cos(ang)).astype(np.float32)
    Ws = (w[:, None] * np.sin(ang)).astype(np.float32)  # note: imag = -(F @ Ws)
    return Wc, Ws


# ---- build the three constant matrices once (ship these to C#) ----
WC, WS = build_dft_matrices()
MEL = build_mel_matrix()


def frame_signal(x: np.ndarray) -> np.ndarray:
    """(T,) -> (num_frames, n_fft) with center padding, like torch.stft(center=True)."""
    x = np.pad(x, (PAD, PAD), mode="constant")
    num_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(num_frames)[:, None]
    return x[idx]                                        # (num_frames, n_fft)


def amplitude_to_db(power, amin=1e-10, top_db=80.0, multiplier=10.0):
    x_db = multiplier * np.log10(np.maximum(power, amin))
    x_db = np.maximum(x_db, x_db.max() - top_db)         # global max over (frames,mels)
    return x_db


def compute_fbank(wav: np.ndarray) -> np.ndarray:
    """mono 16 kHz waveform (T,) -> log-mel fbank (num_frames, N_MELS) float32."""
    wav = np.asarray(wav, dtype=np.float32)
    frames = frame_signal(wav)                           # (F, n_fft)
    real = frames @ WC                                   # (F, n_freq)
    imag = -(frames @ WS)
    power = real * real + imag * imag                    # (F, n_freq)
    mel = power @ MEL                                     # (F, n_mels)
    return amplitude_to_db(mel).astype(np.float32)


def wav_to_feature(wav: np.ndarray) -> np.ndarray:
    """-> (1, num_frames, N_MELS) ready for the ONNX model."""
    return compute_fbank(wav)[np.newaxis, :, :]


if __name__ == "__main__":
    # self-check vs SpeechBrain Fbank
    import torch
    from speechbrain.inference import EncoderClassifier
    m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").eval()
    fb = m.hparams.modules["compute_features"]
    np.random.seed(0)
    for secs in (1.0, 2.0, 3.0):
        wav = (np.random.randn(int(secs * SAMPLE_RATE)).astype(np.float32)) * 0.1
        ref = fb(torch.tensor(wav).unsqueeze(0)).squeeze(0).detach().numpy()
        mine = compute_fbank(wav)
        d = np.abs(ref - mine)
        print(f"{secs}s  ref{ref.shape} mine{mine.shape}  max|diff|={d.max():.3e} mean={d.mean():.3e}")

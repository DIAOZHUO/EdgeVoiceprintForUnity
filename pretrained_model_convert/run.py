"""
Export speechbrain/spkrec-ecapa-voxceleb to ONNX for edge voice-print.

Design decisions (verified in this repo):
  * The ONNX graph outputs the RAW embedding  = ecapa(mean_var_norm(fbank)).
    This matches SpeechBrain's SpeakerRecognition.verify_batch, which scores
    cosine similarity on encode_batch(..., normalize=False).
  * SpeechBrain's optional embedding normalization `mean_var_norm_emb` is, for
    this model, just a mean subtraction (norm_type="global", std_norm=False ->
    std == 1). We export that 192-d mean vector to `emb_mean.npy` so it can be
    applied on-device as an OPTIONAL flag:  emb_centered = emb - emb_mean.
  * The time axis is DYNAMIC, so one model handles arbitrary-length utterances
    (ECAPA attentive stat-pooling supports this). Input is the log-mel fbank
    feature (1, num_frames, 80) produced by fbank_cs.py, NOT the raw waveform:
    the STFT stays out of the graph so the model runs on any runtime, and
    fbank_cs.py (matmul-only, ported 1:1 to C# in the Unity project) computes it.
  * Exported with the dynamo exporter (torch.onnx.export(..., dynamo=True), the
    default since torch 2.9). The legacy TorchScript exporter is not used, and
    the graph is opset 18: asking dynamo for a lower opset fails on Pad.

Artifacts written next to this file:
  ecapa_tdnn.onnx (+ .data)  fp32, dynamic length, raw embedding (weights in external .onnx.data)
  emb_mean.npy               192-d mean vector for optional on-device centering
  ecapa_tdnn_int8.onnx       dynamically-quantized INT8 model (edge size/speed)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from speechbrain.inference import EncoderClassifier

from pretrained_model_convert.fbank_cs import wav_to_feature

HERE = Path(__file__).resolve().parent
FP32_PATH = HERE / "ecapa_tdnn.onnx"
INT8_PATH = HERE / "ecapa_tdnn_int8.onnx"
MEAN_PATH = HERE / "emb_mean.npy"
OPSET = 18                       # dynamo exporter's native opset; downgrading fails on Pad


class Wrapper(nn.Module):
    """input_norm -> ECAPA. Outputs the RAW 192-d embedding (no emb centering)."""

    def __init__(self, device="cpu"):
        super().__init__()
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        modules = model.hparams.modules
        self.input_norm = modules["mean_var_norm"]       # per-sentence mean/var norm
        self.ecapa = modules["embedding_model"]           # ECAPA-TDNN

        # mean_var_norm_emb is a pure mean subtraction for this model; keep the
        # vector so the caller can optionally center the embedding on-device.
        self.register_buffer(
            "emb_mean",
            model.hparams.mean_var_norm_emb.glob_mean.detach().clone().reshape(-1),
        )

    def forward(self, features):
        # features: (batch, num_frames, 80) log-mel fbank from fbank_cs.py
        batch = features.shape[0]
        lengths = torch.ones(batch, device=features.device)  # full length, no padding
        feats = self.input_norm(features, lengths)
        emb = self.ecapa(feats, lengths)                      # (batch, 1, 192)
        return emb


def export_fp32(wrapper: nn.Module) -> None:
    # 2s @ 16k -> 201 frames. Dummy length only seeds tracing; dim 1 is dynamic.
    dummy = torch.from_numpy(wav_to_feature(torch.randn(32000).numpy()))
    torch.onnx.export(
        wrapper,
        dummy,
        str(FP32_PATH),
        dynamo=True,
        opset_version=OPSET,
        input_names=["waveform_feature"],
        output_names=["embedding"],
        dynamic_axes={
            "waveform_feature": {0: "batch", 1: "num_frames"},
            "embedding": {0: "batch"},
        },
    )
    np.save(MEAN_PATH, wrapper.emb_mean.cpu().numpy())
    print(f"wrote {FP32_PATH.name}  and  {MEAN_PATH.name} (emb_mean, 192-d)")


def quantize_int8() -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"wrote {INT8_PATH.name} (dynamic INT8)")


if __name__ == "__main__":
    wrapper = Wrapper().eval()
    export_fp32(wrapper)
    quantize_int8()
    print("done. Check numerics with test_result/voiceprint_on_clip.py, then "
          "copy to Unity with export_unity.py.")

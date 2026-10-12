"""
Copy the voice-print models into the Unity project.

    python -m pretrained_model_convert.export_unity

Writes into the Unity package, UnityVoicePrint/Packages/com.enzian.voiceprint/Models/:
  ecapa_tdnn_int8.bytes   ECAPA (single-file ONNX, loaded by ONNX Runtime from bytes)
  emb_mean.bytes          192 x float32 LE, optional embedding centering vector

The fp32 model (ecapa_tdnn.onnx + .onnx.data) uses external weights, which
ONNX Runtime cannot load from a byte array. Pass --fp32 to also write a merged
single-file ecapa_tdnn.bytes (~84 MB). With 3 enrollment clips per speaker the
INT8 model scores the same as fp32 on samples_inmu3/ (test_result/voiceprint_on_clip.py).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DST = ROOT / "UnityVoicePrint" / "Packages" / "com.enzian.voiceprint" / "Models"


def main(fp32: bool) -> None:
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "ecapa_tdnn_int8.onnx", DST / "ecapa_tdnn_int8.bytes")
    np.load(HERE / "emb_mean.npy").astype("<f4").tofile(DST / "emb_mean.bytes")
    if fp32:
        import onnx
        m = onnx.load(str(HERE / "ecapa_tdnn.onnx"))       # pulls in .onnx.data
        onnx.save_model(m, str(DST / "ecapa_tdnn.bytes"), save_as_external_data=False)
    print(f"wrote models to {DST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true", help="also write merged single-file fp32 model")
    main(ap.parse_args().fp32)

"""Export best.pt to ONNX (opset 12, fixed 1x3x640x640).

Outputs:
  detections  [8400, 5+K]  (x1, y1, x2, y2, score, mask coeffs x K)
              for every grid cell, before threshold/NMS
  prototypes  [K, 160, 160] prototype masks (pre-sigmoid)
The consumer thresholds, runs NMS on the boxes, then builds each mask
as sigmoid(coeffs . prototypes) cropped to its box (see predict.py).
Updates runs/metrics.json with the ONNX size.
"""
import argparse
import json
import os

import torch

import model as model_mod
from model import IMG_SIZE, PROTO_K, PROTO_SIZE, ExportWrapper, ObjectSeparator


def main(ckpt_path="runs/best.pt", out_dir="runs"):
    n_cells = sum((IMG_SIZE // s) ** 2 for s in (8, 16, 32))  # 8400 @640
    model = ObjectSeparator()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapper = ExportWrapper(model)
    wrapper.eval()

    onnx_path = os.path.join(out_dir, "best.onnx")
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["input"],
            output_names=["detections", "prototypes"],
            opset_version=12, dynamic_axes=None,
        )
    size = os.path.getsize(onnx_path)
    print(f"ONNX exported: {onnx_path} ({size:,} bytes)")

    # Sanity: run the exported graph once via onnxruntime if available.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        det, protos = sess.run(
            None, {"input": np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)})
        assert det.shape == (n_cells, 5 + PROTO_K), det.shape
        assert protos.shape == (PROTO_K, PROTO_SIZE, PROTO_SIZE), protos.shape
        print(f"ONNX check OK: detections {det.shape}, prototypes {protos.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"ONNX runtime check skipped/failed: {e}")

    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["onnx_bytes"] = size
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Updated runs/metrics.json with model size")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/best.pt")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--img", type=int, default=640,
                    help="input size the checkpoint was trained at")
    a = ap.parse_args()
    if a.img != 640:
        model_mod.set_img_size(a.img)
        # Re-read the moved constants into this module's globals.
        globals()["IMG_SIZE"] = model_mod.IMG_SIZE
        globals()["PROTO_SIZE"] = model_mod.PROTO_SIZE
    main(a.ckpt, a.out)

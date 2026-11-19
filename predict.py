"""Run inference on a folder of images + build the layer-split demo.

For every input image, writes into <out>/<stem>/:
  original.jpg   the untouched input
  viz.jpg        each detected object filled/outlined in a distinct colour
  obj_00.png ... one RGBA cutout per object (object on transparency,
                 cropped to its box) -- the "separate layer" output
and appends a line to <out>/results.txt.

Uses the ONNX model if given an .onnx file, else a PyTorch checkpoint.
Masks are assembled as sigmoid(coeffs . prototypes), cropped to the
predicted box, binarized at 0.5, then mapped back from the 640 letterbox
to original image coordinates.
"""
import argparse
import colorsys
import os

import numpy as np
from PIL import Image

import model as model_mod
from evaluate import nms
from model import IMG_SIZE, PROTO_K, PROTO_SIZE

PAD_VALUE = 114


def letterbox_image(img, size=IMG_SIZE):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR),
                 ((size - nw) // 2, (size - nh) // 2))
    return canvas, scale, (size - nw) // 2, (size - nh) // 2, nw, nh


def palette(n):
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def predict_onnx(model_path, arr):
    import onnxruntime as ort

    if not hasattr(predict_onnx, "_sess"):
        predict_onnx._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
    sess = predict_onnx._sess
    inp = arr.transpose(2, 0, 1)[None].astype(np.float32)
    det, protos = sess.run(None, {"input": inp})
    return det, protos


def predict_torch(model_path, arr):
    import torch

    from model import ObjectSeparator, decode_all

    if not hasattr(predict_torch, "_model"):
        m = ObjectSeparator()
        ckpt = torch.load(model_path, map_location="cpu")
        m.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
        m.eval()
        predict_torch._model = m
    m = predict_torch._model
    with torch.no_grad():
        t = torch.from_numpy(arr.transpose(2, 0, 1)[None].astype(np.float32))
        outs, protos = m(t)
        det, protos = decode_all(outs, protos)
    return det[0].numpy(), protos[0].numpy()


def assemble(det, protos, conf, nms_thres):
    """det [N,5+K] canvas coords -> (boxes, scores, masks[PROTO] bool)."""
    det = np.asarray(det, dtype=np.float64)
    det = det[det[:, 4] >= conf]
    if len(det) == 0:
        return (np.zeros((0, 4)), np.zeros(0),
                np.zeros((0, PROTO_SIZE, PROTO_SIZE), dtype=bool))
    keep = nms(det[:, :4], det[:, 4], nms_thres)
    det = det[keep]
    boxes, scores = det[:, :4], det[:, 4]
    coeffs = det[:, 5:5 + PROTO_K]
    proto = np.asarray(protos, dtype=np.float64)
    logits = coeffs @ proto.reshape(PROTO_K, -1)
    masks = 1.0 / (1.0 + np.exp(-logits))
    masks = masks.reshape(-1, PROTO_SIZE, PROTO_SIZE) >= 0.5
    s = PROTO_SIZE / IMG_SIZE
    for n in range(len(masks)):
        x1, y1, x2, y2 = (int(round(v * s)) for v in boxes[n])
        crop = np.zeros_like(masks[n])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(PROTO_SIZE, x2), min(PROTO_SIZE, y2)
        if x2 > x1 and y2 > y1:
            crop[y1:y2, x1:x2] = masks[n, y1:y2, x1:x2]
        masks[n] = crop
    # A "layer" with zero mask pixels is useless -- drop those detections.
    nonempty = masks.reshape(len(masks), -1).any(axis=1)
    return boxes[nonempty], scores[nonempty], masks[nonempty]


def mask_to_original(mask_proto, scale, dx, dy, nw, nh, w0, h0):
    """Proto-res mask (canvas-aligned) -> bool mask at original size."""
    m = Image.fromarray((mask_proto * 255).astype(np.uint8))
    m = m.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.asarray(m)[dy:dy + nh, dx:dx + nw]
    m2 = Image.fromarray(arr).resize((w0, h0), Image.BILINEAR)
    return np.asarray(m2) >= 128


def erode(mask):
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/best.onnx")
    ap.add_argument("--src", required=True, help="folder of input images")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--img", type=int, default=640,
                    help="input size the model was trained/exported at")
    args = ap.parse_args()
    if args.img != 640:
        model_mod.set_img_size(args.img)
        globals()["IMG_SIZE"] = model_mod.IMG_SIZE
        globals()["PROTO_SIZE"] = model_mod.PROTO_SIZE

    os.makedirs(args.out, exist_ok=True)
    use_onnx = args.model.endswith(".onnx")
    names = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    lines = [f"model={args.model} conf={args.conf} images={len(names)}"]
    for name in names:
        path = os.path.join(args.src, name)
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        canvas, scale, dx, dy, nw, nh = letterbox_image(img, size=IMG_SIZE)
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        det, protos = (predict_onnx(args.model, arr) if use_onnx
                       else predict_torch(args.model, arr))
        boxes, scores, masks = assemble(det, protos, args.conf, args.nms)
        stem = os.path.splitext(name)[0]
        odir = os.path.join(args.out, stem)
        os.makedirs(odir, exist_ok=True)
        img.save(os.path.join(odir, "original.jpg"))
        colours = palette(len(boxes))
        viz = np.asarray(img).astype(np.float32)
        orig = np.asarray(img)
        for n in range(len(boxes)):
            mfull = mask_to_original(masks[n], scale, dx, dy, nw, nh, w0, h0)
            col = np.array(colours[n], dtype=np.float32)
            viz[mfull] = viz[mfull] * 0.55 + col * 0.45
            edge = mfull & ~erode(mfull)
            viz[edge] = col
            # RGBA cutout cropped to the box (original coords).
            x1 = min(max((boxes[n][0] - dx) / scale, 0), w0)
            y1 = min(max((boxes[n][1] - dy) / scale, 0), h0)
            x2 = min(max((boxes[n][2] - dx) / scale, 0), w0)
            y2 = min(max((boxes[n][3] - dy) / scale, 0), h0)
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(round(x2)), int(round(y2))
            if ix2 > ix1 and iy2 > iy1:
                rgba = np.zeros((iy2 - iy1, ix2 - ix1, 4), dtype=np.uint8)
                rgba[..., :3] = orig[iy1:iy2, ix1:ix2]
                rgba[..., 3] = (mfull[iy1:iy2, ix1:ix2] * 255).astype(np.uint8)
                Image.fromarray(rgba, "RGBA").save(
                    os.path.join(odir, f"obj_{n:02d}.png"))
        Image.fromarray(viz.astype(np.uint8)).save(os.path.join(odir, "viz.jpg"))
        desc = "; ".join(f"obj{n} {scores[n]:.2f}" for n in range(len(boxes))) \
            or "no detections"
        lines.append(f"{name}: {len(boxes)} object(s): {desc}")
        print(lines[-1])
    with open(os.path.join(args.out, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote layer-split outputs for {len(names)} images to {args.out}")


if __name__ == "__main__":
    main()

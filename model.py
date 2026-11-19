"""Class-agnostic instance segmentation, ~4M params (Sarowar's own model).

One forward pass separates EVERY object in the image -- no class names,
objectness only -- so each detection can become a moveable/editable layer.

Design (extends the phone-detector's anchor-free center head):
  - MobileNet-style DSConv backbone, strides 4/8/16/32 features.
  - Light top-down FPN producing P8/P16/P32.
  - Anchor-free heads on strides 8/16/32: per cell objectness (1) +
    box distances l,t,r,b (4) + mask coefficients (K).
  - Prototype branch from the stride-4 feature: K prototype masks at
    1/4 input resolution (160x160 for a 640 input). An instance mask is
    sigmoid(coeffs . prototypes), cropped to the predicted box
    (YOLOv8-seg style, K=32: the standard prototype count -- enough mask
    basis diversity for clean shapes without blowing up the head).

Input: 640x640 letterboxed image, float tensor in [0, 1].
Normalization (x - 0.5) / 0.5 is applied inside forward().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 640
STRIDES = (8, 16, 32)
PROTO_K = 32          # number of prototype masks / mask coefficients
PROTO_SIZE = IMG_SIZE // 4   # 160
WIDTH = 3.125


def _c(ch, w=WIDTH):
    """Scale a base channel count by the width multiplier (multiple of 8)."""
    return max(8, int(round(ch * w / 8.0)) * 8)


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each with BN + SiLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class SegHead(nn.Module):
    """Two DSConv tower + 1x1 prediction conv -> 5 + K channels
    (obj, l, t, r, b, mask coeffs x K)."""

    def __init__(self, cin, hidden, k=PROTO_K):
        super().__init__()
        self.tower = nn.Sequential(DSConv(cin, hidden), DSConv(hidden, hidden))
        self.pred = nn.Conv2d(hidden, 5 + k, 1)
        # Bias init: objectness starts pessimistic (few positives per image).
        nn.init.constant_(self.pred.bias[0], -4.0)

    def forward(self, x):
        return self.pred(self.tower(x))


class ObjectSeparator(nn.Module):
    def __init__(self, width=WIDTH, k=PROTO_K):
        super().__init__()
        self.k = k
        c = lambda ch: _c(ch, width)
        cf = c(128)  # FPN / head width
        # Backbone (same stage layout as the phone detector).
        self.stem = ConvBNAct(3, c(32), 3, stride=2)      # 320 (s2)
        self.b1 = DSConv(c(32), c(48), stride=2)          # 160 (s4)
        self.b2 = DSConv(c(48), c(64))                    # 160 -> F4
        self.b3 = DSConv(c(64), c(96), stride=2)          # 80  (s8)
        self.b4 = DSConv(c(96), c(96))                    # 80  -> F8
        self.b5 = DSConv(c(96), c(160), stride=2)         # 40  (s16)
        self.b6 = DSConv(c(160), c(160))                  # 40  -> F16
        self.b7 = DSConv(c(160), c(256), stride=2)        # 20  (s32)
        self.b8 = DSConv(c(256), c(256))                  # 20  -> F32
        # Light top-down FPN.
        self.lat32 = nn.Conv2d(c(256), cf, 1)
        self.lat16 = nn.Conv2d(c(160), cf, 1)
        self.lat8 = nn.Conv2d(c(96), cf, 1)
        self.sm32 = DSConv(cf, cf)
        self.sm16 = DSConv(cf, cf)
        self.sm8 = DSConv(cf, cf)
        # Heads on P8 / P16 / P32.
        self.head8 = SegHead(cf, cf, k)
        self.head16 = SegHead(cf, cf, k)
        self.head32 = SegHead(cf, cf, k)
        # Prototype branch from F4: K masks at 1/4 resolution.
        self.proto_tower = DSConv(c(64), cf)
        self.proto_pred = nn.Conv2d(cf, k, 1)
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        """Returns (head_outputs, protos):
        head_outputs: list of 3 tensors [B, 5+K, H, W] in STRIDES order;
        protos: [B, K, 160, 160] prototype masks (pre-sigmoid)."""
        x = (x - self.mean) / self.std
        x = self.stem(x)
        f4 = self.b2(self.b1(x))
        f8 = self.b4(self.b3(f4))
        f16 = self.b6(self.b5(f8))
        f32 = self.b8(self.b7(f16))
        p32 = self.sm32(self.lat32(f32))
        p16 = self.sm16(self.lat16(f16) + F.interpolate(
            p32, size=f16.shape[-2:], mode="nearest"))
        p8 = self.sm8(self.lat8(f8) + F.interpolate(
            p16, size=f8.shape[-2:], mode="nearest"))
        protos = self.proto_pred(self.proto_tower(f4))
        return [self.head8(p8), self.head16(p16), self.head32(p32)], protos


def decode(outputs, protos, img_size=IMG_SIZE, conf_thres=0.25):
    """Decode raw head outputs to per-image detections with masks.

    Returns a list (per image) of dicts:
      boxes  [N,4] xyxy in img_size canvas coords
      scores [N]
      masks  [N, PROTO_SIZE, PROTO_SIZE] bool, binarized at 0.5 and
             cropped to the predicted box (in proto coords)
    Before NMS (NMS is the caller's job, on boxes)."""
    k = protos.shape[1]
    results = []
    batch = outputs[0].shape[0]
    per_image = [[] for _ in range(batch)]
    for out, stride in zip(outputs, STRIDES):
        b, _, h, w = out.shape
        obj = out[:, 0].sigmoid()                        # [B,H,W]
        dist = F.softplus(out[:, 1:5]) * stride           # [B,4,H,W] pixels
        coeffs = out[:, 5:5 + k]                          # [B,K,H,W]
        ys, xs = torch.meshgrid(
            torch.arange(h, device=out.device, dtype=out.dtype),
            torch.arange(w, device=out.device, dtype=out.dtype),
            indexing="ij",
        )
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        x1 = cx - dist[:, 0]
        y1 = cy - dist[:, 1]
        x2 = cx + dist[:, 2]
        y2 = cy + dist[:, 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)     # [B,H,W,4]
        for i in range(b):
            sel = obj[i] >= conf_thres
            if sel.any():
                per_image[i].append(torch.cat([
                    boxes[i][sel],
                    obj[i][sel].unsqueeze(-1),
                    coeffs[i].permute(1, 2, 0)[sel],
                ], dim=-1))                               # [n, 5+K]
    scale = PROTO_SIZE / img_size
    for i in range(batch):
        if not per_image[i]:
            results.append({
                "boxes": torch.zeros((0, 4), device=protos.device),
                "scores": torch.zeros((0,), device=protos.device),
                "masks": torch.zeros((0, PROTO_SIZE, PROTO_SIZE),
                                     dtype=torch.bool, device=protos.device),
            })
            continue
        det = torch.cat(per_image[i], dim=0)              # [N, 5+K]
        boxes = det[:, :4]
        scores = det[:, 4]
        coeff = det[:, 5:5 + k]                           # [N,K]
        proto = protos[i]                                 # [K,P,P]
        masks = torch.sigmoid(coeff @ proto.reshape(k, -1))
        masks = masks.reshape(-1, PROTO_SIZE, PROTO_SIZE)
        # Crop each mask to its predicted box (proto coords).
        pb = (boxes * scale).round().long()
        pb[:, [0, 2]] = pb[:, [0, 2]].clamp(0, PROTO_SIZE)
        pb[:, [1, 3]] = pb[:, [1, 3]].clamp(0, PROTO_SIZE)
        keep = torch.zeros_like(masks, dtype=torch.bool)
        for n in range(masks.shape[0]):
            px1, py1, px2, py2 = (int(v) for v in pb[n])
            if px2 > px1 and py2 > py1:
                keep[n, py1:py2, px1:px2] = True
        masks = (masks >= 0.5) & keep
        results.append({"boxes": boxes, "scores": scores, "masks": masks})
    return results


def decode_all(outputs, protos):
    """Fixed-shape decode for export: returns (detections, protos) with
    detections [B, 8400, 5+K] = (x1, y1, x2, y2, score, coeffs...) for
    every grid cell in IMG_SIZE coords, no thresholding. Threshold + NMS
    + mask assembly are the consumer's job (keeps the ONNX graph static)."""
    k = protos.shape[1]
    batch = outputs[0].shape[0]
    parts = []
    for out, stride in zip(outputs, STRIDES):
        b, _, h, w = out.shape
        obj = out[:, 0].sigmoid()
        dist = F.softplus(out[:, 1:5]) * stride
        ys, xs = torch.meshgrid(
            torch.arange(h, device=out.device, dtype=out.dtype),
            torch.arange(w, device=out.device, dtype=out.dtype),
            indexing="ij",
        )
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        boxes = torch.stack(
            [cx - dist[:, 0], cy - dist[:, 1], cx + dist[:, 2], cy + dist[:, 3]],
            dim=-1,
        )                                                 # [B,H,W,4]
        coeff = out[:, 5:5 + k].permute(0, 2, 3, 1)      # [B,H,W,K]
        parts.append(torch.cat(
            [boxes, obj.unsqueeze(-1), coeff], dim=-1).reshape(b, h * w, 5 + k))
    return torch.cat(parts, dim=1), protos


class ExportWrapper(nn.Module):
    """ONNX-friendly wrapper: input [1,3,640,640] in [0,1] ->
    (detections [8400, 5+K], prototypes [K,160,160])."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outs, protos = self.model(x)
        det, protos = decode_all(outs, protos)
        return det[0], protos[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ObjectSeparator()
    n = count_parameters(m)
    print(f"Parameters: {n:,} ({n / 1e6:.3f}M)")
    outs, protos = m(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
    print("Head outputs:", [tuple(t.shape) for t in outs])
    print("Protos:", tuple(protos.shape))
    det = decode(outs, protos, conf_thres=0.25)
    print("Decode ok:", {k: tuple(v.shape) for k, v in det[0].items()})

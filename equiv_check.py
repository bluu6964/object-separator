"""Equivalence proofs for the optimized data/target pipeline.

The optimized train.py must produce BIT-IDENTICAL training inputs to
the original implementation:

  1. build_targets (vectorized) vs the original per-cell loop version
     (copied verbatim below): obj/box/pos/inst tensors exactly equal,
     on synthetic edge cases and on real dataset batches.
  2. Precomputed mask_cells (from build_targets) vs the original
     per-step search inside compute_loss: identical cell lists.
  3. compute_loss with mask_cells vs without: identical loss values.
  4. SegDataset with the rasterization cache vs live rasterization:
     identical items with augmentation OFF, and identical items with
     augmentation ON when both datasets are seeded identically and
     iterated in lockstep (same RNG draw sequence).

Run AFTER build_cache.py for the local data:
  python equiv_check.py [--img 512]
Exits non-zero on the first mismatch.
"""
import argparse
import math
import random
import sys

import numpy as np
import torch

import model as model_mod
import train as train_mod

IMG_SIZE = train_mod.IMG_SIZE  # set in main via set_img_size
STRIDES = train_mod.STRIDES
MASK_CELLS_PER_INST = train_mod.MASK_CELLS_PER_INST


# ---------------------------------------------------------------------
# Original implementations, copied verbatim from train.py @ 330d4b7.
# ---------------------------------------------------------------------
def build_targets_old(boxes_list, device):
    targets = []
    for si, stride in enumerate(STRIDES):
        g = IMG_SIZE // stride
        b = len(boxes_list)
        obj = torch.zeros((b, g, g), device=device)
        box = torch.zeros((b, 4, g, g), device=device)
        pos = torch.zeros((b, g, g), dtype=torch.bool, device=device)
        inst = torch.full((b, g, g), -1, dtype=torch.long, device=device)
        for i, boxes in enumerate(boxes_list):
            if len(boxes) == 0:
                continue
            boxes = boxes.to(device)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            claimed = {}
            for bi in range(len(boxes)):
                x1, y1, x2, y2 = boxes[bi].tolist()
                gx1 = max(0, int(math.floor(x1 / stride)))
                gy1 = max(0, int(math.floor(y1 / stride)))
                gx2 = min(g - 1, int(math.floor((x2 - 1e-6) / stride)))
                gy2 = min(g - 1, int(math.floor((y2 - 1e-6) / stride)))
                for gy in range(gy1, gy2 + 1):
                    for gx in range(gx1, gx2 + 1):
                        cx, cy = (gx + 0.5) * stride, (gy + 0.5) * stride
                        if x1 <= cx < x2 and y1 <= cy < y2:
                            key = (gy, gx)
                            if key not in claimed or areas[bi] < claimed[key][0]:
                                claimed[key] = (areas[bi], bi, cx, cy)
            if stride == STRIDES[0]:
                for bi in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[bi].tolist()
                    gx = min(g - 1, max(0, int(((x1 + x2) / 2) / stride)))
                    gy = min(g - 1, max(0, int(((y1 + y2) / 2) / stride)))
                    key = (gy, gx)
                    if key not in claimed:
                        claimed[key] = (areas[bi], bi,
                                        (gx + 0.5) * stride, (gy + 0.5) * stride)
            for (gy, gx), (_, bi, cx, cy) in claimed.items():
                x1, y1, x2, y2 = boxes[bi].tolist()
                obj[i, gy, gx] = 1.0
                pos[i, gy, gx] = True
                inst[i, gy, gx] = bi
                box[i, 0, gy, gx] = (cx - x1) / stride
                box[i, 1, gy, gx] = (cy - y1) / stride
                box[i, 2, gy, gx] = (x2 - cx) / stride
                box[i, 3, gy, gx] = (y2 - cy) / stride
        targets.append((obj, box, pos, inst))
    return targets


def mask_cells_old(targets, boxes_list):
    """The original per-step search from compute_loss, returning the
    truncated (si, gy, gx) lists it would have used."""
    out = []
    for i, boxes in enumerate(boxes_list):
        per = []
        for m in range(len(boxes)):
            cells = []
            x1, y1, x2, y2 = boxes[m].tolist()
            ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
            for si, stride in enumerate(STRIDES):
                inst_map = targets[si][3][i]
                ys, xs = torch.nonzero(inst_map == m, as_tuple=True)
                for gy, gx in zip(ys.tolist(), xs.tolist()):
                    d = ((gx + 0.5) * stride - ccx) ** 2 + ((gy + 0.5) * stride - ccy) ** 2
                    cells.append((d, si, gy, gx))
            cells.sort(key=lambda t: t[0])
            per.append([(si, gy, gx)
                        for (_, si, gy, gx) in cells[:MASK_CELLS_PER_INST]])
        out.append(per)
    return out


# ---------------------------------------------------------------------
def check_targets(boxes_list, tag):
    dev = torch.device("cpu")
    new_t, new_cells = train_mod.build_targets(boxes_list, dev)
    old_t = build_targets_old(boxes_list, dev)
    old_cells = mask_cells_old(old_t, boxes_list)
    for si in range(len(STRIDES)):
        for j, name in enumerate(("obj", "box", "pos", "inst")):
            assert torch.equal(new_t[si][j], old_t[si][j]), \
                f"{tag}: scale {si} {name} differs"
    assert new_cells == old_cells, f"{tag}: mask_cells differ"
    n_pos = sum(int(t[2].sum()) for t in new_t)
    print(f"  OK {tag}: targets identical, mask_cells identical "
          f"({n_pos} positive cells)")


def synthetic_cases():
    S = IMG_SIZE
    rng = random.Random(7)
    cases = {
        "empty image in batch": [torch.zeros((0, 4)),
                                 torch.tensor([[10., 10., 50., 60.]])],
        "single tiny box": [torch.tensor([[100., 100., 103., 104.]])],
        "full-canvas box": [torch.tensor([[0., 0., float(S), float(S)]])],
        "identical twin boxes (area tie)": [
            torch.tensor([[50., 50., 200., 200.],
                          [50., 50., 200., 200.]])],
        "shared centre, nested": [
            torch.tensor([[100., 100., 300., 300.],
                          [150., 150., 250., 250.],
                          [190., 190., 210., 210.]])],
        "box sliver at edge": [torch.tensor([[0., 0., 2., float(S)]])],
        "many random boxes": [
            torch.tensor([[rng.uniform(0, S - 40), rng.uniform(0, S - 40),
                           rng.uniform(0, S - 40) + rng.uniform(2, 40),
                           rng.uniform(0, S - 40) + rng.uniform(2, 40)]
                          for _ in range(25)])],
    }
    # fix the random case: ensure x2>x1, y2>y1 properly
    fixed = []
    for row in cases["many random boxes"][0].tolist():
        x1, y1, w, h = row
        fixed.append([x1, y1, x1 + abs(w - x1) % 200 + 2,
                      y1 + abs(h - y1) % 200 + 2])
    cases["many random boxes"] = [torch.tensor(fixed)]
    for tag, boxes_list in cases.items():
        check_targets(boxes_list, f"synthetic/{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--img", type=int, default=512)
    args = ap.parse_args()

    model_mod.set_img_size(args.img)
    train_mod.IMG_SIZE = model_mod.IMG_SIZE
    train_mod.PROTO_SIZE = model_mod.PROTO_SIZE
    globals()["IMG_SIZE"] = model_mod.IMG_SIZE
    dev = torch.device("cpu")

    print("[1] build_targets + mask_cells: synthetic edge cases")
    synthetic_cases()

    print("[2] build_targets + mask_cells: real dataset batches")
    ds = train_mod.SegDataset(args.data, "train", augment=False, seed=42)
    ds.use_cache = False
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                        collate_fn=train_mod.collate)
    for bi, (_, boxes, _) in enumerate(loader):
        check_targets(boxes, f"real batch {bi}")
        if bi >= 4:
            break

    print("[3] compute_loss: mask_cells vs searched (same outputs)")
    torch.manual_seed(123)
    model = model_mod.ObjectSeparator().eval()
    imgs, boxes, masks = next(iter(loader))
    targets, cells = train_mod.build_targets(boxes, dev)
    with torch.no_grad():
        outs, protos = model(imgs)
        a = train_mod.compute_loss(outs, protos, targets, boxes, masks,
                                   mask_cells=cells)
        b = train_mod.compute_loss(outs, protos, targets, boxes, masks,
                                   mask_cells=None)
    for name, x, y in zip(("loss", "obj", "box", "iou", "mask", "npos"),
                          a, b):
        assert x == y, f"loss {name}: {x} != {y}"
    print(f"  OK loss identical: {a[0]:.6f} (obj {a[1]:.4f} box {a[2]:.4f} "
          f"iou {a[3]:.4f} mask {a[4]:.4f}, npos {a[5]})")

    print("[4] dataset: cache vs live rasterization")
    for augment in (False, True):
        ds_live = train_mod.SegDataset(args.data, "train", augment=augment,
                                       seed=42)
        ds_live.use_cache = False
        ds_cache = train_mod.SegDataset(args.data, "train", augment=augment,
                                        seed=42)
        ds_cache.use_cache = True
        assert ds_cache._load_cache(ds_cache.files[0]) is not None, \
            "no cache entries found -- run build_cache.py first"
        n = 24
        for i in range(n):
            a = ds_live[i]
            b = ds_cache[i]
            assert torch.equal(a[0], b[0]), f"augment={augment} item {i}: image"
            assert torch.equal(a[1], b[1]), f"augment={augment} item {i}: boxes"
            assert torch.equal(a[2], b[2]), f"augment={augment} item {i}: masks"
        print(f"  OK augment={augment}: {n} items identical "
              f"(image, boxes, masks)")

    print("ALL EQUIVALENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

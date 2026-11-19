"""Train the class-agnostic object separator.

Losses (anchor-free, FCOS-style assignment -- cells inside a GT box,
smallest-area instance wins a contested cell, center-cell fallback at
stride 8 so every GT gets >= 1 positive):
  - objectness: BCE over all grid cells, all scales
  - box: smooth-L1 on (l,t,r,b)/stride + (1 - IoU) on decoded boxes,
    positive cells only
  - mask: BCE + Dice between sigmoid(coeffs . prototypes) and the GT
    mask, cropped to the GT box at prototype resolution; computed for
    up to MASK_CELLS_PER_INST positive cells per GT instance (the cells
    nearest the instance centre), positives only.

Augmentation: horizontal flip, scale/translate jitter, brightness/
contrast (image and masks are transformed consistently).

Logs per-epoch train losses + val box mAP@0.5 and mask mAP@0.5
(class-agnostic); saves best.pt (by mask mAP) / last.pt and
runs/metrics.json. --resume continues from last.pt toward the TOTAL
--epochs target (chained-job pattern from the phone-detector project).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from evaluate import evaluate_maps
from model import (IMG_SIZE, PROTO_K, PROTO_SIZE, STRIDES, ObjectSeparator,
                   count_parameters)

PAD_VALUE = 114
MASK_CELLS_PER_INST = 4


def letterbox_params(w, h, size=IMG_SIZE):
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    return scale, dx, dy, nw, nh


class SegDataset(Dataset):
    """Items: (image tensor [3,640,640] in [0,1], boxes [M,4] xyxy in
    canvas coords, masks [M,160,160] float in canvas/proto coords)."""

    def __init__(self, root, split, augment=False, limit=None, seed=42):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.files = sorted(f for f in os.listdir(self.img_dir)
                            if f.endswith(".jpg"))
        if limit:
            rng = random.Random(seed)
            rng.shuffle(self.files)
            self.files = self.files[:limit]
        self.augment = augment
        self.rng = random.Random(seed + (1 if augment else 0))

    def __len__(self):
        return len(self.files)

    def _load_label(self, name):
        stem = os.path.splitext(name)[0]
        with open(os.path.join(self.lbl_dir, stem + ".json")) as f:
            return json.load(f)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        w, h = img.size
        label = self._load_label(name)
        scale, dx, dy, nw, nh = letterbox_params(w, h)
        # Letterbox the image.
        canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE),
                           (PAD_VALUE, PAD_VALUE, PAD_VALUE))
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
        img = canvas
        # Transform instances to canvas coords; rasterize masks at 640.
        boxes = []
        mask_imgs = []
        for inst in label["instances"]:
            bx, by, bw, bh = inst["bbox"]
            x1, y1 = bx * scale + dx, by * scale + dy
            x2, y2 = (bx + bw) * scale + dx, (by + bh) * scale + dy
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            m = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
            draw = ImageDraw.Draw(m)
            for poly in inst["poly"]:
                pts = [(poly[i] * scale + dx, poly[i + 1] * scale + dy)
                       for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=1)
            if np.asarray(m).sum() == 0:
                continue
            boxes.append([x1, y1, x2, y2])
            mask_imgs.append(m)
        rng = self.rng
        if self.augment:
            if rng.random() < 0.5:  # horizontal flip (image + masks + boxes)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_imgs = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in mask_imgs]
                boxes = [[IMG_SIZE - b[2], b[1], IMG_SIZE - b[0], b[3]]
                         for b in boxes]
            if rng.random() < 0.7:  # scale/translate jitter (all follow)
                sc = rng.uniform(0.8, 1.2)
                nwj, nhj = int(IMG_SIZE * sc), int(IMG_SIZE * sc)
                ox = rng.randint(min(0, IMG_SIZE - nwj), max(0, IMG_SIZE - nwj)) if nwj != IMG_SIZE else 0
                oy = rng.randint(min(0, IMG_SIZE - nhj), max(0, IMG_SIZE - nhj)) if nhj != IMG_SIZE else 0

                def jitter(im, fill):
                    out = Image.new(im.mode, (IMG_SIZE, IMG_SIZE), fill)
                    out.paste(im.resize((nwj, nhj), Image.BILINEAR), (ox, oy))
                    return out
                img = jitter(img, (PAD_VALUE, PAD_VALUE, PAD_VALUE))
                mask_imgs = [jitter(m, 0) for m in mask_imgs]
                boxes = [[b[0] * sc + ox, b[1] * sc + oy,
                          b[2] * sc + ox, b[3] * sc + oy] for b in boxes]
            if rng.random() < 0.5:  # brightness / contrast (image only)
                img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
                img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, IMG_SIZE)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, IMG_SIZE)
        if mask_imgs:
            stack = np.stack([np.asarray(m, dtype=np.float32)
                              for m in mask_imgs])          # [M,640,640]
            t = torch.from_numpy(stack).unsqueeze(1)          # [M,1,640,640]
            small = F.max_pool2d(t, IMG_SIZE // PROTO_SIZE)   # [M,1,160,160]
            masks = small.squeeze(1)
            # Drop instances whose box died under augmentation clipping.
            keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
            boxes = boxes[keep]
            masks = masks[torch.from_numpy(keep)]
        else:
            masks = torch.zeros((0, PROTO_SIZE, PROTO_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.from_numpy(boxes), masks


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    boxes = [b[1] for b in batch]
    masks = [b[2] for b in batch]
    return imgs, boxes, masks


def build_targets(boxes_list, device):
    """Per scale: obj [B,H,W], box [B,4,H,W] (stride units), positive
    mask [B,H,W], instance index [B,H,W] (-1 = none)."""
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


def compute_loss(outputs, protos, targets, boxes_list, masks_list):
    total_obj = 0.0
    total_box = 0.0
    total_iou = 0.0
    n_pos_total = 0
    for out, (obj_t, box_t, pos, _inst) in zip(outputs, targets):
        obj_logits = out[:, 0]
        total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits, obj_t)
        n_pos = int(pos.sum())
        n_pos_total += n_pos
        if n_pos > 0:
            stride = IMG_SIZE // out.shape[-1]
            pred = F.softplus(out[:, 1:5])
            pred_pos = pred.permute(0, 2, 3, 1)[pos]
            tgt_pos = box_t.permute(0, 2, 3, 1)[pos]
            total_box = total_box + F.smooth_l1_loss(pred_pos, tgt_pos)
            gy, gx = torch.meshgrid(
                torch.arange(out.shape[-2], device=out.device),
                torch.arange(out.shape[-1], device=out.device), indexing="ij")
            cen = torch.stack([(gx + 0.5) * stride, (gy + 0.5) * stride], dim=-1)
            cen_full = cen.unsqueeze(0).expand(out.shape[0], -1, -1, -1)
            cen_pos = cen_full[pos]
            px1 = cen_pos[:, 0] - pred_pos[:, 0] * stride
            py1 = cen_pos[:, 1] - pred_pos[:, 1] * stride
            px2 = cen_pos[:, 0] + pred_pos[:, 2] * stride
            py2 = cen_pos[:, 1] + pred_pos[:, 3] * stride
            tx1 = cen_pos[:, 0] - tgt_pos[:, 0] * stride
            ty1 = cen_pos[:, 1] - tgt_pos[:, 1] * stride
            tx2 = cen_pos[:, 0] + tgt_pos[:, 2] * stride
            ty2 = cen_pos[:, 1] + tgt_pos[:, 3] * stride
            ix1 = torch.maximum(px1, tx1); iy1 = torch.maximum(py1, ty1)
            ix2 = torch.minimum(px2, tx2); iy2 = torch.minimum(py2, ty2)
            inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
            area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
            area_t = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
            iou = inter / (area_p + area_t - inter + 1e-9)
            total_iou = total_iou + (1.0 - iou).mean()

    # ---- mask loss (positives only, <= MASK_CELLS_PER_INST per instance)
    mask_losses = []
    proto_flat_cache = {}
    for i, boxes in enumerate(boxes_list):
        if len(boxes) == 0:
            continue
        gt_masks = masks_list[i].to(protos.device)          # [M,160,160]
        proto_flat = proto_flat_cache.setdefault(
            i, protos[i].reshape(PROTO_K, -1))              # [K, 160*160]
        bc = boxes.to(protos.device)
        for m in range(len(boxes)):
            cells = []  # (dist_to_centre, scale_idx, gy, gx)
            x1, y1, x2, y2 = bc[m].tolist()
            ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
            for si, stride in enumerate(STRIDES):
                inst_map = targets[si][3][i]
                ys, xs = torch.nonzero(inst_map == m, as_tuple=True)
                for gy, gx in zip(ys.tolist(), xs.tolist()):
                    d = ((gx + 0.5) * stride - ccx) ** 2 + ((gy + 0.5) * stride - ccy) ** 2
                    cells.append((d, si, gy, gx))
            if not cells:
                continue
            cells.sort(key=lambda t: t[0])
            cells = cells[:MASK_CELLS_PER_INST]
            coeffs = torch.stack([
                outputs[si][i, 5:5 + PROTO_K, gy, gx]
                for (_, si, gy, gx) in cells])              # [n,K]
            logits = (coeffs @ proto_flat).reshape(
                -1, PROTO_SIZE, PROTO_SIZE)                 # [n,160,160]
            s = PROTO_SIZE / IMG_SIZE
            px1, py1 = max(0, int(x1 * s)), max(0, int(y1 * s))
            px2, py2 = min(PROTO_SIZE, int(math.ceil(x2 * s))), min(PROTO_SIZE, int(math.ceil(y2 * s)))
            if px2 <= px1 or py2 <= py1:
                continue
            pred = logits[:, py1:py2, px1:px2]
            tgt = gt_masks[m, py1:py2, px1:px2].unsqueeze(0).expand_as(pred)
            bce = F.binary_cross_entropy_with_logits(pred, tgt)
            prob = torch.sigmoid(pred)
            inter = (prob * tgt).sum()
            denom = prob.sum() + tgt.sum()
            dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
            mask_losses.append(bce + dice)
    total_mask = (torch.stack(mask_losses).mean() if mask_losses
                  else torch.zeros((), device=protos.device))
    loss = total_obj + 2.0 * total_box + total_iou + total_mask
    return (loss, float(total_obj), float(total_box), float(total_iou),
            float(total_mask), n_pos_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/last.pt if it exists; --epochs "
                         "is the TOTAL target epoch count")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after N optimizer steps (timing probe mode; "
                         "no checkpoints or metrics are written)")
    ap.add_argument("--save-every-min", type=float, default=30.0,
                    help="also save last.pt at least this often (wall time)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.out, exist_ok=True)

    model = ObjectSeparator()
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.3f}M)", flush=True)
    if not (3_000_000 <= n_params <= 5_000_000):
        print("WARNING: parameter count outside 3M-5M target", flush=True)

    train_ds = SegDataset(args.data, "train", augment=True,
                          limit=args.limit_train, seed=args.seed)
    val_ds = SegDataset(args.data, "val", augment=False, seed=args.seed)
    print(f"Train images: {len(train_ds)}  Val images: {len(val_ds)}", flush=True)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=collate,
                        drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs))

    start_epoch = 0
    best_map = -1.0
    history = []
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        else:
            for _ in range(int(ckpt.get("epoch", 0))):
                sched.step()
        start_epoch = int(ckpt.get("epoch", 0))
        best_map = float(ckpt.get("best_map", -1.0))
        history = ckpt.get("history", [])
        print(f"Resumed from {last_path}: completed epochs={start_epoch}, "
              f"best_val_mask_mAP50={best_map:.4f}", flush=True)
    if start_epoch >= args.epochs and not args.max_steps:
        print(f"Already at target: {start_epoch} >= {args.epochs} epochs. "
              "Nothing to do.", flush=True)
        return

    def save_last(completed_epochs, val_mask_map):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": completed_epochs,
                    "val_mask_map50": val_mask_map, "best_map": best_map,
                    "params": n_params, "history": history}, last_path)

    def write_metrics(final=None):
        metrics = {"params": n_params, "params_m": round(n_params / 1e6, 3),
                   "epochs": args.epochs,
                   "completed_epochs": history[-1]["epoch"] if history else start_epoch,
                   "best_val_mask_map50": best_map, "history": history}
        if final is not None:
            metrics["final_val_box_map50"] = final[0]
            metrics["final_val_mask_map50"] = final[1]
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    probe = args.max_steps is not None
    last_save = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run = [0.0] * 5  # loss, obj, box, iou, mask
        steps = 0
        imgs_seen = 0
        for imgs, boxes, masks in loader:
            imgs = imgs.to(device)
            targets = build_targets(boxes, device)
            outs, protos = model(imgs)
            loss, lo, lb, li, lm, _ = compute_loss(
                outs, protos, targets, boxes, masks)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            for j, v in enumerate((loss, lo, lb, li, lm)):
                run[j] += float(v)
            steps += 1
            imgs_seen += int(imgs.shape[0])
            if probe and steps >= args.max_steps:
                dt = time.time() - t0
                print(f"PROBE steps={steps} seconds={dt:.1f} "
                      f"s_per_step={dt / steps:.3f} "
                      f"img_per_sec={imgs_seen / dt:.1f} "
                      f"steps_per_epoch={len(loader)}", flush=True)
                return
            if not probe and (time.time() - last_save) > args.save_every_min * 60:
                save_last(epoch, history[-1]["val_mask_map50"] if history else -1.0)
                last_save = time.time()
                print(f"  [periodic save] last.pt at epoch-in-progress "
                      f"{epoch + 1}", flush=True)
        sched.step()
        dt_train = time.time() - t0
        box_map, mask_map = evaluate_maps(model, val_ds, device,
                                          limit=args.eval_limit)
        dt = time.time() - t0
        rec = {"epoch": epoch + 1,
               "loss": run[0] / max(1, steps), "obj": run[1] / max(1, steps),
               "box": run[2] / max(1, steps), "iou_loss": run[3] / max(1, steps),
               "mask": run[4] / max(1, steps),
               "val_box_map50": box_map, "val_mask_map50": mask_map,
               "seconds": round(dt, 1), "steps": steps,
               "img_per_sec": round(imgs_seen / max(dt_train, 1e-9), 1)}
        history.append(rec)
        print(f"epoch {epoch + 1}/{args.epochs} loss={rec['loss']:.4f} "
              f"obj={rec['obj']:.4f} box={rec['box']:.4f} "
              f"iou={rec['iou_loss']:.4f} mask={rec['mask']:.4f} "
              f"val_box_mAP50={box_map:.4f} val_mask_mAP50={mask_map:.4f} "
              f"({dt:.0f}s, {rec['img_per_sec']:.0f} img/s)", flush=True)
        if mask_map >= best_map:
            best_map = mask_map
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_mask_map50": mask_map, "val_box_map50": box_map,
                        "params": n_params},
                       os.path.join(args.out, "best.pt"))
        save_last(epoch + 1, mask_map)
        last_save = time.time()
        write_metrics()

    final = evaluate_maps(model, val_ds, device, limit=None)
    print(f"Final val (full staged val): box mAP@0.5={final[0]:.4f} "
          f"mask mAP@0.5={final[1]:.4f}", flush=True)
    write_metrics(final=final)
    print("Wrote runs/metrics.json", flush=True)


if __name__ == "__main__":
    main()

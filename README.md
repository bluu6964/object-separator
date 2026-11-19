# Object Separator

Sarowar's own **class-agnostic instance segmentation** model: image in →
every object separated with its own mask — no class names, objectness
only — so each object can become a moveable/editable layer (e.g. inside
Timeline Pro). Fully offline, hand-rolled PyTorch (no ultralytics / no
heavy framework), trained on GitHub Actions CPU, exported to ONNX.

Sibling project: [phone-detector](https://github.com/bluu6964/phone-detector)
(same patterns: anchor-free center head, letterbox, chained resumable
Actions training, ONNX with fixed-shape decode in-graph).

## Model (3,941,367 params ≈ 3.94M)

- Input: 640×640 letterbox, `[0,1]` tensor; normalization inside the graph.
- Backbone: MobileNet-style DSConv (depthwise + pointwise), features at
  strides 4/8/16/32, width multiplier 3.125.
- Light top-down FPN → P8/P16/P32.
- Anchor-free heads (strides 8/16/32): per cell **objectness (1) + box
  distances l,t,r,b (4) + mask coefficients (32)**.
- Prototype branch from the stride-4 feature: **K = 32 prototype masks
  at 160×160** (1/4 input res). Instance mask =
  `sigmoid(coeffs · prototypes)`, cropped to the predicted box
  (YOLOv8-seg style; K=32 is the standard prototype count — enough mask
  basis diversity without bloating the head).
- Post: score threshold + box NMS; masks binarized at 0.5.
- ONNX (fp32) ≈ 15.7 MB; outputs `detections [8400, 37]` (every grid
  cell: box + score + coeffs) and `prototypes [32, 160, 160]`.

Losses: objectness BCE; box smooth-L1 + (1 − IoU) on positives; mask
BCE + Dice on positives only (≤ 4 cells per GT instance, nearest its
centre), cropped to the GT box at prototype resolution. Assignment is
FCOS-style: cells inside a GT box, smallest-area instance wins a
contested cell, centre-cell fallback at stride 8.

## Data (COCO 2017 instance segmentation, labels dropped)

Every category is ignored — each usable instance is just "an object".
Usable = `iscrowd == 0` and polygon segmentation (crowd/RLE excluded).

Measured from the annotation JSONs:

| split | images | images with ≥1 instance | usable instances | crowd skipped |
|---|---|---|---|---|
| train2017 | 118,287 | 117,266 | 849,949 | 10,052 |
| val2017 | 5,000 | 4,952 | 36,335 | 446 |

Smoke subset (seed 42): 500 train images (3,655 instances) + 100 val
images (703 instances).

```bash
python data_prep.py --counts-only   # verify corpus counts from the JSONs
python data_prep.py --smoke         # stage the smoke subset
python data_prep.py --full          # stage the full corpus
```

## Smoke run (path verification, not accuracy) — DONE 2026-11-19

```bash
gh workflow run smoke.yml           # or the Actions tab
```

Runs data prep, trains 2 epochs, **resumes** to 5 total epochs (proves
the chained-job checkpoint pattern), exports ONNX, and builds the
layer-split demo on 5 val images: for each — `original.jpg`, `viz.jpg`
(each object in a distinct colour) and `obj_NN.png` RGBA cutouts
(the separate layers). Artifact: `object-separator-smoke`.

**Measured results** — run
[44503697787](https://github.com/bluu6964/object-separator/actions/runs/44503697787)
(56 min, all steps green):

| epoch | total loss | obj | box | iou | mask | val box mAP@0.5 | val mask mAP@0.5 |
|---|---|---|---|---|---|---|---|
| 1 | 50.60 | 1.804 | 22.63 | 2.388 | 1.144 | 0.0138 | 0.0050 |
| 2 | 39.95 | 1.272 | 17.81 | 2.051 | 1.007 | 0.0168 | 0.0028 |
| 3 (resumed) | 38.27 | 1.269 | 16.98 | 2.045 | 1.000 | 0.0174 | 0.0031 |
| 4 | 34.82 | 1.131 | 15.41 | 1.882 | 0.994 | 0.0241 | 0.0060 |
| 5 | 36.54 | 1.245 | 16.16 | 1.973 | 0.998 | 0.0185 | 0.0046 |

- All losses finite; total loss fell 50.6 → ~35–37 (epoch 5 ticked up —
  500-image smoke noise). Resume verified: run 2 logged
  `Resumed from runs/last.pt: completed epochs=2`.
- mAP is near zero, as expected: the smoke goal is path verification,
  not accuracy. Best val mask mAP@0.5 = 0.0060 (epoch 4).
- ONNX from the runner: **15,751,355 bytes**, loads in onnxruntime,
  outputs `detections (8400, 37)` + `prototypes (32, 160, 160)`;
  ~0.9 s/image inference on a 1.5-CPU shared VM (random input).
- Layer-split demo produced for 5 val images (original + viz + RGBA
  cutouts). At conf 0.25 the 5-epoch model over-fires (38–105
  detections/image, masks still box-like) — an objectness-calibration
  issue that only real training fixes, not a pipeline defect.
- **Measured runner throughput: ~0.9 img/s train-only at 640×640,
  batch 8 (≈ 8.9 s/step); eval ≈ 0.68 s/img.** This drives the
  full-training estimate below.

Local quick check (needs the CPU torch venv):

```bash
python model.py                     # param count + output shapes
python train.py --epochs 5 --eval-limit 100          # smoke training
python export.py                                     # runs/best.pt -> ONNX
python predict.py --model runs/best.onnx --src samples_in --out samples_out
```

## Full training (chained, resumable — not yet started)

Same pattern as phone-detector v2: `data-pack.yml` builds the full
dataset pack once (release `data-full`); `train.yml` restores
`last.pt`/`best.pt` from release `ckpt-v1`, trains toward the TOTAL
`--epochs` target, and uploads checkpoints back — re-dispatching
continues the chain.

### Time estimate (from the measured smoke throughput — an estimate)

Assumptions: 0.9 img/s at 640×640 / batch 8 on the 4-core Actions
runner (measured); 512×512 scales by (512/640)² ≈ 0.64 → ~1.4 img/s
(compute-bound assumption, optimistic); one chained job ≈ 5.5 usable
hours (340-min timeout); "days" = continuous back-to-back chaining,
24 h/day. Steps/epoch at batch 8 = images ÷ 8 (full data: 14,658).

| option | config | h/epoch | total | jobs | wall |
|---|---|---|---|---|---|
| A | 640px, full 117,266 imgs, 30 ep | ~36 | ~1,086 h | ~198 | ~45 days |
| B | 512px, full data, 24 ep | ~23 | ~556 h | ~101 | ~23 days |
| C ⭐ | 512px, 40k-image subset (~290k instances), 24 ep | ~7.9 | ~190 h | ~35 | ~8 days |
| D | 640px, 40k subset, 24 ep | ~12.3 | ~296 h | ~54 | ~12 days |

**Honest headline: at 640 px on free CPU runners, full-data training
(option A) is not workable.** Option C is the recommended start: a
40k-image subset still carries ~290k instance masks of supervision,
and the chain can extend to more data/epochs later since checkpoints
resume. Unmeasured headroom that could improve throughput 1.2–1.5×:
`num_workers > 0`, batch 16, channels-last — worth a probe before the
full run. Full training has NOT been started (phase-1 scope).

## Files

- `model.py` — architecture, decode, ONNX export wrapper
- `train.py` — dataset (polygon rasterization + augmentation), targets,
  losses, training loop with `--resume` / `--max-steps` probe
- `evaluate.py` — NMS, class-agnostic box + mask mAP@0.5
- `data_prep.py` — COCO download/count/staging + manifest
- `export.py` — best.pt → best.onnx (+ onnxruntime sanity check)
- `predict.py` — inference + layer-split demo outputs

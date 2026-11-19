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

## Smoke run (path verification, not accuracy)

```bash
gh workflow run smoke.yml           # or the Actions tab
```

Runs data prep, trains 2 epochs, **resumes** to 5 total epochs (proves
the chained-job checkpoint pattern), exports ONNX, and builds the
layer-split demo on 5 val images: for each — `original.jpg`, `viz.jpg`
(each object in a distinct colour) and `obj_NN.png` RGBA cutouts
(the separate layers). Artifact: `object-separator-smoke`.

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
continues the chain. `train.py --max-steps 300 --batch 8` is the timing
probe used to size the chain. Measured smoke numbers and the resulting
full-training estimate are recorded in the phase-1 report (and will be
added here once the probe has run on a runner).

## Files

- `model.py` — architecture, decode, ONNX export wrapper
- `train.py` — dataset (polygon rasterization + augmentation), targets,
  losses, training loop with `--resume` / `--max-steps` probe
- `evaluate.py` — NMS, class-agnostic box + mask mAP@0.5
- `data_prep.py` — COCO download/count/staging + manifest
- `export.py` — best.pt → best.onnx (+ onnxruntime sanity check)
- `predict.py` — inference + layer-split demo outputs

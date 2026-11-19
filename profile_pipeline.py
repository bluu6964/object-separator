"""Profile the training step's CPU-side split (data vs targets vs compute).

Usage:
  python profile_pipeline.py [--batches 6] [--steps 4] [--batch 8]

Reports, at the CURRENT model.IMG_SIZE (set via --img, default 512 to
match Colab Option C):
  A1  DataLoader wall time per batch (workers=0 and workers=2) --
      SegDataset.__getitem__ (letterbox + rasterize + augment + pool)
      plus collate.
  A2  build_targets wall time per batch (main-process, per step).
  A3  cProfile hot spots over raw __getitem__ calls.
  B   model forward + compute_loss + backward + optimizer step time
      per step on PRE-BUILT targets (what the GPU would see if the
      data pipeline were free), plus compute_loss alone.
"""
import argparse
import cProfile
import io
import pstats
import time

import torch
from torch.utils.data import DataLoader

import model as model_mod
import train as train_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--phase", default="all",
                    choices=["all", "a1", "a2", "a3", "b"],
                    help="run one phase per process (this VM OOMs when "
                         "DataLoader workers + model share a process)")
    args = ap.parse_args()

    model_mod.set_img_size(args.img)
    train_mod.IMG_SIZE = model_mod.IMG_SIZE
    train_mod.PROTO_SIZE = model_mod.PROTO_SIZE
    torch.manual_seed(42)

    ds = train_mod.SegDataset(args.data, "train", augment=True, seed=42)
    print(f"img={args.img} batch={args.batch} train images={len(ds)}")

    # ---- A1: DataLoader throughput, workers 0 and 2 -------------------
    sample_batches = []
    for workers in ((0, 2) if args.phase in ("all", "a1") else ()):
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=workers, collate_fn=train_mod.collate)
        times = []
        for i, batch in enumerate(loader):
            if i == 0:
                t0 = time.perf_counter()  # skip first (warmup) batch
                continue
            times.append(time.perf_counter() - t0)
            if workers == 0:
                sample_batches.append(batch)
            t0 = time.perf_counter()
            if len(times) >= args.batches:
                break
        avg = sum(times) / len(times)
        print(f"A1 dataloader workers={workers}: {avg:.3f} s/batch "
              f"({args.batch / avg:.2f} img/s) over {len(times)} batches")

    if args.phase in ("a2", "b") and not sample_batches:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, collate_fn=train_mod.collate)
        for i, batch in enumerate(loader):
            sample_batches.append(batch)
            if len(sample_batches) >= 3:
                break

    # ---- A2: build_targets per batch ----------------------------------
    if args.phase in ("all", "a2"):
        t0 = time.perf_counter()
        for _, boxes, _ in sample_batches:
            train_mod.build_targets(boxes, torch.device("cpu"))
        dt = (time.perf_counter() - t0) / len(sample_batches)
        print(f"A2 build_targets (incl. mask_cells): {dt:.4f} s/batch")

    # ---- A3: cProfile over raw __getitem__ ----------------------------
    if args.phase in ("all", "a3"):
        pr = cProfile.Profile()
        pr.enable()
        for i in range(20):
            ds[i % len(ds)]
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
        print("A3 __getitem__ hot spots (20 calls):")
        print("\n".join(s.getvalue().splitlines()[4:24]))

    # ---- B: fwd + loss + bwd on pre-built targets --------------------
    if args.phase not in ("all", "b"):
        return
    imgs, boxes, masks = sample_batches[0]
    targets, mask_cells = train_mod.build_targets(boxes, torch.device("cpu"))
    model = model_mod.ObjectSeparator()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    # loss alone (no backward): precomputed mask cells vs per-step search
    for tag, mc in (("mask_cells precomputed", mask_cells),
                    ("mask_cells searched (old)", None)):
        t0 = time.perf_counter()
        for _ in range(args.steps):
            outs, protos = model(imgs)
            train_mod.compute_loss(outs, protos, targets, boxes, masks,
                                   mask_cells=mc)
        dt_loss = (time.perf_counter() - t0) / args.steps
        print(f"B  fwd+loss (no bwd, {tag}): {dt_loss:.3f} s/step")
    # full step (new path)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        outs, protos = model(imgs)
        loss = train_mod.compute_loss(outs, protos, targets, boxes, masks,
                                      mask_cells=mask_cells)[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    dt_step = (time.perf_counter() - t0) / args.steps
    print(f"B  fwd+loss+bwd+opt (new path):  {dt_step:.3f} s/step "
          f"({args.batch / dt_step:.2f} img/s)")


if __name__ == "__main__":
    main()

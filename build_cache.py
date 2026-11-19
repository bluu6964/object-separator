"""Build the rasterized-instance cache for a staged dataset.

train.py's SegDataset otherwise re-rasterizes every instance's
polygons (PIL draw + a full-mask emptiness sum per instance) on EVERY
epoch -- 24x redundant work that starves the GPU on 2-vCPU Colab.
This script runs train.rasterize_instances ONCE per image and stores
the pre-augmentation result next to the labels:

  <data>/cache-<IMG_SIZE>/<split>/<stem>.npz
      boxes    float64 [M,4]  letterboxed xyxy (exact values)
      masks    uint8   [M, IMG_SIZE*IMG_SIZE//8]  bitpacked 0/1 masks
      img_size int

Augmentation is NOT baked in: the dataset loads these and applies the
same random flip/jitter/brightness path as before, so epochs still
see different views -- only the polygon drawing is saved. Entries are
skipped if present (idempotent); the dataset falls back to live
rasterization for any image without a valid entry.

Usage:
  python build_cache.py --data data --img 512 [--workers 4]
                        [--splits train val]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

import model as model_mod
import train as train_mod

_IMG = 512


def _init(img):
    global _IMG
    _IMG = img
    model_mod.set_img_size(img)
    train_mod.IMG_SIZE = model_mod.IMG_SIZE
    train_mod.PROTO_SIZE = model_mod.PROTO_SIZE


def _build_one(task):
    root, split, name = task
    stem = os.path.splitext(name)[0]
    out_path = os.path.join(root, f"cache-{_IMG}", split, stem + ".npz")
    if os.path.exists(out_path):
        return (name, -1, 0)  # already cached
    with open(os.path.join(root, "labels", split, stem + ".json")) as f:
        label = json.load(f)
    with Image.open(os.path.join(root, "images", split, name)) as im:
        w, h = im.size
    boxes, mask_imgs = train_mod.rasterize_instances(label, w, h)
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:  # file handle: np.savez* would append
        np.savez_compressed(             # ".npz" to a bare path
            fh,
            boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            masks=train_mod.pack_mask_imgs(mask_imgs),
            img_size=np.int64(_IMG))
    os.replace(tmp, out_path)
    return (name, len(boxes), os.path.getsize(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    _init(args.img)
    total_files = 0
    total_bytes = 0
    total_inst = 0
    for split in args.splits:
        img_dir = os.path.join(args.data, "images", split)
        files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        out_dir = os.path.join(args.data, f"cache-{args.img}", split)
        os.makedirs(out_dir, exist_ok=True)
        tasks = [(args.data, split, f) for f in files]
        done = skipped = 0
        with Pool(args.workers, initializer=_init,
                  initargs=(args.img,)) as pool:
            for name, n_inst, n_bytes in pool.imap_unordered(
                    _build_one, tasks, chunksize=32):
                if n_inst < 0:
                    skipped += 1
                else:
                    done += 1
                    total_inst += n_inst
                    total_bytes += n_bytes
                if (done + skipped) % 2000 == 0:
                    print(f"  {split}: {done + skipped}/{len(tasks)}",
                          flush=True)
        n_all = sum(1 for f in os.listdir(out_dir) if f.endswith(".npz"))
        size_all = sum(os.path.getsize(os.path.join(out_dir, f))
                       for f in os.listdir(out_dir) if f.endswith(".npz"))
        total_files += n_all
        print(f"{split}: {n_all} cache entries "
              f"({size_all / 1e6:.1f} MB total, "
              f"{size_all / max(1, n_all) / 1024:.1f} KB/image avg); "
              f"built {done}, skipped {skipped}", flush=True)
    print(f"Cache for img={args.img} under "
          f"{os.path.join(args.data, f'cache-{args.img}')}: "
          f"{total_files} images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

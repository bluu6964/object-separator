"""Prepare COCO 2017 instance-segmentation data, class-agnostic.

Every annotation's category is IGNORED -- each usable instance is just
"an object". Usable = iscrowd == 0 and polygon segmentation present
(COCO crowd regions are RLE and are excluded by design).

Layout produced (mirrors the phone-detector repo):
  data/images/{train,val}/*.jpg
  data/labels/{train,val}/*.json   {"file","w","h",
      "instances":[{"bbox":[x,y,w,h], "poly":[[x1,y1,...], ...]}, ...]}
  data/manifest.json               full-corpus + staged counts

Usage:
  python data_prep.py --counts-only   # verify corpus counts from the JSONs
  python data_prep.py --smoke         # stage 500 train / 100 val images
  python data_prep.py --full          # stage the full corpus
  python data_prep.py --train-limit 40000 --val-limit 2000
                                      # stage a fixed-seed subset (the
                                      # Colab Option C pack)
Subset selection is the same seeded shuffle as --smoke/--full (sorted
image ids, random.Random(42).shuffle), so a subset is always a prefix
of the full staging order.
"""
import argparse
import json
import os
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_BASE = "http://images.cocodataset.org"
SEED = 42
SMOKE_TRAIN, SMOKE_VAL = 500, 100


def ensure_annotations(src_dir):
    train_json = os.path.join(src_dir, "annotations", "instances_train2017.json")
    val_json = os.path.join(src_dir, "annotations", "instances_val2017.json")
    if os.path.exists(train_json) and os.path.exists(val_json):
        return train_json, val_json
    os.makedirs(src_dir, exist_ok=True)
    zip_path = os.path.join(src_dir, "annotations_trainval2017.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {ANN_URL} ...", flush=True)
        urllib.request.urlretrieve(ANN_URL, zip_path)
    print("Extracting annotations ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(src_dir)
    return train_json, val_json


def usable_instances(ann):
    """Return the polygon list if the annotation is usable, else None."""
    if ann.get("iscrowd", 0) != 0:
        return None
    seg = ann.get("segmentation")
    if not isinstance(seg, list):
        return None  # RLE (crowd-style) encoding -- excluded by design
    polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
    return polys or None


def load_split(json_path):
    """Parse one instances_*.json. Returns (images_by_id, instances_by_image,
    stats). Classes are dropped here -- instances keep only bbox + polygons."""
    with open(json_path) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    by_image = {}
    n_inst = 0
    n_crowd = 0
    n_rle = 0
    for ann in coco["annotations"]:
        polys = usable_instances(ann)
        if polys is None:
            if ann.get("iscrowd", 0) != 0:
                n_crowd += 1
            else:
                n_rle += 1
            continue
        by_image.setdefault(ann["image_id"], []).append(
            {"bbox": [float(v) for v in ann["bbox"]], "poly": polys})
        n_inst += 1
    stats = {
        "images_total": len(images),
        "images_with_instances": len(by_image),
        "instances_usable": n_inst,
        "instances_skipped_crowd": n_crowd,
        "instances_skipped_rle": n_rle,
    }
    return images, by_image, stats


def download_one(args):
    url, dest = args
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        return os.path.getsize(dest) > 0
    except Exception:  # noqa: BLE001
        return False


def stage_split(split, images, by_image, out_dir, limit, workers=16):
    ids = sorted(by_image.keys())
    rng = random.Random(SEED)
    rng.shuffle(ids)
    if limit:
        ids = ids[:limit]
    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    tasks = []
    for iid in ids:
        im = images[iid]
        dest = os.path.join(img_dir, im["file_name"])
        tasks.append((f"{IMG_BASE}/{split}2017/{im['file_name']}", dest))
        label = {"file": im["file_name"], "w": im["width"], "h": im["height"],
                 "instances": by_image[iid]}
        stem = os.path.splitext(im["file_name"])[0]
        with open(os.path.join(lbl_dir, stem + ".json"), "w") as f:
            json.dump(label, f)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = sum(ex.map(download_one, tasks))
    n_inst = sum(len(by_image[i]) for i in ids)
    print(f"{split}: staged {ok}/{len(ids)} images, {n_inst} instances", flush=True)
    return {"images": ok, "instances": n_inst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_src")
    ap.add_argument("--out", default="data")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--train-limit", type=int, default=None,
                    help="stage only the first N train images (seeded "
                         "shuffle order); combine with --val-limit")
    ap.add_argument("--val-limit", type=int, default=None,
                    help="stage only the first N val images")
    ap.add_argument("--counts-only", action="store_true")
    args = ap.parse_args()

    train_json, val_json = ensure_annotations(args.src)
    manifest = {"seed": SEED, "full": {}, "staged": None, "smoke": bool(args.smoke)}
    splits = {}
    for split, path in (("train", train_json), ("val", val_json)):
        print(f"Parsing {path} ...", flush=True)
        images, by_image, stats = load_split(path)
        splits[split] = (images, by_image)
        manifest["full"][split] = stats
        print(f"{split}: {stats}", flush=True)

    if args.counts_only:
        print(json.dumps(manifest["full"], indent=2))
        return
    if not (args.smoke or args.full
            or args.train_limit is not None or args.val_limit is not None):
        print("Pass --smoke, --full or --train-limit/--val-limit "
              "(or --counts-only).", file=sys.stderr)
        return

    staged = {}
    if args.smoke:
        staged["train"] = stage_split("train", *splits["train"], args.out, SMOKE_TRAIN)
        staged["val"] = stage_split("val", *splits["val"], args.out, SMOKE_VAL)
    elif args.full:
        staged["train"] = stage_split("train", *splits["train"], args.out, None)
        staged["val"] = stage_split("val", *splits["val"], args.out, None)
    else:
        staged["train"] = stage_split("train", *splits["train"], args.out,
                                       args.train_limit)
        staged["val"] = stage_split("val", *splits["val"], args.out,
                                     args.val_limit)
        manifest["subset"] = {"train_limit": args.train_limit,
                              "val_limit": args.val_limit, "seed": SEED}
    manifest["staged"] = staged
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()

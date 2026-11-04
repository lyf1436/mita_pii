"""Prepare the internal four-task photo set (COCO-style boxes) as detection data: plate boxes, not masks.

Usage:
    python -m data_pipeline.coco_to_detection --json data/raw/train_4task.json --save_dir data/v0_0_1
    python -m data_pipeline.coco_to_detection --json data/raw/train_4task.json --save_dir data/v0_0_1 \
        --source geico --test_fraction 0.2 --overwrite

Assumption: the same annotation file as coco_to_segmentation, with the same meaning.

Expectation: a train part and a test part in the data version folder, in the detection format
(data_pipeline/formats.py): each image with its size and its plate boxes, in pixels of the image as the trainer
decodes it. Images that cannot be read, or whose size does not fit their annotations, are left out and listed.

Rationale:
- The plates, the images kept and the train/test folders are chosen exactly as in coco_to_segmentation, so the
  two formats of one source hold the same images on the same side of the split.
- Boxes are kept as they are, not rounded, so object sizes can be measured without the outward rounding that
  masks need.
- Boxes annotated at another scale of the same image are rescaled to the image, so every box is in the pixels a
  model will see."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.coco_to_segmentation import build_items, plate_category_ids, same_shape_ratio, test_subsets
from data_pipeline.formats import DETECTION_CLASSES, DETECTION_FEATURES, split_names
from data_pipeline.hf_save import save_generated

NUM_PROCESSES = 8
PLATE = DETECTION_CLASSES.index("plate")


def generate(items, source):
    for item in items:
        raw = Path(item["path"]).read_bytes()       # stored as-is: no re-encoding of the image
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)   # orientation tag applied
        if img is None:
            print(f"skipped {item['path']}: not a readable image")
            continue
        H, W = img.shape[:2]
        if not same_shape_ratio(item["size"], (H, W)):
            print(f"skipped {item['path']}: annotated as {item['size'][0]}x{item['size'][1]}, decodes as {H}x{W}")
            continue
        sy, sx = H / item["size"][0], W / item["size"][1]     # 1 unless annotated at another scale
        yield dict(image_id=item["image_id"], source=source, subset=item["subset"],
                   image={"bytes": raw, "path": Path(item["path"]).name}, width=W, height=H,
                   objects=dict(bbox=[[x * sx, y * sy, w * sx, h * sy] for x, y, w, h in item["boxes"]],
                                category=[PLATE] * len(item["boxes"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, type=Path, help="the annotation file")
    ap.add_argument("--save_dir", required=True, type=Path, help="data version directory, e.g. data/v0_0_1")
    ap.add_argument("--source", help="dataset name prefix (default: the annotation file's name)")
    ap.add_argument("--test_fraction", type=float, default=0.1,
                    help="share of photo folders held out for testing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source = args.source or args.json.stem
    data = json.loads(args.json.read_text())
    items = build_items(data, plate_category_ids(data["categories"]))
    print(f"{len(items)} images, {sum(len(it['boxes']) for it in items)} plate boxes on "
          f"{sum(1 for it in items if it['boxes'])} of them")

    test = test_subsets(items, args.test_fraction)
    splits = dict(zip(split_names(source, "detection"), (
        [it for it in items if it["subset"] not in test],
        [it for it in items if it["subset"] in test],
    )))
    n_folders = len({it["subset"] for it in items})
    for name, split_items in splits.items():
        if not split_items:
            raise ValueError(f"{name} would be empty: {n_folders} photo folder(s) and --test_fraction "
                             f"{args.test_fraction} put {len(test)} of them in the test split")
        out = args.save_dir / name
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"{out} exists (pass --overwrite to replace it)")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for name, split_items in splits.items():
        out = args.save_dir / name
        shutil.rmtree(out, ignore_errors=True)
        ds = save_generated(generate, DETECTION_FEATURES, {"items": split_items, "source": source}, out,
                            num_proc=min(NUM_PROCESSES, len(split_items)))
        skipped = f", {len(split_items) - len(ds)} skipped (listed above)" if len(ds) < len(split_items) else ""
        print(f"{out}: {len(ds)} images, {sum(len(o['bbox']) for o in ds['objects'])} boxes in "
              f"{len({it['subset'] for it in split_items})} folder(s){skipped}")


if __name__ == "__main__":
    main()

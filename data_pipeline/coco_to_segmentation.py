"""Prepare the internal four-task photo set (COCO-style boxes) as segmentation data.

Usage:
    python -m data_pipeline.coco_to_segmentation --json data/raw/train_4task.json --save_dir data/v0_0_1
    python -m data_pipeline.coco_to_segmentation --json data/raw/train_4task.json --save_dir data/v0_0_1 \
        --source geico --test_fraction 0.2 --overwrite

Assumption: the file labels four tasks (licence plate, odometer, VIN, expiry sticker) as boxes, the licence-plate
categories name the issuing US state, categories named NOT_<task> mark what the task's object is not, the boxes
were drawn on the photos as a viewer shows them (turned upright by their orientation tag), and the `file_name`
paths are readable from wherever this runs.

Expectation: a train part and a test part in the data version folder, in the same segmentation format as the
other sources, so they can be concatenated for training (see trainer/common.py `load_merged_dataset`). Every
image comes with a mask that marks each pixel as background or plate. Images that cannot be read, or whose size
does not fit their annotations, are left out and listed.

Rationale:
- Only the licence-plate task is kept, and its state categories collapse into one class: the state is an OCR
  label for one and the same object, and the other three tasks have no class in the segmentation format. Classes
  are not added here, because a class list only grows by appending and adding one would change what the labels of
  the other sources mean.
- Boxes are filled in as masks because everything inside a box is to be blurred, and their edges are rounded
  outwards, so no edge pixel of a plate is left unmarked.
- Images of the other tasks are kept with an all-background mask: they are photos of the same vehicles with no
  plate in them, which is what the model should leave alone.
- Whole photo folders are held out for testing, so several shots of one subject never straddle the split.
- Images are stored byte for byte and masks are kept at the image's own resolution, so the labels do not depend
  on how training later crops or scales the images.
- An image's size is measured with its orientation tag applied, as the trainer decodes it. Phone photos are often
  stored sideways with a tag that turns them upright; measured without the tag, the mask would come out sideways.
- A mask is resized to its image only when the two differ in scale alone. Any other mismatch means the boxes do
  not belong to the pixels as stored, and resizing would move them off the plates."""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.formats import SEGMENTATION_CLASSES, SEGMENTATION_FEATURES, split_names
from data_pipeline.hf_save import save_generated

PLATE = SEGMENTATION_CLASSES.index("plate")
PLATE_SUPERCATEGORY = "license_plate"
NEGATIVE_PREFIX = "NOT_"              # e.g. NOT_LP: a region that looks like a plate but is not one
SEED = 0


def plate_category_ids(categories):
    """The ids of the categories that are a licence plate."""
    ids = {c["id"] for c in categories
           if c["supercategory"] == PLATE_SUPERCATEGORY and not c["name"].startswith(NEGATIVE_PREFIX)}
    if not ids:
        raise ValueError(f"no {PLATE_SUPERCATEGORY!r} categories without the {NEGATIVE_PREFIX!r} prefix; the "
                         f"file has the supercategories {sorted({c['supercategory'] for c in categories})}")
    return ids


def build_items(data, plate_ids):
    """One entry per image: where it is, the folder that groups it, its annotated size, and its plate boxes."""
    boxes = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] in plate_ids:
            boxes[str(ann["image_id"])].append(ann["bbox"])
    return [dict(path=img["file_name"], image_id=str(img["id"]), subset=str(Path(img["file_name"]).parent),
                 size=(img["height"], img["width"]), boxes=boxes.get(str(img["id"]), []))
            for img in data["images"]]


def rasterise(boxes, size):
    """Mark every plate box over an image of the annotated size, rounding box edges outwards."""
    mask = np.zeros(tuple(size), np.uint8)
    for x, y, w, h in boxes:
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = int(np.ceil(x + w)), int(np.ceil(y + h))
        mask[max(y1, 0):y2, max(x1, 0):x2] = PLATE
    return mask


def same_shape_ratio(a, b):
    """Whether two (height, width) sizes differ in scale alone, allowing a pixel of rounding."""
    return abs(a[0] * b[1] / a[1] - b[0]) <= 1


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
        mask = rasterise(item["boxes"], item["size"])
        if mask.shape != (H, W):                    # annotated at another scale: follow the image
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        ok, png = cv2.imencode(".png", mask)
        assert ok
        yield dict(image_id=item["image_id"], source=source, subset=item["subset"],
                   image={"bytes": raw, "path": Path(item["path"]).name},
                   mask={"bytes": png.tobytes(), "path": None})


def test_subsets(items, test_fraction):
    """The folders that form the test split: the same ones on every run."""
    subsets = sorted({it["subset"] for it in items})
    random.Random(SEED).shuffle(subsets)
    return set(subsets[:round(len(subsets) * test_fraction)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, type=Path, help="the annotation file")
    ap.add_argument("--save_dir", required=True, type=Path, help="data version directory, e.g. data/v0_0_1")
    ap.add_argument("--source", help="dataset name prefix (default: the annotation file's name)")
    ap.add_argument("--test_fraction", type=float, default=0.1,
                    help="share of photo folders held out for testing")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source = args.source or args.json.stem
    data = json.loads(args.json.read_text())
    items = build_items(data, plate_category_ids(data["categories"]))
    print(f"{len(items)} images, {sum(len(it['boxes']) for it in items)} plate boxes on "
          f"{sum(1 for it in items if it['boxes'])} of them")

    test = test_subsets(items, args.test_fraction)
    splits = dict(zip(split_names(source, "segmentation"), (
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
        ds = save_generated(generate, SEGMENTATION_FEATURES, {"items": split_items, "source": source}, out,
                            num_proc=min(args.num_proc, len(split_items)))
        skipped = f", {len(split_items) - len(ds)} skipped (listed above)" if len(ds) < len(split_items) else ""
        print(f"{out}: {len(ds)} images in {len({it['subset'] for it in split_items})} folder(s){skipped}")


if __name__ == "__main__":
    main()

"""Prepare a COCO-style box annotation file (e.g. the internal four-task photo set) as detection data: boxes, not masks.

Usage:
    python -m data_pipeline.coco_to_detection --json data/raw/train_4task.json --save_dir data/v0_0_1
    python -m data_pipeline.coco_to_detection --json data/raw/train_4task.json --save_dir data/v0_0_1 \
        --source geico --label_rename '{"license_plate": "plate", "<other supercategory>": null}' \
        --test_fraction 0.2 --num_proc 16 --overwrite

Assumption: every category has a supercategory that names the kind of object it is (e.g. the licence-plate
categories name the issuing US state and share the supercategory `license_plate`), categories whose name starts
with the negative prefix (default NOT_) mark what an object is not, the boxes were drawn on the photos as a viewer
shows them (turned upright by their orientation tag), and the `file_name` paths are readable from wherever this runs.

Expectation: a train part and a test part in the data version folder, `<source>_detection_train` and
`<source>_detection_test`: each image with its size and its boxes, in pixels of the image as the trainer decodes
it, each box with its label's name. Images that cannot be read, or whose size does not fit their annotations, are
left out and listed.

Columns: image_id, source, subset (the photo folder), image (stored bytes), width, height,
objects {bbox: [x, y, width, height], label: name}.

Labels:
- A box's label is its category's supercategory, renamed by --label_rename if it is listed there.
- --label_rename is a JSON object {supercategory: new name}. Several supercategories may get the same new name:
  they become one label. A new name of null drops the supercategory's boxes.

Rationale:
- Labels come from the file and the arguments only, and are stored as names, so the script needs no list of
  classes and the data says what each box is without a lookup table.
- Negative categories are left out: they share the supercategory of the object they are mistaken for.
- With the same file and arguments as coco_to_segmentation, the same images land on the same side of the split
  (whole photo folders, chosen with the same seed), so the two formats of one source can be compared.
- Boxes are kept as they are, not rounded, so object sizes can be measured without the outward rounding that
  masks need.
- Boxes annotated at another scale of the same image are rescaled to the image, so every box is in the pixels a
  model will see.
- Images are read and checked in --num_proc worker processes, each taking an equal share of the images."""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from datasets import Features, Image, Sequence, Value

from data_pipeline.hf_save import save_generated

SEED = 0

FEATURES = Features({
    "image_id": Value("string"),            # the annotation file's image id
    "source": Value("string"),
    "subset": Value("string"),              # photo folder; used for group-wise splits
    "image": Image(),
    "width": Value("int32"),                # of the image as decoded, orientation tag applied
    "height": Value("int32"),
    "objects": Sequence({
        "bbox": Sequence(Value("float32"), length=4),   # x, y, width, height in pixels of the decoded image
        "label": Value("string"),
    }),
})


def category_labels(categories, label_rename, negative_prefix):
    """Category id -> label, for the categories whose boxes are kept."""
    supercategories = {c["supercategory"] for c in categories}
    unknown = set(label_rename) - supercategories
    if unknown:
        raise ValueError(f"--label_rename names {sorted(unknown)}, which are not supercategories of the file; "
                         f"it has {sorted(supercategories)}")
    labels = {}
    for c in categories:
        if negative_prefix and c["name"].startswith(negative_prefix):
            continue
        label = label_rename.get(c["supercategory"], c["supercategory"])
        if label is not None:
            labels[c["id"]] = label
    return labels


def build_items(data, labels):
    """One entry per image: where it is, the folder that groups it, its annotated size, and its kept boxes."""
    boxes = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] in labels:
            boxes[str(ann["image_id"])].append((ann["bbox"], labels[ann["category_id"]]))
    return [dict(path=img["file_name"], image_id=str(img["id"]), subset=str(Path(img["file_name"]).parent),
                 size=(img["height"], img["width"]), boxes=boxes.get(str(img["id"]), []))
            for img in data["images"]]


def same_shape_ratio(a, b):
    """Whether two (height, width) sizes differ in scale alone, allowing a pixel of rounding."""
    return abs(a[0] * b[1] / a[1] - b[0]) <= 1


def generate(items, source):
    for item in items:
        raw = Path(item["path"]).read_bytes()       # stored as-is: no re-encoding of the image
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)   # orientation tag applied
        if img is None:
            print(f"skipped {item['path']}: not a readable image", flush=True)
            continue
        H, W = img.shape[:2]
        if not same_shape_ratio(item["size"], (H, W)):
            print(f"skipped {item['path']}: annotated as {item['size'][0]}x{item['size'][1]}, decodes as {H}x{W}",
                  flush=True)
            continue
        sy, sx = H / item["size"][0], W / item["size"][1]     # 1 unless annotated at another scale
        yield dict(image_id=item["image_id"], source=source, subset=item["subset"],
                   image={"bytes": raw, "path": Path(item["path"]).name}, width=W, height=H,
                   objects=dict(bbox=[[x * sx, y * sy, w * sx, h * sy] for (x, y, w, h), _ in item["boxes"]],
                                label=[label for _, label in item["boxes"]]))


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
    ap.add_argument("--label_rename", type=json.loads, default={},
                    help='JSON object {supercategory: new label or null to drop}, e.g. \'{"license_plate": "plate"}\'')
    ap.add_argument("--negative_prefix", default="NOT_",
                    help="categories whose name starts with this are left out ('' keeps every category)")
    ap.add_argument("--test_fraction", type=float, default=0.1,
                    help="share of photo folders held out for testing")
    ap.add_argument("--num_proc", type=int, default=8, help="worker processes that read the images")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not isinstance(args.label_rename, dict):
        raise ValueError("--label_rename must be a JSON object")

    source = args.source or args.json.stem
    data = json.loads(args.json.read_text())
    labels = category_labels(data["categories"], args.label_rename, args.negative_prefix)
    items = build_items(data, labels)
    print(f"{len(items)} images; labels {sorted(set(labels.values()))}; "
          f"{sum(len(it['boxes']) for it in items)} boxes on {sum(1 for it in items if it['boxes'])} images")

    test = test_subsets(items, args.test_fraction)
    splits = {
        f"{source}_detection_train": [it for it in items if it["subset"] not in test],
        f"{source}_detection_test": [it for it in items if it["subset"] in test],
    }
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
        # each worker process gets an equal share of `items`, the only list argument
        ds = save_generated(generate, FEATURES, {"items": split_items, "source": source}, out,
                            num_proc=min(args.num_proc, len(split_items)))
        skipped = f", {len(split_items) - len(ds)} skipped (listed above)" if len(ds) < len(split_items) else ""
        print(f"{out}: {len(ds)} images, {sum(len(o['bbox']) for o in ds['objects'])} boxes in "
              f"{len({it['subset'] for it in split_items})} folder(s){skipped}")


if __name__ == "__main__":
    main()

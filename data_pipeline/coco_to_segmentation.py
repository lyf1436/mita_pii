"""Prepare a COCO-style box annotation file (e.g. the internal four-task photo set) as segmentation data.

Usage:
    python -m data_pipeline.coco_to_segmentation --json data/raw/train_4task.json --save_dir data/v0_0_1
    python -m data_pipeline.coco_to_segmentation --json data/raw/train_4task.json --save_dir data/v0_0_1 \
        --source geico --label_rename '{"license_plate": "plate", "<other supercategory>": null}' \
        --test_fraction 0.2 --num_proc 16 --overwrite

Assumption: every category has a supercategory that names the kind of object it is (e.g. the licence-plate
categories name the issuing US state and share the supercategory `license_plate`), categories whose name starts
with the negative prefix (default NOT_) mark what an object is not, the boxes were drawn on the photos as a viewer
shows them (turned upright by their orientation tag), and the `file_name` paths are readable from wherever this runs.

Expectation: a train part and a test part in the data version folder, `<source>_segmentation_train` and
`<source>_segmentation_test`. Every image comes with a mask that marks each pixel as background or one of the
labels, and with the names of its mask values. Images that cannot be read, or whose size does not fit their
annotations, are left out and listed.

Columns: image_id, source, subset (the photo folder), image (stored bytes), mask (single-channel PNG, pixel value
v = class label_names[v]), label_names. Other segmentation pipelines save the same columns, so their datasets
concatenate with these; the trainer reads the class names from label_names, so no pipeline needs to know
another's classes.

Labels:
- A box's label is its category's supercategory, renamed by --label_rename if it is listed there.
- --label_rename is a JSON object {supercategory: new name}. Several supercategories may get the same new name:
  they become one label. A new name of null drops the supercategory's boxes; a new name of "background" makes its
  boxes background, which is the same thing in a mask.
- Mask value 0 is "background"; the labels follow in alphabetical order.

Rationale:
- Labels come from the file and the arguments only, so the script needs no list of classes and the person
  preparing the data decides how fine-grained they are.
- Negative categories are left out: they share the supercategory of the object they are mistaken for, and marking
  them would teach the model the opposite of what they mean.
- Boxes are filled in as masks because everything inside a box is to be blurred, and their edges are rounded
  outwards, so no edge pixel of an object is left unmarked. Where boxes overlap, the box listed later wins.
- Images without a kept box are kept with an all-background mask: they are photos of the same scenes with no
  object in them, which is what the model should leave alone.
- Whole photo folders are held out for testing, so several shots of one subject never straddle the split.
- Images are stored byte for byte and masks are kept at the image's own resolution, so the labels do not depend
  on how training later crops or scales the images.
- An image's size is measured with its orientation tag applied, as the trainer decodes it. Phone photos are often
  stored sideways with a tag that turns them upright; measured without the tag, the mask would come out sideways.
- A mask is resized to its image only when the two differ in scale alone. Any other mismatch means the boxes do
  not belong to the pixels as stored, and resizing would move them off the objects.
- Images are read, checked and rasterised in --num_proc worker processes, each taking an equal share of the
  images: decoding photos is the slow part, and the shares are independent."""
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

BACKGROUND = "background"
SEED = 0

FEATURES = Features({
    "image_id": Value("string"),            # the annotation file's image id
    "source": Value("string"),
    "subset": Value("string"),              # photo folder; used for group-wise splits
    "image": Image(),
    "mask": Image(),                        # single-channel PNG of indices into label_names
    "label_names": Sequence(Value("string")),
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
        if label is not None and label != BACKGROUND:
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


def rasterise(boxes, size, value):
    """Mark every box with its label's mask value over an image of the annotated size, rounding box edges outwards."""
    mask = np.zeros(tuple(size), np.uint8)
    for (x, y, w, h), label in boxes:
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = int(np.ceil(x + w)), int(np.ceil(y + h))
        mask[max(y1, 0):y2, max(x1, 0):x2] = value[label]
    return mask


def same_shape_ratio(a, b):
    """Whether two (height, width) sizes differ in scale alone, allowing a pixel of rounding."""
    return abs(a[0] * b[1] / a[1] - b[0]) <= 1


def generate(items, source, label_names):
    value = {name: v for v, name in enumerate(label_names)}
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
        mask = rasterise(item["boxes"], item["size"], value)
        if mask.shape != (H, W):                    # annotated at another scale: follow the image
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        ok, png = cv2.imencode(".png", mask)
        assert ok
        yield dict(image_id=item["image_id"], source=source, subset=item["subset"],
                   image={"bytes": raw, "path": Path(item["path"]).name},
                   mask={"bytes": png.tobytes(), "path": None}, label_names=list(label_names))


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
    ap.add_argument("--num_proc", type=int, default=8, help="worker processes that read and rasterise the images")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not isinstance(args.label_rename, dict):
        raise ValueError("--label_rename must be a JSON object")

    source = args.source or args.json.stem
    data = json.loads(args.json.read_text())
    labels = category_labels(data["categories"], args.label_rename, args.negative_prefix)
    label_names = [BACKGROUND] + sorted(set(labels.values()))
    if len(label_names) > 256:
        raise ValueError(f"{len(label_names)} labels do not fit an 8-bit mask")
    items = build_items(data, labels)
    print(f"{len(items)} images; labels {label_names[1:]}; "
          f"{sum(len(it['boxes']) for it in items)} boxes on {sum(1 for it in items if it['boxes'])} images")

    test = test_subsets(items, args.test_fraction)
    splits = {
        f"{source}_segmentation_train": [it for it in items if it["subset"] not in test],
        f"{source}_segmentation_test": [it for it in items if it["subset"] in test],
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
        # each worker process gets an equal share of `items`, the only list argument (label_names is a tuple so
        # that it is passed whole to every worker)
        ds = save_generated(generate, FEATURES,
                            {"items": split_items, "source": source, "label_names": tuple(label_names)},
                            out, num_proc=min(args.num_proc, len(split_items)))
        skipped = f", {len(split_items) - len(ds)} skipped (listed above)" if len(ds) < len(split_items) else ""
        print(f"{out}: {len(ds)} images in {len({it['subset'] for it in split_items})} folder(s){skipped}")


if __name__ == "__main__":
    main()

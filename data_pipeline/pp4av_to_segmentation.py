"""Prepare PP4AV (street images with face and licence-plate boxes) as segmentation data.

Usage:
    python -m data_pipeline.pp4av_to_segmentation --save_dir data/v1
    python -m data_pipeline.pp4av_to_segmentation --save_dir data/v1 --raw_dir /data/pp4av --overwrite

Assumption: a labelled box is a close enough outline of a face or plate for blurring.

Expectation: a train part and a test part in the data version folder. Every image comes with a mask that marks
each pixel as background, face or plate. The test part consists of whole cities the train part never contains.

Rationale:
- Boxes are filled in as masks because everything inside a box is to be blurred. Box edges are rounded outwards,
  so no edge pixel of a face or plate is left unmarked.
- Whole cities are held out for testing: the images of one city are frames of one driving video and many are
  near-duplicates, so a random split would put almost the same image on both sides and overstate test quality.
- Masks are kept at the original image resolution, so the labels do not depend on how training later crops or
  scales the images.
- The source archives are downloaded and unpacked once and reused on later runs."""
import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.formats import SEGMENTATION_CLASSES, SEGMENTATION_FEATURES, split_names
from data_pipeline.hf_save import save_generated

HF_REPO = "khaclinh/pp4av"
SOURCE = "pp4av"
IMG_EXTS = {".png", ".jpg", ".jpeg"}
# PP4AV label file class -> index in SEGMENTATION_CLASSES
LABEL_TO_CLASS = {0: SEGMENTATION_CLASSES.index("face"), 1: SEGMENTATION_CLASSES.index("plate")}


def fetch_and_extract(raw_dir, zip_name, out_name):
    out_dir = raw_dir / out_name
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    zip_path = raw_dir / "raw" / "data" / zip_name
    if not zip_path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO, f"data/{zip_name}", repo_type="dataset", local_dir=raw_dir / "raw")
    print(f"extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir


def index_images(img_dir, ann_dir):
    """One entry per image: its path, its city, and its label file if it has one."""
    labels = {(p.parent.name, p.stem): p for p in ann_dir.rglob("*.txt")}
    return [(str(p), p.parent.name, str(labels[(p.parent.name, p.stem)]) if (p.parent.name, p.stem) in labels else None)
            for p in sorted(q for q in img_dir.rglob("*") if q.suffix.lower() in IMG_EXTS)]


def rasterise(label_path, W, H):
    """Mark every labelled box with its class over the whole image, rounding box edges outwards. Where boxes
    overlap, the box listed later wins."""
    mask = np.zeros((H, W), np.uint8)
    if label_path is None:
        return mask
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = LABEL_TO_CLASS[int(parts[0])]
        cx, cy, bw, bh = map(float, parts[1:5])
        x1, y1 = int(np.floor((cx - bw / 2) * W)), int(np.floor((cy - bh / 2) * H))
        x2, y2 = int(np.ceil((cx + bw / 2) * W)), int(np.ceil((cy + bh / 2) * H))
        mask[max(y1, 0):y2, max(x1, 0):x2] = cls
    return mask


def generate(items):
    for img_path, subset, label_path in items:
        raw = Path(img_path).read_bytes()        # stored as-is: no re-encoding of the image
        H, W = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED).shape[:2]
        ok, png = cv2.imencode(".png", rasterise(label_path, W, H))
        assert ok
        yield dict(image_id=f"{subset}/{Path(img_path).stem}", source=SOURCE, subset=subset,
                   image={"bytes": raw, "path": Path(img_path).name},
                   mask={"bytes": png.tobytes(), "path": None})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save_dir", required=True, type=Path, help="data version directory, e.g. data/v1")
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw/pp4av"), help="download / extraction cache")
    ap.add_argument("--test_subsets", nargs="+", default=["switzerland", "strasbourg"],
                    help="subsets (cities) that form the test split")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    img_dir = fetch_and_extract(args.raw_dir, "images.zip", "images")
    ann_dir = fetch_and_extract(args.raw_dir, "annotations.zip", "annotations")
    items = index_images(img_dir, ann_dir)
    subsets = sorted({s for _, s, _ in items})
    unknown = set(args.test_subsets) - set(subsets)
    if unknown:
        raise ValueError(f"unknown test subsets {sorted(unknown)}; PP4AV has {subsets}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    splits = dict(zip(split_names(SOURCE, "segmentation"), (
        [it for it in items if it[1] not in args.test_subsets],
        [it for it in items if it[1] in args.test_subsets],
    )))
    for name, split_items in splits.items():
        out = args.save_dir / name
        if out.exists():
            if not args.overwrite:
                raise FileExistsError(f"{out} exists (pass --overwrite to replace it)")
            shutil.rmtree(out)
        ds = save_generated(generate, SEGMENTATION_FEATURES, {"items": split_items}, out,
                            num_proc=min(args.num_proc, len(split_items)))
        print(f"{out}: {len(ds)} images from {sorted({s for _, s, _ in split_items})}")


if __name__ == "__main__":
    main()

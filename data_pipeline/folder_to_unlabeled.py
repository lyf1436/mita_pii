"""Prepare any folder of images as unlabeled data, for trying a model on images it was neither trained nor tested on.

Usage:
    python -m data_pipeline.folder_to_unlabeled --input_dir /path/to/photos --save_dir data/v0_0_1
    python -m data_pipeline.folder_to_unlabeled --input_dir /path/to/photos --save_dir data/v0_0_1 --source dashcam

Assumption: the images can sit in any number of nested folders, in mixed image formats, next to files that are
not images.

Expectation: one dataset holding every readable image under the folder, each remembering its path inside the
folder. Files that cannot be read as images are skipped and listed instead of stopping the run.

Rationale: images are stored byte for byte, so a demo shows exactly the original files. Unlabeled data becomes a
single dataset rather than a train and test pair, because without labels it can be neither trained on nor
scored."""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.formats import UNLABELED_FEATURES
from data_pipeline.hf_save import save_generated

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(input_dir):
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def generate(paths, input_dir, source):
    for p in map(Path, paths):
        raw = p.read_bytes()
        if cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8) is None:
            continue                                     # not a decodable image; reported by main()
        rel = p.relative_to(input_dir)
        yield dict(image_id=rel.as_posix(), source=source,
                   subset="" if rel.parent == Path(".") else rel.parent.as_posix(),
                   image={"bytes": raw, "path": p.name})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, type=Path, help="folder searched recursively for images")
    ap.add_argument("--save_dir", required=True, type=Path, help="data version directory, e.g. data/v0_0_1")
    ap.add_argument("--source", help="dataset name prefix (default: the input folder's name)")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    source = args.source or input_dir.name
    paths = [str(p) for p in find_images(input_dir)]
    if not paths:
        raise FileNotFoundError(f"no images ({', '.join(sorted(IMG_EXTS))}) under {input_dir}")

    out = args.save_dir / f"{source}_unlabeled"
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists (pass --overwrite to replace it)")
        shutil.rmtree(out)
    ds = save_generated(generate, UNLABELED_FEATURES, {"paths": paths, "input_dir": input_dir, "source": source},
                        out, num_proc=min(args.num_proc, len(paths)))

    skipped = sorted(set(paths) - {str(input_dir / i) for i in ds["image_id"]})
    print(f"{out}: {len(ds)} images from {len(set(ds['subset']))} folder(s)")
    if skipped:
        print(f"skipped {len(skipped)} file(s) that could not be decoded:\n  " + "\n  ".join(skipped[:20]) +
              ("\n  ..." if len(skipped) > 20 else ""))


if __name__ == "__main__":
    main()

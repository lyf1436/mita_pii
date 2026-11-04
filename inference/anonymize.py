"""Blur faces and licence plates in an image file or a folder of images with a trained segmentation model.

Usage:
    python -m inference.anonymize --weights model_weights/<model id> --input photo.jpg --output_dir out
    python -m inference.anonymize --weights model_weights/<model id> --input /path/to/photos --output_dir out \
        --threshold 0.3 --save_masks

Assumption: the weights folder was written by the segmentation trainer.

Expectation: one blurred copy of every readable image, under the same relative path in the output folder, and
optionally the blurred region of each as a black-and-white PNG next to it. Files that are not readable images are
listed and left out.

Rationale:
- The whole image is resized so its longer side is the model input size, padded to a square and labelled in one
  pass, as validation does, so a script result matches the scores in training.
- Everything the model gives a probability of face or plate above the threshold is blurred, and the region is
  grown by a few pixels first, so the edges of a face or plate are blurred too.
- Images are read upright (orientation tag applied), as the model was trained on them, and written upright."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data_pipeline.folder_to_unlabeled import IMG_EXTS, find_images
from models.mita_segmentation import MitaSegmentation


@torch.no_grad()
def private_prob(model, img_rgb):
    """RGB uint8 [H, W, 3] -> probability that a pixel is not background, float32 [H, W]."""
    return (1 - model.predict(img_rgb)[0]).cpu().numpy()


def anonymize(img_rgb, prob, threshold=0.5, dilate_px=3, blur_sigma=8):
    """The image with every pixel above the threshold blurred, and the blurred region as a 0/1 mask."""
    mask = (prob > threshold).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), blur_sigma)
    return np.where(mask[..., None] == 1, blurred, img_rgb), mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path, help="model_weights/<model id>")
    ap.add_argument("--input", required=True, type=Path, help="an image file, or a folder searched recursively")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5, help="probability above which a pixel is blurred")
    ap.add_argument("--dilate_px", type=int, default=3, help="grow the blurred region by this many pixels")
    ap.add_argument("--blur_sigma", type=float, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_masks", action="store_true", help="also write each blurred region as <name>_mask.png")
    args = ap.parse_args()

    if args.input.is_file():
        root, paths = args.input.parent, [args.input]
    else:
        root, paths = args.input, find_images(args.input)
    if not paths:
        raise FileNotFoundError(f"no images ({', '.join(sorted(IMG_EXTS))}) at {args.input}")
    model = MitaSegmentation.load(args.weights, device=args.device)
    print(f"{len(paths)} image(s), threshold {args.threshold:g}, device {args.device}")

    skipped = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)                # orientation tag applied
        if img is None:
            skipped.append(path)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        anon, mask = anonymize(img, private_prob(model, img), args.threshold, args.dilate_px, args.blur_sigma)
        out = args.output_dir / path.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(anon, cv2.COLOR_RGB2BGR))
        if args.save_masks:
            cv2.imwrite(str(out.with_name(f"{out.stem}_mask.png")), mask * 255)
        print(f"{out}: {100 * mask.mean():.2f}% of pixels blurred")

    print(f"wrote {len(paths) - len(skipped)} image(s) to {args.output_dir}")
    if skipped:
        print(f"skipped {len(skipped)} file(s) that could not be read:\n  " + "\n  ".join(map(str, skipped)))


if __name__ == "__main__":
    main()

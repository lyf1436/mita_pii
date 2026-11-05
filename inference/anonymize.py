"""Blur faces and licence plates in an image file or a folder of images with a trained segmentation model.

Usage:
    python -m inference.anonymize --weights model_weights/<model id> --input photo.jpg --output_dir out
    python -m inference.anonymize --weights model_weights/<model id> --input /path/to/photos --output_dir out \
        --threshold 0.3 --dilate_px 3 --blur_sigma 8 --save_masks

Assumption: the weights folder was written by the segmentation trainer.

Expectation: one blurred copy of every readable image, under the same relative path in the output folder, and
optionally the blurred region of each as a black-and-white PNG next to it. Files that are not readable images are
listed and left out.

Rationale:
- The whole image is resized so its longer side is the model input size (shrunk if larger, enlarged if smaller),
  padded to a square and labelled in one pass, as validation does, so a script result matches the scores in
  training. Images of any sizes and shapes share each forward pass, up to INFERENCE_BATCH at a time; each is
  resized and padded on its own, so batching only changes speed and memory, not the result.
- Everything the model gives a probability of face or plate above the threshold is blurred, and the region is
  grown a little first, so the edges of a face or plate are blurred too.
- The growth and the blur strength are given in pixels of an image whose longer side is `reference_size`, and
  scale with the image: the same photo at any resolution comes out blurred the same way. In fixed pixels, a
  high-resolution photo would be barely blurred and a thumbnail smeared.
- Images are read upright (orientation tag applied), as the model was trained on them, and written upright."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data_pipeline.folder_to_unlabeled import IMG_EXTS, find_images
from models.mita_segmentation import MitaSegmentation

INFERENCE_BATCH = 32          # images per forward pass
REFERENCE_SIZE = 480          # longer image side at which dilate_px and blur_sigma are measured


def private_probs(model, imgs_rgb, batch_size=INFERENCE_BATCH):
    """A list of RGB uint8 images [H, W, 3] of any sizes -> for each, in order, the probability that a pixel is not
    background, float32 [H, W] at the image's own resolution. The images go through the model `batch_size` at a
    time; within a batch each is resized whole so its longer side is the model input, and padded to a square."""
    for start in range(0, len(imgs_rgb), batch_size):
        for prob in model.predict_batch(imgs_rgb[start:start + batch_size]):
            yield (1 - prob[0]).cpu().numpy()


def private_prob(model, img_rgb):
    """RGB uint8 [H, W, 3] -> probability that a pixel is not background, float32 [H, W]."""
    return next(private_probs(model, [img_rgb]))


def anonymize(img_rgb, prob, threshold=0.5, dilate_px=3, blur_sigma=8, reference_size=REFERENCE_SIZE):
    """The image with every pixel above the threshold blurred, and the blurred region as a 0/1 mask.

    `dilate_px` (growth of the region) and `blur_sigma` are in pixels of the image scaled so its longer side is
    `reference_size`; both scale with the image's own longer side, so the result does not depend on resolution."""
    scale = max(img_rgb.shape[:2]) / reference_size
    mask = (prob > threshold).astype(np.uint8)
    radius = round(dilate_px * scale)
    if radius > 0:
        mask = cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    if not mask.any():
        return img_rgb.copy(), mask
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), blur_sigma * scale) if blur_sigma > 0 else img_rgb
    return np.where(mask[..., None] == 1, blurred, img_rgb), mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path, help="model_weights/<model id>")
    ap.add_argument("--input", required=True, type=Path, help="an image file, or a folder searched recursively")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5, help="probability above which a pixel is blurred")
    ap.add_argument("--dilate_px", type=float, default=3,
                    help=f"grow the blurred region by this many pixels at a longer side of {REFERENCE_SIZE}")
    ap.add_argument("--blur_sigma", type=float, default=8,
                    help=f"Gaussian blur strength in pixels at a longer side of {REFERENCE_SIZE}")
    ap.add_argument("--reference_size", type=int, default=REFERENCE_SIZE,
                    help="longer image side at which --dilate_px and --blur_sigma are measured")
    ap.add_argument("--batch_size", type=int, default=INFERENCE_BATCH, help="images per forward pass")
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
    for start in range(0, len(paths), args.batch_size):
        batch = []
        for path in paths[start:start + args.batch_size]:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)            # orientation tag applied
            if img is None:
                skipped.append(path)
            else:
                batch.append((path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not batch:
            continue
        for (path, img), prob in zip(batch, private_probs(model, [img for _, img in batch], args.batch_size)):
            anon, mask = anonymize(img, prob, args.threshold, args.dilate_px, args.blur_sigma, args.reference_size)
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

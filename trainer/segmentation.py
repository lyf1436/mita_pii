"""Training and validation of a per-pixel segmentation model.

Assumption: the data is in the segmentation format, and the classes other than background are rare and small:
faces and plates cover a fraction of a percent of the pixels.

Expectation:
- Logging, validation and stopping happen at the same shares of training images whatever the batch size or
  number of devices, so runs that differ in those are compared fairly.
- Validation scores what inference would produce: whole validation images, labelled at full resolution.
- The model is saved after every validation, so a run that is interrupted or stopped keeps its latest state.
- A run stops once its validation loss is no longer falling fast enough.

Rationale:
- Training crops are cut so that, once resized to the input size, objects appear at the scale used at inference,
  with some random variation for robustness.
- Some crops are placed around an object: random crops would almost always contain background only.
- The loss adds a Dice term per object class to cross-entropy. Cross-entropy counts pixels, so background
  dominates it; Dice scores each class by overlap, whatever its size.
- The validation loss weighs the Dice term by 1 whatever weight training uses, so runs trained with different
  weights are compared on one scale.
- Stopping looks at the average fall of the validation loss over several validations rather than one comparison,
  so a single noisy validation does not end a run.
- The learning-rate schedule is planned over the full run even when the run stops early, so the possibility of
  stopping does not change how training proceeds before it."""
import json
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import Image as HFImage
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.formats import SEGMENTATION_CLASSES
from models.mita_segmentation import MitaSegmentation
from segm.optim.factory import create_optimizer, create_scheduler
from trainer.common import (LOGS_ROOT, WEIGHTS_ROOT, load_merged_dataset, model_id, train_valid_split)

MODEL_NAME = "mita_segmentation"
cv2.setNumThreads(0)                     # let DataLoader workers parallelise, not OpenCV threads


def decode_row(row):
    """A stored row -> the RGB image and its class mask."""
    img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(row["mask"]["bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask


def undecoded(ds):
    """The dataset with images and masks left as stored bytes, so the loading workers decode them in parallel."""
    return ds.cast_column("image", HFImage(decode=False)).cast_column("mask", HFImage(decode=False))


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, ds, input_size, upscale, tcfg):
        self.ds, self.input_size, self.upscale = undecoded(ds), input_size, upscale
        self.scale_jitter, self.object_crop_prob = tuple(tcfg["scale_jitter"]), tcfg["object_crop_prob"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, mask = decode_row(self.ds[i])
        H, W = img.shape[:2]
        size = self.input_size
        scale = self.upscale * random.uniform(*self.scale_jitter)
        win = min(int(round(size / scale)), H, W)                 # window side in original pixels

        if random.random() < self.object_crop_prob and mask.any():
            n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
            x1, y1, bw, bh = stats[random.randrange(1, n), :4]     # component 0 is the background
            x2, y2 = x1 + bw, y1 + bh
            x0 = random.uniform(max(0, x2 - win), max(0, min(x1, W - win)))
            y0 = random.uniform(max(0, y2 - win), max(0, min(y1, H - win)))
        else:
            x0, y0 = random.uniform(0, W - win), random.uniform(0, H - win)
        x0, y0 = int(min(x0, W - win)), int(min(y0, H - win))

        crop = cv2.resize(img[y0:y0 + win, x0:x0 + win], (size, size), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(mask[y0:y0 + win, x0:x0 + win], (size, size), interpolation=cv2.INTER_NEAREST)
        if random.random() < 0.5:
            crop, crop_mask = crop[:, ::-1], crop_mask[:, ::-1]
        alpha, beta = random.uniform(0.8, 1.2), random.uniform(-20, 20)
        crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return (torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(crop_mask)).long())


def dice_loss(logits, target, cls):
    """Soft Dice loss of one class: 0 = predicted region matches the labelled one, 1 = no overlap."""
    p = logits.softmax(1)[:, cls]
    t = (target == cls).float()
    inter = (p * t).sum()
    return 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)


@torch.no_grad()
def evaluate(model, ds, batch_size=16, accelerator=None):
    """Label every image in the dataset at full resolution and score the result: validation loss, and Dice and IoU
    per class, counted over all pixels of all images. Across several devices each takes a share of the images and
    the counts are combined.

    The loss weighs the Dice term by 1 whatever weight training uses, so it keeps one scale across runs."""
    n_cls = len(model.classes)
    if accelerator is not None and accelerator.num_processes > 1:
        ds = ds.shard(accelerator.num_processes, accelerator.process_index, contiguous=True)
    device = model.mean.device
    # rows: argmax intersection, argmax predicted, labelled, soft intersection, soft predicted; last column of
    # row 0 holds the summed cross-entropy
    sums = torch.zeros(5, n_cls + 1, dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    for row in undecoded(ds):
        img, mask = decode_row(row)
        prob = model.predict(img, batch_size=batch_size)
        pred = prob.argmax(0)
        target = torch.from_numpy(mask).to(device).long()
        onehot = F.one_hot(target, n_cls).permute(2, 0, 1).float()
        sums[0, :n_cls] += torch.bincount(target[pred == target], minlength=n_cls)[:n_cls].double()
        sums[1, :n_cls] += torch.bincount(pred.flatten(), minlength=n_cls)[:n_cls].double()
        sums[2, :n_cls] += torch.bincount(target.flatten(), minlength=n_cls)[:n_cls].double()
        sums[3, :n_cls] += (prob * onehot).sum((1, 2)).double()
        sums[4, :n_cls] += prob.sum((1, 2)).double()
        sums[0, n_cls] += -torch.log(prob.gather(0, target[None]).clamp_min(1e-8)).sum().double()
    model.train(was_training)
    if accelerator is not None:
        sums = accelerator.reduce(sums, reduction="sum")
    sums = sums.cpu().numpy()
    ce_sum, sums = sums[0, n_cls], sums[:, :n_cls]
    inter, pred_n, true_n, soft_inter, soft_pred = sums
    dice = 2 * inter / np.maximum(pred_n + true_n, 1)
    iou = inter / np.maximum(pred_n + true_n - inter, 1)
    soft_dice_loss = 1 - (2 * soft_inter + 1) / (soft_pred + true_n + 1)      # as dice_loss, over the whole set
    metrics = dict(ce=float(ce_sum / max(true_n.sum(), 1)))
    metrics["loss"] = metrics["ce"] + float(soft_dice_loss[1:].mean())
    for c, name in enumerate(model.classes):
        metrics[f"dice_{name}"], metrics[f"iou_{name}"] = float(dice[c]), float(iou[c])
    metrics["mean_dice"] = float(dice[1:].mean())          # over the non-background classes
    metrics["mean_iou"] = float(iou[1:].mean())
    return metrics


def crosses(step, total, n):
    """Whether this step ends one of n equal parts of the run (every step if the run is shorter than n)."""
    return step * n // total > (step - 1) * n // total


def loss_slope(history, window):
    """Average change of the validation loss per validation over the last few validations (negative = falling),
    or None while there are too few."""
    if window <= 0 or len(history) <= window:
        return None
    return (history[-1] - history[-1 - window]) / window


def build_model(cfg):
    classes = cfg["classification"]["classes"]
    if classes != SEGMENTATION_CLASSES[:len(classes)]:
        raise ValueError(f"[classification] classes {classes} must match the segmentation format's label ids "
                         f"{SEGMENTATION_CLASSES} (data_pipeline/formats.py)")
    t = cfg["training"]
    return MitaSegmentation(backbone=t["backbone"], classes=classes, grid=t["grid"], px_per_patch=t["px_per_patch"],
                            dropout=t["dropout"], drop_path_rate=t["drop_path_rate"])


def save_model(model, out_dir, cfg, state):
    model.save(out_dir)
    shutil.copy(cfg["_path"], Path(out_dir) / "config.toml")
    (Path(out_dir) / "train_state.json").write_text(json.dumps(dict(state, training=cfg["training"]), indent=2))


def train(cfg, log_name=None, weights_dir=None, report=None):
    """Train one model as configured and return its last validation scores and the model.

    Logging and validation happen a fixed number of times over the planned run. After each validation the scores go
    to the callback if one is given (it may end the run by raising), the model is saved if a folder is given, and the
    run stops once the validation loss falls by less than the configured amount per validation, averaged over the
    configured number of validations."""
    if cfg["model_name"] != MODEL_NAME:
        raise ValueError(f"the segmentation trainer trains {MODEL_NAME!r}, config asks for {cfg['model_name']!r}")
    t = cfg["training"]
    accelerator = Accelerator(
        mixed_precision=cfg["accelerate"].get("mixed_precision", "no"),
        # as MiTA-Segmenter's train.py: the encoder's ImageNet classification head gets no gradient
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    torch.manual_seed(t["seed"] + accelerator.process_index)
    random.seed(t["seed"] + accelerator.process_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ds = load_merged_dataset(cfg["dataset_paths"])
    train_ds, valid_ds = train_valid_split(ds, t["valid_fraction"], t.get("valid_subsets", []), t["seed"])

    model = build_model(cfg)
    loader = torch.utils.data.DataLoader(
        CropDataset(train_ds, model.input_size, model.upscale, t), batch_size=t["batch_size"], shuffle=True,
        drop_last=True, num_workers=t["num_workers"], pin_memory=True, persistent_workers=t["num_workers"] > 0,
    )
    loader = accelerator.prepare(loader)                       # sharded across devices
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * t["epochs"]
    opt_args = SimpleNamespace(
        opt=t["opt"], lr=t["lr"], weight_decay=t["weight_decay"], momentum=t["momentum"],
        sched=t["sched"], poly_power=t["poly_power"], poly_step_size=t["poly_step_size"], min_lr=t["min_lr"],
        epochs=t["epochs"], iter_max=total_steps, iter_warmup=0.0,
    )
    optimizer = create_optimizer(opt_args, model.net)
    lr_scheduler = create_scheduler(opt_args, optimizer)      # stepped per update via step_update, not prepared
    model, optimizer = accelerator.prepare(model, optimizer)
    raw_model = accelerator.unwrap_model(model)

    log_name = log_name or model_id(cfg)
    writer = SummaryWriter(str(LOGS_ROOT / log_name)) if accelerator.is_main_process else None
    if writer:
        writer.add_text("config", "```toml\n" + Path(cfg["_path"]).read_text() + "\n```")
        writer.add_text("training", "```json\n" + json.dumps(t, indent=2) + "\n```")
        accelerator.print(f"{len(train_ds)} train / {len(valid_ds)} valid images, {steps_per_epoch} steps per epoch, "
                          f"{total_steps} steps planned on {accelerator.num_processes} device(s); "
                          f"evaluating every ~{total_steps / t['num_evals']:.0f} steps; logs in {LOGS_ROOT / log_name}")

    def batches():
        for epoch in range(t["epochs"]):
            for batch in loader:
                yield epoch, batch

    fg = list(enumerate(raw_model.classes))[1:]
    # x axis of every TensorBoard curve: training images seen (the same for any batch size or device count)
    state, metrics, history = dict(epoch=0, step=0, images=0, evals=0, stopped_early=None), {}, []
    acc = dict(loss=0.0, ce=0.0, **{f"dice_{n}": 0.0 for _, n in fg}, n=0, t=time.perf_counter())
    model.train()
    try:
        for state["epoch"], (ims, masks) in batches():
            with accelerator.autocast():
                logits = model(ims)
            logits = logits.float()
            ce = F.cross_entropy(logits, masks)
            dice = {name: dice_loss(logits, masks, c) for c, name in fg}
            loss = ce + t["dice_weight"] * sum(dice.values()) / len(dice)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            state["step"] += 1
            state["images"] += t["batch_size"] * accelerator.num_processes
            lr_scheduler.step_update(num_updates=state["step"])

            acc["loss"] += loss.item(); acc["ce"] += ce.item(); acc["n"] += 1
            for name, d in dice.items():
                acc[f"dice_{name}"] += d.item()
            if crosses(state["step"], total_steps, t["num_logs"]) and writer:
                n, dt = acc.pop("n"), time.perf_counter() - acc.pop("t")
                for k, v in acc.items():
                    writer.add_scalar(f"train/{k}", v / n, state["images"])
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], state["images"])
                writer.add_scalar("perf/img_per_s", n * t["batch_size"] * accelerator.num_processes / dt, state["images"])
                if torch.cuda.is_available():
                    writer.add_scalar("perf/gpu_mem_gb", torch.cuda.max_memory_allocated() / 1e9, state["images"])
                acc = dict({k: 0.0 for k in acc}, n=0, t=time.perf_counter())

            if not crosses(state["step"], total_steps, t["num_evals"]):
                continue
            metrics = evaluate(raw_model, valid_ds, t["eval_batch_size"], accelerator)
            state["evals"] += 1
            history.append(metrics["loss"])
            slope = loss_slope(history, t["early_stopping_window"])
            if slope is not None and -slope < t["early_stopping_min_decrease"]:
                state["stopped_early"] = (f"validation loss fell {-slope:.4g} per evaluation over the last "
                                          f"{t['early_stopping_window']}, less than {t['early_stopping_min_decrease']}")
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"valid/{k}", v, state["images"])
                if slope is not None:
                    writer.add_scalar("valid/loss_slope", slope, state["images"])
                accelerator.print(f"eval {state['evals']}/{t['num_evals']} (epoch {state['epoch'] + 1}/{t['epochs']})  " +
                                  "  ".join(f"{k} {v:.4f}" for k, v in metrics.items() if k in ("loss", "mean_dice", "mean_iou")) +
                                  ("" if slope is None else f"  loss slope {slope:.4g}"))
            if weights_dir is not None and accelerator.is_main_process:
                save_model(raw_model, weights_dir, cfg, dict(state, valid=metrics))
            if report is not None:
                report(metrics, state["evals"])
            if state["stopped_early"]:
                accelerator.print(f"early stop at step {state['step']}/{total_steps}: {state['stopped_early']}")
                break
    except KeyboardInterrupt:
        accelerator.print("interrupted")
        if weights_dir is not None and accelerator.is_main_process:
            save_model(raw_model, weights_dir, cfg, dict(state, valid=metrics))
        raise
    finally:
        if writer:
            writer.close()
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    return metrics, raw_model


def default_weights_dir(cfg):
    return WEIGHTS_ROOT / model_id(cfg)

"""MiTA-Segmenter for per-pixel classification (e.g. background / face / plate).

Wraps MiTA-Segmenter's `segm` package: a ViT encoder with MiTA attention (ImageNet-pretrained) and a
mask-transformer decoder. Adds what the training notebook established:

- the decoder attention patch that lets it run with any class count (MiTA hard-codes ADE20K's 150)
- the geometry: every input is `grid` x `grid` patches, and images are upscaled so that one patch covers
  `px_per_patch` original pixels
- input normalisation inside `forward`, so callers pass uint8 images
- `predict`: sliding-window inference on a full image, returning class probabilities at its resolution
- `save` / `load` of a weights directory (config.json + model.safetensors)
"""
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file

import segm.model.dec_blocks as dec_blocks
import segm.model.factory as segm_factory
from segm.config import load_config
from segm.model.factory import create_segmenter
from segm.model.utils import merge_windows, sliding_window

NORM_STATS = {   # segm/data/utils.py STATS (importing segm.data would pull in its dataset dependencies)
    "vit": dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    "deit": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}

# --- decoder patch: MiTA's decoder attention has two ADE20K numbers written in as literals, `150` (class
# tokens) and `300` (its width, 3 heads x 100). The same forward pass with those literals replaced by the
# values they stand for; at 150 classes it is bit-identical to MiTA's original (checked in _patch_decoder).
_mita_decoder_forward = dec_blocks.Attention.forward


def _decoder_forward_any_n_cls(self, x):
    b, n, c = x.shape
    n_cls, inner = self.n_cls, self.heads * self.dim_head          # MiTA: 150, 300
    h = width = int((n - n_cls) ** 0.5)
    w = self.proj(x)
    rep = self.pool(w[:, :-n_cls, :].reshape(b, h, width, inner).permute(0, 3, 1, 2)).reshape(b, inner, -1).permute(0, 2, 1)
    w = w.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
    rep = rep.reshape(b, rep.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3)
    rep_delta, attn = self.attention(rep, w, w)
    rep = rep + self.step_rep * rep_delta
    x_delta, _ = self.attention(rep, rep, rep)
    x_delta = attn.transpose(-1, -2) @ x_delta
    x_delta = self.step_x * x_delta
    return self.to_out(rearrange(x_delta, "b h n k -> b n (h k)"))


def _patch_decoder(d_model, n_heads, grid):
    if dec_blocks.Attention.forward is _decoder_forward_any_n_cls:
        return
    # dim_head=100 is fixed inside segm.model.decoder.MaskTransformer
    att = dec_blocks.Attention(d_model, heads=n_heads, dim_head=100).eval()
    att.n_cls = 150
    x = torch.randn(2, grid ** 2 + 150, d_model)
    with torch.no_grad():
        assert torch.equal(_mita_decoder_forward(att, x), _decoder_forward_any_n_cls(att, x)), \
            "decoder patch differs from MiTA's original at 150 classes"
    dec_blocks.Attention.forward = _decoder_forward_any_n_cls


@contextlib.contextmanager
def _no_imagenet_download():
    """create_segmenter always loads ImageNet weights; skip that when trained weights are loaded right after."""
    names = ("load_pretrained", "load_custom_pretrained", "checkpoint_filter_fn")
    saved = {n: getattr(segm_factory, n) for n in names}
    saved_torch_load = torch.load
    segm_factory.load_pretrained = segm_factory.load_custom_pretrained = lambda *a, **k: None
    # vit_base_patch8_384 reads a local file and loads it with strict=True: hand it the model's own weights
    segm_factory.checkpoint_filter_fn = lambda state_dict, model: model.state_dict()
    torch.load = lambda *a, **k: {}
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(segm_factory, n, f)
        torch.load = saved_torch_load


def usable_backbones():
    """Non-distilled ViTs in segm/config.yml: distilled DeiT models add a second extra token, which MiTA
    attention does not accept."""
    return [name for name, m in load_config()["model"].items() if not m.get("distilled", False)]


class MitaSegmentation(nn.Module):
    def __init__(self, backbone, classes, grid=32, px_per_patch=8, dropout=0.0, drop_path_rate=0.1,
                 pretrained=True):
        super().__init__()
        assert grid in (32, 40), "MiTA-Segmenter attention only accepts 32x32 or 40x40 patch grids"
        # The decoder ends in a LayerNorm over each pixel's class scores. With 2 classes it outputs only
        # (+1, -1) or (-1, +1), so no gradient reaches the rest of the network: at least 3 classes are needed.
        assert len(classes) >= 3, "MiTA-Segmenter's decoder needs at least 3 classes (see comment above)"
        segm_cfg = load_config()
        if backbone not in usable_backbones():
            raise ValueError(f"backbone {backbone!r} cannot be used. Choose one of: {', '.join(usable_backbones())}")
        if backbone == "vit_base_patch8_384" and pretrained:
            weights = Path(os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth"))
            if not weights.exists():
                raise FileNotFoundError(f"vit_base_patch8_384 needs its ImageNet weights at {weights}")

        self.config = dict(backbone=backbone, classes=list(classes), grid=grid, px_per_patch=px_per_patch,
                           dropout=dropout, drop_path_rate=drop_path_rate)
        enc = segm_cfg["model"][backbone]
        self.classes = list(classes)
        self.patch_size = enc["patch_size"]
        self.input_size = grid * enc["patch_size"]              # model input side in pixels
        self.upscale = enc["patch_size"] / px_per_patch          # original image -> model resolution

        decoder = dict(segm_cfg["decoder"]["mask_transformer"], name="mask_transformer")
        _patch_decoder(enc["d_model"], decoder["n_heads"], grid)
        segm_model_cfg = dict(
            backbone=backbone, image_size=(self.input_size, self.input_size), patch_size=enc["patch_size"],
            d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
            normalization=enc["normalization"], distilled=False, dropout=dropout, drop_path_rate=drop_path_rate,
            n_cls=len(classes), decoder=decoder,
        )
        with contextlib.nullcontext() if pretrained else _no_imagenet_download():
            self.net = create_segmenter(segm_model_cfg)
        for m in self.net.decoder.modules():                     # tell the patched decoder attention the class count
            if isinstance(m, dec_blocks.Attention):
                m.n_cls = len(classes)

        stats = NORM_STATS[enc["normalization"]]
        self.register_buffer("mean", torch.tensor(stats["mean"]).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("std", torch.tensor(stats["std"]).view(1, 3, 1, 1) * 255, persistent=False)

    def forward(self, images):
        """uint8 (or 0-255 float) [B, 3, input_size, input_size] -> class logits [B, n_cls, input_size, input_size]."""
        return self.net((images.float() - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, img_rgb, batch_size=16, amp_dtype=torch.bfloat16):
        """RGB uint8 [H, W, 3] (numpy) -> class probabilities float32 [n_cls, H, W] at the original resolution.

        Upscales by `self.upscale`, runs the model on sliding windows of the input size (MiTA recipe: stride =
        window), merges them and resizes back."""
        device = self.mean.device
        H, W = img_rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1)[None].to(device).float()
        x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        sh, sw = x.shape[2:]
        pad_h, pad_w = max(0, self.input_size - sh), max(0, self.input_size - sw)
        if pad_h or pad_w:                                       # sliding_window needs at least one full window
            x = F.pad(x, (0, pad_w, 0, pad_h))
        windows = sliding_window(x, False, self.input_size, self.input_size)
        crops = torch.cat(windows.pop("crop"))
        out = []
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            for i in range(0, len(crops), batch_size):
                out.append(self(crops[i:i + batch_size]).float())
        windows["seg_maps"] = torch.cat(out)
        if not (pad_h or pad_w):
            return merge_windows(windows, self.input_size, (H, W))  # softmax over classes, resized to (H, W)
        prob = merge_windows(windows, self.input_size, windows["shape"])[:, :sh, :sw]   # drop the padding
        return F.interpolate(prob[None], (H, W), mode="bilinear", align_corners=False)[0]

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(dict(model="mita_segmentation", **self.config), indent=2))
        save_file({k: v.contiguous() for k, v in self.net.state_dict().items()}, out_dir / "model.safetensors")

    @classmethod
    def load(cls, weights_dir, device="cpu"):
        weights_dir = Path(weights_dir)
        cfg = json.loads((weights_dir / "config.json").read_text())
        assert cfg.pop("model") == "mita_segmentation", f"{weights_dir} does not hold a mita_segmentation model"
        model = cls(**cfg, pretrained=False)
        model.net.load_state_dict(load_file(weights_dir / "model.safetensors", device="cpu"), strict=True)
        return model.to(device).eval()

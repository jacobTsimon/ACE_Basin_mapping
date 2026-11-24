"""
Shape-aware training losses for the saltpan segmentation model.

  BoundaryLoss      - Kervadec et al. 2019 (arXiv:1812.07032). Weights each pixel's
                      foreground probability by the signed distance to the true
                      boundary (negative inside a pan, positive outside).
  HausdorffDTLoss   - distance-transform Hausdorff loss, Karimi & Salcudean 2019
                      (arXiv:1904.10030). Squared error weighted by the distance of
                      each pixel to the true AND the predicted boundary.
  ShapeCompoundLoss - CE + weighted shape terms with an optional epoch ramp.

All losses take (logits, target):
    logits: (B, 2, H, W) raw model output (class 1 = saltpan)
    target: (B, H, W) long tensor with values {0, 1}; pixels equal to
            `ignore_index` (-1) are excluded. Your trainer currently clamps nodata
            to 0 before the loss, which still works (nothing is ignored); to
            exclude nodata properly, pass the un-clamped target and build the CE
            with ignore_index=-1 (see the integration notes).

Distance transforms run on the CPU with scipy once per image per step; at
256x256 and batch 16 this is a few tens of milliseconds. Distance maps are
computed from the *augmented* target, so they stay correct under flips/rotations.

Departures from the original implementations (deliberate, flagged):
  - distances are clipped at d_max pixels and divided by it, so every weight is in
    [-1, 1] (boundary) or [0, 1] (Hausdorff) and the terms are on a CE-like scale;
  - tiles with no true foreground get the maximum distance everywhere, so false
    positives on empty tiles are penalized (the originals leave these at zero);
  - `per_object_interior` (BoundaryLoss) is my own option, not from the paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as _edt
from scipy.ndimage import label as _label
from scipy.ndimage import maximum as _ndi_maximum

_EIGHT = np.ones((3, 3), dtype=int)   # 8-connectivity, matches the metrics code


def signed_distance_map(fg, d_max=20.0, per_object_interior=False, empty_value=1.0):
    """
    fg: (H, W) bool. Returns float32 (H, W):
        outside a pan:  +min(d, d_max) / d_max          (0 .. 1)
        inside a pan:   -min(d - 1, d_max) / d_max      (0 at the boundary, negative inside)
    per_object_interior=True instead scales the inside values per object, so the
    deepest pixel of EVERY pan is -1 regardless of pan size. With the paper's
    version, a 3-px-wide pan has interior values of only ~-0.05, so small pans
    get almost no pull toward predicting foreground; this option equalizes that.
    No foreground at all -> `empty_value` everywhere (penalize any predicted pan).
    """
    fg = np.asarray(fg, bool)
    if not fg.any():
        return np.full(fg.shape, empty_value, np.float32)
    if fg.all():
        return np.full(fg.shape, -1.0, np.float32)

    d_out = _edt(~fg)                  # distance to nearest foreground pixel (outside)
    d_in = _edt(fg)                    # distance to nearest background pixel (inside)
    outside = np.clip(d_out, 0, d_max) / d_max

    if per_object_interior:
        lbl, n = _label(fg, structure=_EIGHT)
        depth = np.asarray(_ndi_maximum(d_in, lbl, index=np.arange(1, n + 1)), dtype=float)
        depth = np.concatenate([[1.0], np.maximum(depth, 1.0)])
        inside = -d_in / depth[lbl]
    else:
        inside = -np.clip(d_in - 1, 0, d_max) / d_max

    return np.where(fg, inside, outside).astype(np.float32)


def boundary_distance_field(mask, d_max=32.0, empty_value=1.0):
    """
    (H, W) bool -> float32 unsigned distance to the nearest boundary of `mask`,
    clipped at d_max and divided by it (0 .. 1). An empty (or full) mask has no
    boundary, so it gets `empty_value` (maximum distance) everywhere.
    """
    mask = np.asarray(mask, bool)
    if not mask.any() or mask.all():
        return np.full(mask.shape, empty_value, np.float32)
    d = _edt(mask) + _edt(~mask)
    return (np.clip(d, 0, d_max) / d_max).astype(np.float32)


class BoundaryLoss(nn.Module):
    """
    loss = mean over valid pixels of  p_fg * phi,  phi = normalized signed distance
    to the true boundary. Minimizing it raises p_fg inside pans (phi < 0) and
    lowers it outside (phi > 0), with force proportional to the distance from the
    boundary - so far-away false positives are penalized hard, while errors right
    at the boundary barely register. Can be negative; that is expected.
    """

    def __init__(self, d_max=20.0, per_object_interior=False, ignore_index=-1, empty_value=1.0):
        super().__init__()
        self.d_max = d_max
        self.per_object_interior = per_object_interior
        self.ignore_index = ignore_index
        self.empty_value = empty_value

    def forward(self, logits, target):
        p_fg = F.softmax(logits, dim=1)[:, 1]                      # (B, H, W)
        valid = target != self.ignore_index
        fg_np = ((target == 1) & valid).detach().cpu().numpy()
        phi = np.stack([
            signed_distance_map(f, self.d_max, self.per_object_interior, self.empty_value)
            for f in fg_np
        ])
        phi = torch.from_numpy(phi).to(device=p_fg.device, dtype=p_fg.dtype)
        v = valid.to(p_fg.dtype)
        return (p_fg * phi * v).sum() / v.sum().clamp(min=1.0)


class HausdorffDTLoss(nn.Module):
    """
    loss = mean over valid pixels of  (p_fg - q)^2 * (dG^alpha + dS^alpha)
      q  = true mask,  S = prediction thresholded at 0.5 (no gradient)
      dG = normalized distance to the nearest true boundary
      dS = normalized distance to the nearest predicted boundary
    Errors that are far from both boundaries (hallucinated blobs far from any
    pan, pans with no prediction anywhere nearby) get the largest weights; errors
    within a pixel or two of a boundary get almost none. Good against large
    errors, weak against small systematic boundary bias.
    """

    def __init__(self, alpha=2.0, d_max=32.0, ignore_index=-1, empty_value=1.0):
        super().__init__()
        self.alpha = alpha
        self.d_max = d_max
        self.ignore_index = ignore_index
        self.empty_value = empty_value

    def forward(self, logits, target):
        p_fg = F.softmax(logits, dim=1)[:, 1]
        valid = target != self.ignore_index
        q = (target == 1) & valid
        pred_bin = (p_fg.detach() > 0.5) & valid

        q_np, s_np = q.detach().cpu().numpy(), pred_bin.cpu().numpy()
        dG = np.stack([boundary_distance_field(m, self.d_max, self.empty_value) for m in q_np])
        dS = np.stack([boundary_distance_field(m, self.d_max, self.empty_value) for m in s_np])
        weight = torch.from_numpy(dG ** self.alpha + dS ** self.alpha).to(
            device=p_fg.device, dtype=p_fg.dtype)

        v = valid.to(p_fg.dtype)
        err = (p_fg - q.to(p_fg.dtype)) ** 2
        return (err * weight * v).sum() / v.sum().clamp(min=1.0)


class ShapeCompoundLoss(nn.Module):
    """
    CE + w_boundary * BoundaryLoss + w_hd * HausdorffDTLoss, with the shape terms
    ramped in linearly: weight = w * clip((epoch - start_epoch) / ramp_epochs, 0, 1).
    Call set_epoch(epoch) at the start of every epoch. Shape terms are zero before
    start_epoch; ramp_epochs=0 means full weight immediately from start_epoch.

    self.last holds the unweighted components of the most recent call (floats), so
    you can log CE and the shape terms separately.
    """

    def __init__(self, ce_loss_fn, boundary_loss_fn=None, hd_loss_fn=None,
                 w_boundary=1.0, w_hd=1.0, ramp_epochs=20, start_epoch=0):
        super().__init__()
        self.ce_loss_fn = ce_loss_fn
        self.boundary_loss_fn = boundary_loss_fn
        self.hd_loss_fn = hd_loss_fn
        self.w_boundary = w_boundary
        self.w_hd = w_hd
        self.ramp_epochs = ramp_epochs
        self.start_epoch = start_epoch
        self.epoch = 0
        self.last = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def ramp(self):
        if self.epoch < self.start_epoch:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        return float(np.clip((self.epoch - self.start_epoch) / self.ramp_epochs, 0.0, 1.0))

    def forward(self, logits, target):
        ce = self.ce_loss_fn(logits, target)
        total = ce
        r = self.ramp()
        self.last = {"ce": float(ce.detach()), "ramp": r}

        if self.boundary_loss_fn is not None:
            b = self.boundary_loss_fn(logits, target)
            self.last["boundary"] = float(b.detach())
            if r > 0:
                total = total + r * self.w_boundary * b
        if self.hd_loss_fn is not None:
            h = self.hd_loss_fn(logits, target)
            self.last["hausdorff"] = float(h.detach())
            if r > 0:
                total = total + r * self.w_hd * h
        return total
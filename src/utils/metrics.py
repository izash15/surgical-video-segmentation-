
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: (N,H,W) int64
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0,3,1,2).float()

@torch.no_grad()
def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """
    pred: (N,H,W) int64
    target: (N,H,W) int64
    returns cm: (num_classes, num_classes) where rows=target, cols=pred
    """
    n = num_classes
    k = (target >= 0) & (target < n)
    inds = n * target[k].to(torch.int64) + pred[k]
    cm = torch.bincount(inds, minlength=n**2).reshape(n, n).float()
    return cm

@torch.no_grad()
def iou_from_cm(cm: torch.Tensor, ignore_index=None):
    # cm: (C,C)
    C = cm.shape[0]
    ious = []
    for c in range(C):
        if ignore_index is not None and c == ignore_index:
            continue
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else torch.tensor(0.0, device=cm.device))
    if len(ious) == 0:
        return torch.tensor(0.0, device=cm.device)
    return torch.stack(ious).mean()

@torch.no_grad()
def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index=None, eps: float=1e-6):
    # logits: (N,C,H,W), target: (N,H,W)
    pred = logits.argmax(1)
    C = num_classes
    dices = []
    for c in range(C):
        if ignore_index is not None and c == ignore_index:
            continue
        p = (pred==c)
        t = (target==c)
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        dices.append((2*inter + eps) / (denom + eps))
    if len(dices)==0:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(dices).mean()

class DiceCELoss(torch.nn.Module):
    def __init__(self, num_classes: int, weight=None, ignore_index=None, ce_weight: float=1.0, dice_weight: float=1.0, eps: float=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.eps = eps

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        # Dice on probabilities
        probs = torch.softmax(logits, dim=1)
        target_oh = torch.nn.functional.one_hot(target.clamp(min=0), num_classes=self.num_classes).permute(0,3,1,2).float()
        if self.ignore_index is not None:
            # mask out ignore_index
            mask = (target != self.ignore_index).float().unsqueeze(1)
            target_oh = target_oh * mask
            probs = probs * mask
        inter = (probs * target_oh).sum(dim=(0,2,3))
        denom = probs.sum(dim=(0,2,3)) + target_oh.sum(dim=(0,2,3))
        dice_per_class = (2*inter + self.eps) / (denom + self.eps)
        if self.ignore_index is not None and self.ignore_index < self.num_classes:
            valid = torch.ones(self.num_classes, device=logits.device, dtype=torch.bool)
            valid[self.ignore_index] = False
            dice = dice_per_class[valid].mean()
        else:
            dice = dice_per_class.mean()
        return self.ce_w * ce + self.dice_w * (1 - dice)

# --- Robust losses ---
class SafeBCEWithLogits(nn.Module):
    def __init__(self, pos_weight=None, logit_clip=15.0, reduction='mean'):
        super().__init__()
        self.crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
        self.clip = float(logit_clip)
    def forward(self, logits, targets):
        logits = logits.clamp(-self.clip, self.clip).float()
        targets = targets.float()
        return self.crit(logits, targets)

def dice_from_logits_safe(logits, target, num_classes, ignore_index=None, eps=1e-6):
    # Multiclass: logits (N,C,H,W), target (N,H,W) int; Multilabel: target (N,C,H,W) float (handle upstream)
    if logits.shape[1] == 1 or num_classes == 1:
        # binary (treat as multilabel with C=1)
        probs = torch.sigmoid(logits)
        target = target.float()
        inter = (probs * target).sum(dim=(2,3))
        denom = probs.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2*inter + eps) / (denom + eps)
        return dice.mean()
    else:
        # multiclass
        probs = torch.softmax(logits.float(), dim=1)
        N, C, H, W = probs.shape
        target_oh = torch.zeros_like(probs).scatter_(1, target.clamp(min=0).unsqueeze(1), 1.0)
        dices = []
        for c in range(C):
            if ignore_index is not None and c == ignore_index: continue
            # before: pc = probs[:, c]; tc = target_oh[:, c]
            pc = probs[:, c:c+1, ...]          # shape: (N, 1, H, W)
            tc = target_oh[:, c:c+1, ...]      # shape: (N, 1, H, W)

            inter = (pc * tc).sum(dim=(1,2,3))
            denom = pc.sum(dim=(1,2,3)) + tc.sum(dim=(1,2,3))
            dices.append((2*inter + eps) / (denom + eps))

        if len(dices) == 0:
            return torch.tensor(0.0, device=logits.device)
        return torch.stack(dices, dim=0).mean()


@torch.no_grad()
def dice_from_preds(preds: torch.Tensor,
                    targets: torch.Tensor,
                    num_classes: int,
                    ignore_index: int | None = None,
                    eps: float = 1e-6) -> float:
    """
    preds   : [B,H,W] long, discrete predictions (argmaxed)
    targets : [B,H,W] long
    Returns per-image class-mean Dice, averaged over images (NaN-safe).
    - If ignore_index not in [0, C-1] (e.g., 255), pixels with that value are ignored.
    - If ignore_index in [0, C-1] (e.g., 0), that *class* is excluded from the class-mean.
    """
    assert preds.shape == targets.shape and preds.dim() == 3
    assert preds.dtype == torch.long and targets.dtype == torch.long

    B, H, W = preds.shape
    per_img = preds.new_full((B,), float("nan"), dtype=torch.float32)

    # Interpret ignore_index
    pixel_ignore = None
    class_exclude = None
    if ignore_index is not None:
        if 0 <= ignore_index < num_classes:
            class_exclude = ignore_index
        else:
            pixel_ignore = ignore_index

    for b in range(B):
        p = preds[b].reshape(-1)
        t = targets[b].reshape(-1)

        # Mask out ignored pixels BEFORE one_hot
        if pixel_ignore is not None:
            valid = (t != pixel_ignore)
            p = p[valid]; t = t[valid]

        if p.numel() == 0:
            continue  # leave NaN for this image

        # Safety: clamp labels into [0, C-1] to avoid one_hot OOB
        p = p.clamp_(0, num_classes - 1)
        t = t.clamp_(0, num_classes - 1)

        po = F.one_hot(p, num_classes=num_classes).to(torch.float32)  # [N,C]
        to = F.one_hot(t, num_classes=num_classes).to(torch.float32)  # [N,C]

        inter = (po * to).sum(dim=0)                 # [C]
        denom = po.sum(dim=0) + to.sum(dim=0)        # [C]
        dice_c = (2 * inter + eps) / (denom + eps)   # [C]

        if class_exclude is not None:
            if 0 <= class_exclude < num_classes and dice_c.numel() > 1:
                dice_c = torch.cat([dice_c[:class_exclude], dice_c[class_exclude+1:]])

        if dice_c.numel() > 0:
            per_img[b] = dice_c.mean()

    return torch.nanmean(per_img).item()


@torch.no_grad()
def iou_from_preds(preds: torch.Tensor,
                   targets: torch.Tensor,
                   num_classes: int,
                   ignore_index: int | None = None,
                   eps: float = 1e-6) -> float:
    """
    preds   : [B,H,W] long, discrete predictions (argmaxed)
    targets : [B,H,W] long
    Returns per-image class-mean IoU, averaged over images (NaN-safe).
    - If ignore_index not in [0, C-1] (e.g., 255), pixels with that value are ignored.
    - If ignore_index in [0, C-1] (e.g., 0), that *class* is excluded from the class-mean.
    """
    assert preds.shape == targets.shape and preds.dim() == 3
    assert preds.dtype == torch.long and targets.dtype == torch.long

    B, H, W = preds.shape
    per_img = preds.new_full((B,), float("nan"), dtype=torch.float32)

    pixel_ignore = None
    class_exclude = None
    if ignore_index is not None:
        if 0 <= ignore_index < num_classes:
            class_exclude = ignore_index
        else:
            pixel_ignore = ignore_index

    for b in range(B):
        p = preds[b].reshape(-1)
        t = targets[b].reshape(-1)

        if pixel_ignore is not None:
            valid = (t != pixel_ignore)
            p = p[valid]; t = t[valid]

        if p.numel() == 0:
            continue

        p = p.clamp_(0, num_classes - 1)
        t = t.clamp_(0, num_classes - 1)

        po = F.one_hot(p, num_classes=num_classes).to(torch.float32)  # [N,C]
        to = F.one_hot(t, num_classes=num_classes).to(torch.float32)  # [N,C]

        inter = (po * to).sum(dim=0)                  # [C]
        union = po.sum(dim=0) + to.sum(dim=0) - inter # [C]
        iou_c = (inter + eps) / (union + eps)         # [C]

        if class_exclude is not None:
            if 0 <= class_exclude < num_classes and iou_c.numel() > 1:
                iou_c = torch.cat([iou_c[:class_exclude], iou_c[class_exclude+1:]])

        if iou_c.numel() > 0:
            per_img[b] = iou_c.mean()

    return torch.nanmean(per_img).item()


class DiceCELoss_Safe(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, logit_clip=15.0, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        if num_classes == 1:
            self.ce = SafeBCEWithLogits(logit_clip=logit_clip)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.clip = logit_clip

    def forward(self, logits, target):
        if self.num_classes == 1:
            ce = self.ce(logits, target)
        else:
            ce = self.ce(logits.clamp(-self.clip, self.clip).float(), target.long())
        dice = 1.0 - dice_from_logits_safe(logits, target, self.num_classes, self.ignore_index)
        loss = self.ce_weight * ce + self.dice_weight * dice
        # NaN/Inf guard
        loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
        return loss


# metrics.py
# -----------------------------------------------------------
# MASD & Surface Dice (NSD@tau) for segmentation masks
# - SciPy-free (pure NumPy; optional scikit-learn accel)
# - Works for 2D/3D, anisotropic spacing
# - Batch helper for PyTorch tensors with macro averaging
# -----------------------------------------------------------

# Torch is only needed for the batch wrapper; keep import lightweight


ArrayLike = Union[np.ndarray, "torch.Tensor"]  # for type hints

__all__ = [
    "masd",
    "nsd_surface_dice",
    "compute_surface_metrics_batch",
]

# -----------------------
# Internal NumPy helpers
# -----------------------

def _as_bool_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr if arr.dtype == bool else (arr != 0)


def _surface_mask_connectivity1(mask: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask marking boundary pixels/voxels of `mask` using
    4-connected (2D) / 6-connected (3D) neighborhood. NumPy-only.

    Surface = foreground voxels that have at least one connected neighbor == 0.
    """
    mask = _as_bool_np(mask)
    if mask.ndim not in (2, 3):
        raise ValueError("Only 2D/3D supported.")
    fg = mask

    if mask.ndim == 2:
        up    = np.zeros_like(fg); up[1:]      = fg[:-1]
        down  = np.zeros_like(fg); down[:-1]   = fg[1:]
        left  = np.zeros_like(fg); left[:, 1:] = fg[:, :-1]
        right = np.zeros_like(fg); right[:, :-1] = fg[:, 1:]
        # surface if any neighbor is background
        bg_nb = (~up) | (~down) | (~left) | (~right)
        surface = fg & bg_nb
    else:
        zf = fg
        zm = np.zeros_like(zf);  zm[1:]       = zf[:-1]      # shift -z
        zp = np.zeros_like(zf);  zp[:-1]      = zf[1:]       # shift +z
        ym = np.zeros_like(zf);  ym[:, 1:]    = zf[:, :-1]   # shift -y
        yp = np.zeros_like(zf);  yp[:, :-1]   = zf[:, 1:]    # shift +y
        xm = np.zeros_like(zf);  xm[:, :, 1:] = zf[:, :, :-1]# shift -x
        xp = np.zeros_like(zf);  xp[:, :, :-1]= zf[:, :, 1:] # shift +x
        bg_nb = (~zm) | (~zp) | (~ym) | (~yp) | (~xm) | (~xp)
        surface = fg & bg_nb

    return surface


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return N x D integer coordinates of True entries (argwhere)."""
    return np.argwhere(mask)


def _normalize_spacing(spacing: Union[float, List[float], Tuple[float, ...]], dim: int) -> np.ndarray:
    """
    Normalize spacing to a float array of length `dim`.
    - If scalar, replicate for each dimension.
    - If length == 3 and dim == 2, drop the first (assume (z,y,x) -> (y,x)).
    - If length mismatch, raise.
    """
    sp = np.asarray(spacing if isinstance(spacing, (list, tuple, np.ndarray)) else (spacing,), dtype=np.float64)
    if dim == 2 and sp.size == 3:
        sp = sp[-2:]
    if sp.size == 1:
        sp = np.repeat(sp, dim)
    if sp.size != dim:
        raise ValueError(f"Spacing length {sp.size} does not match dim={dim}.")
    return sp


def _scale_coords(coords: np.ndarray, spacing: Union[float, List[float], Tuple[float, ...]]) -> np.ndarray:
    """Scale integer coords to physical coords with given spacing."""
    if coords.size == 0:
        return coords.astype(np.float64)
    sp = _normalize_spacing(spacing, coords.shape[1])
    return coords.astype(np.float64) * sp[np.newaxis, :]


def _nearest_dists(A_phys: np.ndarray, B_phys: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """
    Distances from each row in A_phys (N x D) to nearest point in B_phys (M x D).
    Uses scikit-learn's NearestNeighbors if available; otherwise chunked NumPy.
    Handles empty inputs.
    """
    N = A_phys.shape[0]
    M = B_phys.shape[0]

    if N == 0:
        return np.zeros(0, dtype=np.float64)
    if M == 0:
        return np.full(N, np.inf, dtype=np.float64)

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(B_phys)
        dists, _ = nn.kneighbors(A_phys, return_distance=True)
        return dists.ravel().astype(np.float64)
    except Exception:
        # Chunked brute-force to limit memory
        out = np.empty(N, dtype=np.float64)
        for i in range(0, N, chunk):
            a = A_phys[i:i+chunk]              # (c, D)
            diff = a[:, None, :] - B_phys[None, :, :]  # (c, M, D)
            d2 = np.sum(diff * diff, axis=2)   # (c, M)
            out[i:i+chunk] = np.sqrt(d2.min(axis=1))
        return out


# -----------------------
# Public metrics (NumPy)
# -----------------------

def masd(pred_bin: np.ndarray, gt_bin: np.ndarray,
         spacing: Union[float, List[float], Tuple[float, ...]] = 1.0) -> float:
    """
    Mean Average Symmetric Surface Distance (lower is better).
    Inputs are binary NumPy arrays (2D or 3D). Distances in physical units.

    Edge cases:
      - Both surfaces empty -> 0.0
      - One empty, one non-empty -> +inf
    """
    pred_bin = _as_bool_np(pred_bin)
    gt_bin   = _as_bool_np(gt_bin)

    if pred_bin.shape != gt_bin.shape:
        raise ValueError("pred_bin and gt_bin must have the same shape.")

    Sp = _surface_mask_connectivity1(pred_bin)
    Sg = _surface_mask_connectivity1(gt_bin)
    np_sp, ng_sp = int(Sp.sum()), int(Sg.sum())

    if np_sp == 0 and ng_sp == 0:
        return 0.0
    if np_sp == 0 or ng_sp == 0:
        return  float("nan")

    P = _scale_coords(_coords_from_mask(Sp), spacing)
    G = _scale_coords(_coords_from_mask(Sg), spacing)

    d_p_to_g = _nearest_dists(P, G)
    d_g_to_p = _nearest_dists(G, P)
    return float((d_p_to_g.sum() + d_g_to_p.sum()) / (np_sp + ng_sp))


def nsd_surface_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, tau: float,
                     spacing: Union[float, List[float], Tuple[float, ...]] = 1.0) -> float:
    """
    Surface Dice at tolerance tau (aka NSD in 'surface-dice' sense). 0..1, higher is better.
    Inputs are binary NumPy arrays (2D/3D). tau in same physical units as `spacing`.

    Edge cases:
      - Both surfaces empty -> 1.0
      - One empty, one non-empty -> 0.0
    """
    pred_bin = _as_bool_np(pred_bin)
    gt_bin   = _as_bool_np(gt_bin)

    if pred_bin.shape != gt_bin.shape:
        raise ValueError("pred_bin and gt_bin must have the same shape.")

    Sp = _surface_mask_connectivity1(pred_bin)
    Sg = _surface_mask_connectivity1(gt_bin)
    np_sp, ng_sp = int(Sp.sum()), int(Sg.sum())

    if np_sp == 0 and ng_sp == 0:
        return 1.0
    if np_sp == 0 or ng_sp == 0:
        return 0.0

    P = _scale_coords(_coords_from_mask(Sp), spacing)
    G = _scale_coords(_coords_from_mask(Sg), spacing)

    d_p_to_g = _nearest_dists(P, G)
    d_g_to_p = _nearest_dists(G, P)

    hits = int((d_p_to_g <= tau).sum()) + int((d_g_to_p <= tau).sum())
    return float(hits / (np_sp + ng_sp))


# --------------------------------------------
# PyTorch batch wrapper (macro-average over classes)
# --------------------------------------------

# def compute_surface_metrics_batch(
#     preds: "torch.Tensor",
#     gts:   "torch.Tensor",
#     num_classes: int,
#     spacing: Union[float, List[float], Tuple[float, ...]] = 1.0,
#     tau: float = 2.0,
#     ignore_index: Optional[int] = None
# ) -> Tuple[float, float, Dict[int, Dict[str, float]]]:
#     """
#     Compute MASD and Surface Dice@tau for a batch of predictions vs GT (class indices).
#     - Supports 2D: (B, H, W) and 3D: (B, D, H, W)
#     - One-vs-rest per class; returns macro means over valid classes.
#     - Skips `ignore_index` if provided.

#     Args:
#       preds: Long tensor of shape (B,H,W) or (B,D,H,W) with predicted class indices.
#       gts:   Long tensor of same shape with ground-truth class indices.
#       num_classes: total number of classes.
#       spacing: scalar or sequence in physical units (e.g., (z,y,x) or (y,x)).
#       tau: tolerance for Surface Dice (same units as spacing).
#       ignore_index: class id to exclude from averaging (e.g., background).

#     Returns:
#       masd_mean: float (macro-average across valid classes)
#       nsd_mean:  float (macro-average across valid classes)
#       per_class: dict[c] -> {"masd": float, "nsd": float}
#     """
#     if preds.shape != gts.shape:
#         raise ValueError("preds and gts must have the same shape.")

#     if preds.ndim not in (3, 4):
#         raise ValueError("Expected preds/gts with shape (B,H,W) or (B,D,H,W).")

#     preds_np = preds.detach().cpu().numpy()
#     gts_np   = gts.detach().cpu().numpy()
#     B = preds_np.shape[0]

#     valid_classes = [c for c in range(num_classes) if (ignore_index is None or c != ignore_index)]
#     per_class_vals = {c: {'masd': [], 'nsd': []} for c in valid_classes}

#     # Iterate batch and classes
#     for b in range(B):
#         p = preds_np[b]
#         g = gts_np[b]
#         for c in valid_classes:
#             p_bin = (p == c)
#             g_bin = (g == c)
#             # Compute class-wise metrics
#             m = masd(p_bin, g_bin, spacing=spacing)
#             s = nsd_surface_dice(p_bin, g_bin, tau=tau, spacing=spacing)
#             per_class_vals[c]['masd'].append(m)
#             per_class_vals[c]['nsd'].append(s)

#     # Aggregate per-class means, then macro-average
#     per_class: Dict[int, Dict[str, float]] = {}
#     masd_list: List[float] = []
#     nsd_list:  List[float] = []

#     for c in valid_classes:
#         m_vals = np.asarray(per_class_vals[c]['masd'], dtype=np.float64)
#         s_vals = np.asarray(per_class_vals[c]['nsd'],  dtype=np.float64)

#         # Use nanmean so frames with undefined MASD (one-empty) don't explode the mean
#         m_mean = float(np.nanmean(m_vals)) if m_vals.size > 0 else float("nan")
#         s_mean = float(np.nanmean(s_vals)) if s_vals.size > 0 else float("nan")

#         per_class[c] = {'masd': m_mean, 'nsd': s_mean}
#         masd_list.append(m_mean)
#         nsd_list.append(s_mean)

#     masd_mean = float(np.nanmean(masd_list)) if len(masd_list) else float("nan")
#     nsd_mean  = float(np.nanmean(nsd_list))  if len(nsd_list)  else float("nan")
#     return masd_mean, nsd_mean, per_class


def compute_surface_metrics_batch(
    preds: "torch.Tensor",
    gts:   "torch.Tensor",
    num_classes: int,
    spacing: Union[float, List[float], Tuple[float, ...]] = 1.0,
    tau: float = 2.0,
    # NEW: split the semantics
    class_exclude: Optional[int] = None,   # e.g., 0 to drop background from macro
    pixel_ignore: Optional[int] = None     # e.g., 255 to mask unlabeled pixels
) -> Tuple[float, float, Dict[int, Dict[str, float]]]:
    """
    Compute MASD and NSD per image and per class, then macro-average across classes.
    - preds/gts: long tensors (B,H,W) or (B,D,H,W) with class indices
    - class_exclude: class id to exclude from macro-mean (drop background from averaging)
    - pixel_ignore: pixel label to remove from eval (mask out unlabeled 255s, etc.)
    - spacing: pixel or physical spacing; must match training if you want comparable numbers
    - MASD: returns NaN for one-empty cases; we average with nanmean
    - NSD: keeps your semantics (both empty->1, one empty->0)
    """
    if preds.shape != gts.shape:
        raise ValueError("preds and gts must have the same shape.")
    if preds.ndim not in (3, 4):
        raise ValueError("Expected preds/gts with shape (B,H,W) or (B,D,H,W).")

    preds_np = preds.detach().cpu().numpy()
    gts_np   = gts.detach().cpu().numpy()
    B = preds_np.shape[0]

    valid_classes = [c for c in range(num_classes) if (class_exclude is None or c != class_exclude)]
    per_class_vals = {c: {'masd': [], 'nsd': []} for c in valid_classes}

    for b in range(B):
        p = preds_np[b]
        g = gts_np[b]

        # mask out pixel_ignore (e.g., 255) identically for preds & gts
        if pixel_ignore is not None:
            # keep only valid region; outside it set both to background 0 so no spurious surfaces
            valid = (g != pixel_ignore)
            # NOTE: we rely on class_exclude to drop bg from averaging, not on removing bg pixels
            p = np.where(valid, p, 0)
            g = np.where(valid, g, 0)

        for c in valid_classes:
            p_bin = (p == c)
            g_bin = (g == c)
            # print("p_bin sum =", int(p_bin.sum()), "  g_bin sum =", int(g_bin.sum()))

            m = masd(p_bin, g_bin, spacing=spacing)                       # returns NaN for one-empty
            s = nsd_surface_dice(p_bin, g_bin, tau=tau, spacing=spacing)  # 1 for both-empty, 0 for one-empty

            per_class_vals[c]['masd'].append(m)
            per_class_vals[c]['nsd'].append(s)

    per_class: Dict[int, Dict[str, float]] = {}
    masd_list: List[float] = []
    nsd_list:  List[float] = []

    for c in valid_classes:
        m_vals = np.asarray(per_class_vals[c]['masd'], dtype=np.float64)
        s_vals = np.asarray(per_class_vals[c]['nsd'],  dtype=np.float64)

        m_mean = float(np.nanmean(m_vals)) if m_vals.size > 0 else float("nan")
        s_mean = float(np.nanmean(s_vals)) if s_vals.size > 0 else float("nan")

        per_class[c] = {'masd': m_mean, 'nsd': s_mean}
        masd_list.append(m_mean)
        nsd_list.append(s_mean)

    masd_mean = float(np.nanmean(masd_list)) if len(masd_list) else float("nan")
    nsd_mean  = float(np.nanmean(nsd_list))  if len(nsd_list)  else float("nan")
    return masd_mean, nsd_mean, per_class



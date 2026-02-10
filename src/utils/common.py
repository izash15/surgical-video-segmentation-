
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def colorize_mask(mask: np.ndarray, colormap=None):
    """
    Convert single-channel class-index mask (H,W) to color image (H,W,3).
    colormap: dict {class_id: (R,G,B)}; default generates a tab20-like map.
    """
    if colormap is None:
        # simple deterministic palette
        rng = np.random.default_rng(1234)
        palette = rng.integers(0, 255, size=(256,3), dtype=np.uint8)
        palette[0] = (0,0,0)
        colormap = {i: tuple(palette[i]) for i in range(256)}
    h, w = mask.shape
    color = np.zeros((h,w,3), dtype=np.uint8)
    for cid, rgb in colormap.items():
        color[mask==cid] = rgb
    return color

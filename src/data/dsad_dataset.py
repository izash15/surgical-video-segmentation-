import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _find_pairs(
    img_dir: Path,
    mask_dir: Path,
    exts_img=(".png", ".jpg", ".jpeg", ".tif"),
    exts_mask=(".png", ".jpg", ".jpeg", ".tif"),
) -> Tuple[List[Path], List[Path]]:
    images, masks = [], []
    img_map = {}
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in exts_img and p.is_file():
            img_map[p.stem] = p
    for q in sorted(mask_dir.rglob("*")):
        if q.suffix.lower() in exts_mask and q.is_file():
            stem = q.stem
            if stem in img_map:
                images.append(img_map[stem])
                masks.append(q)
    return images, masks


class DSADDataset(Dataset):
    """
    Generic dataset expecting:
      - images_subdir: RGB images
      - masks_subdir : single-channel masks with class indices [0..C-1]
    Optional: provide a 'list_file' with one filename stem per line to subset.
    """
    def __init__(
        self,
        root: str,
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        list_file: Optional[str] = None,
        img_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
    ):
        self.root = Path(root)
        self.img_dir = self.root / images_subdir
        self.mask_dir = self.root / masks_subdir
        all_imgs, all_masks = _find_pairs(self.img_dir, self.mask_dir)

        if list_file is not None and os.path.isfile(list_file):
            stems = set(
                [
                    l.strip()
                    for l in open(list_file, "r").read().splitlines()
                    if l.strip()
                ]
            )
            filtered = [
                (i, m) for i, m in zip(all_imgs, all_masks) if i.stem in stems
            ]
            if len(filtered) == 0:
                raise RuntimeError(
                    f"No pairs matched the provided list file: {list_file}"
                )
            all_imgs, all_masks = map(list, zip(*filtered))

        if len(all_imgs) == 0:
            raise RuntimeError(
                f"No image/mask pairs found under {self.img_dir} and {self.mask_dir}."
            )

        self.images = all_imgs
        self.masks = all_masks
        self.size = img_size  # (H, W)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def _load_pair(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        # ensure single-channel mask
        if mask.mode != "L":
            mask = mask.convert("L")
        img_np = np.array(img)
        mask_np = np.array(mask)
        return img_np, mask_np

    def _resize_and_pad(self, img: Image.Image, mask: Image.Image):
        """
        Resize so that the longest side == max(self.size),
        then zero-pad to at least (H, W) = self.size.

        Mimics: LongestMaxSize(max(size)) + PadIfNeeded(H, W, border_mode=0)
        using pure PIL/torchvision.
        """
        from torchvision.transforms import functional as TF

        target_h, target_w = self.size  # e.g. (512, 512)
        max_size = max(target_h, target_w)

        # PIL size: (width, height)
        w, h = img.size

        # scale factor for longest side
        scale = max_size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # resize image and mask
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # compute padding to reach at least (target_h, target_w)
        pad_w = max(target_w - new_w, 0)
        pad_h = max(target_h - new_h, 0)

        # center padding: (left, top, right, bottom)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if pad_w > 0 or pad_h > 0:
            img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

        return img, mask

    def __getitem__(self, idx):
        from torchvision.transforms import functional as TF
        import random

        img_np, mask_np = self._load_pair(idx)

        # back to PIL for transforms
        img = Image.fromarray(img_np)
        mask = Image.fromarray(mask_np)

        # resize + zero-pad
        img, mask = self._resize_and_pad(img, mask)

        # simple horizontal flip augmentation (optional)
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # to tensors
        img_t = TF.to_tensor(img).float()  # [0,1], C×H×W
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img_t, mask_t, str(self.images[idx])

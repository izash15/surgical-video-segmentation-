from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

#to run this run command: 
#python /home/idavis3/surgical-video-segmentation-/src/data/test_vit_prep.py
# run in the home/name directory!

# ── CONFIG — change these two lines to match your paths ──────
DATA_ROOT = Path("/home/idavis3/DSAD_pped/set3/")
ORGAN     = "liver"   # pick any organ folder to test with
# ─────────────────────────────────────────────────────────────

def prepare_for_vit(x):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(x)


def inspect_tensor(name, t):
    """Prints a detailed report on a tensor so you can see exactly what it contains."""
    print(f"\n── {name} ──────────────────────────")
    print(f"  shape  : {tuple(t.shape)}")          # e.g. (3, 512, 512)
    print(f"  dtype  : {t.dtype}")                 # should be float32
    print(f"  min    : {t.min():.4f}")             # after normalisation goes negative
    print(f"  max    : {t.max():.4f}")
    print(f"  mean   : {t.mean():.4f}")
    print(f"  channels: R={t[0].mean():.3f}  G={t[1].mean():.3f}  B={t[2].mean():.3f}")


# ── STEP 1: find one real image from your data ───────────────
image_dir = DATA_ROOT / ORGAN / "images"
image_path = next(image_dir.iterdir())   # grabs the first image it finds
print(f"Testing with: {image_path.name}")

# ── STEP 2: load with PIL and check raw state ─────────────────
pil_image = Image.open(image_path)
print(f"\n── Raw PIL Image ───────────────────────")
print(f"  mode   : {pil_image.mode}")             # want RGB — if RGBA or L, flag it
print(f"  size   : {pil_image.size}")             # (W, H) — note PIL gives W first

# Force RGB just in case
pil_image = pil_image.convert("RGB")
print(f"  mode after .convert('RGB'): {pil_image.mode}")

# ── STEP 3: convert to tensor (mimics what your Dataset will do) ──
# ToTensor() converts PIL (H,W,C) uint8 [0-255] → float32 (C,H,W) [0-1]
to_tensor = transforms.ToTensor()
raw_tensor = to_tensor(pil_image)
inspect_tensor("After ToTensor (before ViT transform)", raw_tensor)

# ── STEP 4: run through prepare_for_vit ──────────────────────
vit_tensor = prepare_for_vit(raw_tensor)
inspect_tensor("After prepare_for_vit", vit_tensor)

# ── STEP 5: simulate what the model receives — add batch dim ──
# Model expects (B, 3, H, W) not (3, H, W)
batch = vit_tensor.unsqueeze(0)   # (3,512,512) → (1,3,512,512)
print(f"\n── Batch tensor (model input) ──────────")
print(f"  shape : {tuple(batch.shape)}")          # expect (1, 3, 512, 512)
print(f"  ready for MiT-B0: {batch.shape[1]==3 and batch.shape[2]%32==0 and batch.dtype==torch.float32}")
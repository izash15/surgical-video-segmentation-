# test_tripathMN.py
import argparse, os, json, time, csv, random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# --- project imports (match your repo structure) ---
from models.tripath import TriPathUNet, TriPathUNetStacked, TriPathDSConv, TriPathCNN,TriPathDC,TriPathDSConv
from models.unet import UNet
from data.dsad_dataset import DSADDataset
from utils.metrics import (
    DiceCELoss, compute_surface_metrics_batch,
    iou_from_cm, dice_from_logits
)
from utils.common import set_seed, count_trainable_parameters


# ===================== helpers =====================
def _nowstamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _make_run_dir(save_dir, run_name):
    run_dir = Path(save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _save_epoch_csv(run_dir, payload):
    csv_f = open(run_dir / "metrics_epoch.csv", "a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "split", "loss", "mIoU", "mDice", "MASD", "NSD", "epoch_secs"
    ])
    if csv_f.tell() == 0:
        csv_w.writeheader()
    csv_w.writerow(payload)
    csv_f.flush()
    csv_f.close()

import torch
import torch.nn.functional as F

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

def _maybe_overwrite_from_ckpt_args(args, ckpt_args):
    # Minimal sync of a few fields if user didn't pass them
    for k in ["num_classes", "img_size", "ignore_bg", "images_subdir", "masks_subdir"]:
        if hasattr(args, k) and getattr(args, k) is not None:
            continue
        if k in ckpt_args:
            setattr(args, k, ckpt_args[k])

def _build_net(args):
    # Keep default consistent with training unless you switch here
    # net = TriPathUNet(in_ch=3, num_classes=args.num_classes, base_ch=32)
    # Alternatives:
    # net = TriPathCNN(in_ch=3, num_classes=args.num_classes, base_ch=32)
    # net = TriPathDC(in_ch=3, num_classes=args.num_classes, base_ch=32)
    net = TriPathDSConv(in_ch=3, num_classes=args.num_classes, base_ch=32)
    # net = UNet(in_ch=3, num_classes=args.num_classes, base_ch=64)
    # net = TriPathUNetStacked(in_ch=3, num_classes=args.num_classes, base_ch=32) 
    return net

def _to_uint8_mask(mask_np):
    mask_np = np.asarray(mask_np, dtype=np.uint8)
    return Image.fromarray(mask_np, mode="L")

def _normalize01(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def _sanitize_masks(masks, num_classes, ignore_val=258, binarize=False):
    """
    Ensures masks are safe for indexing operations:
    - optional binarization (>0 -> 1) but preserves ignore_val
    - remap any remaining invalid labels to ignore_val
    Returns a cloned tensor on the same device.
    """
    m = masks.clone()
    if binarize:
        # Preserve ignore first
        is_ign = (m == ignore_val)
        m = (m > 0).to(m.dtype)
        m[is_ign] = ignore_val

    invalid = (m < 0) | (m >= num_classes)
    m[invalid] = ignore_val
    return m

def _safe_confusion_update(cm, preds, masks, num_classes, ignore_val):
    # Flatten
    pf = preds.view(-1)
    mf = masks.view(-1)
    valid = (mf != ignore_val)
    if valid.any():
        idx = (mf[valid] * num_classes + pf[valid]).to(torch.long)
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        cm += binc.view(num_classes, num_classes).to(cm.device)


# ===================== args =====================
def parse_args():
    ap = argparse.ArgumentParser("Test/Eval DSAD segmentation model (set-level metrics + examples).")
    ap.add_argument("--data-root", type=str, required=True, help="Root folder with images/ and masks/")
    ap.add_argument("--images-subdir", type=str, default="images")
    ap.add_argument("--masks-subdir", type=str, default="masks")
    ap.add_argument("--list-test", type=str, default=None, help="Optional test list file (one stem per line)")
    ap.add_argument("--img-size", type=int, nargs=2, default=[512, 512], help="W H")
    ap.add_argument("--num-classes", type=int, required=True, help="For binary use 2")
    ap.add_argument("--ignore-bg", action="store_true", help="If set, metrics may ignore class 0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="Use mixed precision for inference")
    ap.add_argument("--seed", type=int, default=42)

    # NEW: model loading
    ap.add_argument("--model-dir", type=str, required=True, help="Directory that contains ckpt and (optionally) args.json in checkpoint['args']")
    ap.add_argument("--ckpt-name", type=str, default="best_TriPath.ckpt", help="Checkpoint filename inside model-dir")

    # NEW: output / examples
    ap.add_argument("--save-dir", type=str, default="./eval_outputs")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--sample-examples", type=int, default=8, help="Number of qualitative samples to export")
    ap.add_argument("--tau", type=float, default=2.0, help="NSD tolerance (pixels)")

    # NEW: robust mask handling to prevent CUDA index errors
    ap.add_argument("--ignore-val", type=int, default=258, help="Label value to ignore in loss/metrics")
    ap.add_argument("--binarize-mask", action="store_true",
                    help="Force masks to {0,1,ignore_val}; helpful if per-organ masks still contain global IDs")

    # Debug
    ap.add_argument("--print-uniques", action="store_true", help="Print unique mask labels for first few batches")
    return ap.parse_args()


# ===================== main =====================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare run dir
    args.run_name = args.run_name or f"eval-{_nowstamp()}"
    run_dir = _make_run_dir(args.save_dir, args.run_name)
    (run_dir / "examples").mkdir(parents=True, exist_ok=True)

    # Load checkpoint (+ possibly training-time args)
    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        _maybe_overwrite_from_ckpt_args(args, ckpt["args"])

    # Dataset (test-only)
    ds_test = DSADDataset(
        args.data_root, args.images_subdir, args.masks_subdir,
        list_file=args.list_test, img_size=args.img_size, augment=False
    )
    dl_test = DataLoader(ds_test, batch_size=max(1, args.batch_size), shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Build net and load weights
    net = _build_net(args).to(device)
    print(f"Model params: {count_trainable_parameters(net)/1e6:.2f}M")

    state = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warning] missing keys: {missing}\n[Warning] unexpected keys: {unexpected}")

    total = sum(p.numel() for p in net.parameters())
    loaded = 0
    for k, v in state.items():
        if k in net.state_dict() and net.state_dict()[k].shape == v.shape:
            loaded += v.numel()

    print(f"[ckpt] Loaded params (matching shape): {loaded}/{total} "
        f"({100.0*loaded/total:.2f}%)")
    print(f"[ckpt] Missing keys: {len(missing)}")
    if missing:    print("  - " + "\n  - ".join(missing[:10]) + (" ..." if len(missing)>10 else ""))
    print(f"[ckpt] Unexpected keys: {len(unexpected)}")
    if unexpected: print("  - " + "\n  - ".join(unexpected[:10]) + (" ..." if len(unexpected)>10 else ""))

    net.eval()



    # Loss (kept for reporting consistency)
    IGNORE_VAL = args.ignore_val
    criterion = DiceCELoss(
        num_classes=args.num_classes,
        ignore_index=IGNORE_VAL,
        ce_weight=1.0,
        dice_weight=1.0
    )

    # ---- Inference / evaluation ----
    net.eval()
    cm = torch.zeros(args.num_classes, args.num_classes, device=device)
    total_loss = 0.0
    masd_sum, nsd_sum = 0.0, 0.0
    masd_count, nsd_count = 0, 0
    dice_sum, iou_sum = 0.0,0.0
    dice_count, iou_count = 0,0
    # choose qualitative examples
    N = len(ds_test)
    pick_n = min(args.sample_examples, N)
    sampled_ids = set(random.sample(range(N), pick_n)) if pick_n > 0 else set()
    saved_meta = []

    start_t = time.time()
    ex_count = 0
    printed_uniques = False

    with torch.no_grad():
        pbar = tqdm(dl_test, desc=f"Testing {N} images")
        autocast = torch.cuda.amp.autocast if args.amp and torch.cuda.is_available() else torch.cpu.amp.autocast

        with autocast():
            offset = 0
            for imgs, masks, metas in pbar:
                bsz = imgs.size(0)
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                if args.binarize_mask:
                    # Convert >0 to 1, preserve IGNORE_VAL if present
                    is_ign = (masks == IGNORE_VAL)
                    masks = (masks > 0).to(masks.dtype)
                    masks[is_ign] = IGNORE_VAL

                # print("Pro1 Unique mask:", torch.unique(masks).tolist())
                # Ensure no label outside [0, num_classes-1]
                masks = _sanitize_masks(masks, num_classes=args.num_classes,
                                        ignore_val=IGNORE_VAL, binarize=False)

                logits = net(imgs)
                # loss = criterion(logits, masks).item()
                loss = 1
                # print("Loss part over")
                total_loss += loss * bsz

                preds = logits.argmax(1)
              

                # Safe CM update 
                _safe_confusion_update(cm, preds, masks, args.num_classes, IGNORE_VAL)

                batch_masd, batch_nsd, _ = compute_surface_metrics_batch(
                    preds, masks, num_classes=args.num_classes, tau=args.tau, class_exclude=0,
                pixel_ignore=255)

                # exclude inf (failed to detect)
                if np.isfinite(batch_masd):
                    masd_sum += float(batch_masd)
                    masd_count +=1 

                nsd_sum  += float(batch_nsd)  if np.isfinite(batch_nsd)  else 0.0
                nsd_count  += 1
                batch_dice = dice_from_preds(preds, masks, num_classes=args.num_classes,
                                            ignore_index=0)
                batch_iou  = iou_from_preds(preds, masks, num_classes=args.num_classes,
                                            ignore_index=0)
                dice_sum += batch_dice
                iou_sum  += batch_iou
                dice_count += 1
                iou_count += 1

                # print("MASD part over")
                # Save qualitative samples (pred mask + confidence score map)
                for i in range(bsz):
                    global_idx = offset + i
                    if global_idx in sampled_ids:
                        stem = None
                        if isinstance(metas, (list, tuple)) and len(metas) == bsz:
                            stem = metas[i]
                        if stem is None:
                            stem = f"idx_{global_idx:06d}"
                        
                        stem_str = str(stem)
                        stem_base = os.path.basename(stem_str)            # → "04_000051.png"
                        stem1 = os.path.splitext(stem_base)[0]            # → "04_000051"

                        # ----------- save predicted mask (binary 0/255) -----------
                        pred_np = preds[i].detach().cpu().numpy()         # [H,W], values {0,1}
                        pred_bin = (pred_np > 0).astype(np.uint8) * 255   # -> {0,255}
                        Image.fromarray(pred_bin, mode="L").save(run_dir / "examples" / f"{stem1}_pred.png")

                        # ----------- save GT mask (binary 0/255) -----------
                        gt_np = masks[i].detach().cpu().numpy()
                        # For safety: clamp any ignore label 255 → 0
                        gt_np = np.where(gt_np == IGNORE_VAL, 0, gt_np)
                        gt_bin = (gt_np > 0).astype(np.uint8) * 255
                        Image.fromarray(gt_bin, mode="L").save(run_dir / "examples" / f"{stem1}_gt.png")


                        # confidence (max-softmax)
                        prob = F.softmax(logits[i], dim=0)         # [C,H,W]
                        conf_map, _ = prob.max(dim=0)               # [H,W]
                        conf_np = conf_map.detach().cpu().numpy()
                        conf_img = Image.fromarray((_normalize01(conf_np) * 255.0).astype(np.uint8), mode="L")
                        conf_img.save(run_dir / "examples" / f"{stem1}_confidence.png")

                        saved_meta.append({"stem1": stem1, "mean_confidence": float(conf_np.mean())})
                        ex_count += 1

                offset += bsz

    # Aggregate
    elapsed = time.time() - start_t
    avg_loss = total_loss / max(1, len(dl_test.dataset))

    masd_epoch = masd_sum / max(1, masd_count)
    nsd_epoch  = nsd_sum  / max(1, nsd_count)
    mdice = dice_sum / dice_count
    miou  = iou_sum  / iou_count

    print(f"[TEST] loss={avg_loss:.4f} mIoU={miou:.4f} mDice~={mdice:.4f} MASD={masd_epoch:.4f} NSD={nsd_epoch:.4f} time={elapsed:.1f}s")

    # Save artifacts
    torch.save(cm.detach().cpu(), run_dir / "confusion_matrix.pt")
    with open(run_dir / "metrics_test.json", "w") as f:
        json.dump({
            "loss": avg_loss,
            "mIoU": miou,
            "mDice~": mdice,
            "MASD": masd_epoch,
            "NSD": nsd_epoch,
            "elapsed_sec": elapsed,
            "num_examples_saved": ex_count,
            "tau": args.tau,
            "ckpt": str(ckpt_path),
            "ignore_val": IGNORE_VAL,
            "binarize_mask": bool(args.binarize_mask)
        }, f, indent=2)

    _save_epoch_csv(run_dir, {
        "split": "test",
        "loss": avg_loss,
        "mIoU": miou,
        "mDice": mdice,
        "MASD": masd_epoch,
        "NSD": nsd_epoch,
        "epoch_secs": elapsed
    })

    with open(run_dir / "examples_index.json", "w") as f:
        json.dump(saved_meta, f, indent=2)

    with open(run_dir / "summary.txt", "w") as f:
        f.write(
            f"TEST results\n"
            f"mIoU={miou:.4f}  mDice~={mdice:.4f}  MASD={masd_epoch:.4f}  NSD={nsd_epoch:.4f}\n"
            f"loss={avg_loss:.4f}  images={len(ds_test)}  tau={args.tau}\n"
            f"ckpt={ckpt_path}\n"
            f"ignore_val={IGNORE_VAL}  binarize_mask={args.binarize_mask}\n"
        )


if __name__ == "__main__":
    main()

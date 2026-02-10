
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import csv, json, time
from datetime import datetime

from models.tripath import TriPathUNet,TriPathUNetStacked,TriPathDSConv,TriPathCNN,TriPathDC
from models.segformer import SegFormer
from models.unet import UNet
from data.dsad_dataset import DSADDataset
from utils.metrics import DiceCELoss, compute_confusion_matrix, iou_from_cm, dice_from_logits,compute_surface_metrics_batch  
from utils.metrics import iou_from_preds, dice_from_preds
from utils.common import set_seed, count_trainable_parameters

def parse_args():
    ap = argparse.ArgumentParser("Train UNet on DSAD-like dataset (images + index masks).")
    ap.add_argument("--data-root", type=str, required=True, help="Root folder containing images/ and masks/")
    ap.add_argument("--images-subdir", type=str, default="images")
    ap.add_argument("--masks-subdir", type=str, default="masks")
    ap.add_argument("--list-train", type=str, default=None, help="Optional train list file (one stem per line)")
    ap.add_argument("--list-val", type=str, default=None, help="Optional val list file")
    ap.add_argument("--img-size", type=int, nargs=2, default=[512,512], help="W H")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--ignore-bg", action="store_true", help="Ignore class 0 in metrics/loss Dice term")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-split", type=float, default=0.1, help="Used only if no list files provided")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision")
    ap.add_argument("--save-dir", type=str, default="./experiments/unet_dsad")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", type=str, default=None,
                    help="Run name used for log/ckpt folders; defaults to timestamp")
    ap.add_argument("--log-interval", type=int, default=0,
                    help="If >0, log per-step stats every N steps to JSONL")
    return ap.parse_args()

def build_loaders(args):
    if args.list_train or args.list_val:
        ds_train = DSADDataset(args.data_root, args.images_subdir, args.masks_subdir, list_file=args.list_train, img_size=args.img_size, augment=True)
        ds_val   = DSADDataset(args.data_root, args.images_subdir, args.masks_subdir, list_file=args.list_val,   img_size=args.img_size, augment=False)
    else:
        ds_all = DSADDataset(args.data_root, args.images_subdir, args.masks_subdir, img_size=args.img_size, augment=True)
        n_total = len(ds_all)
        n_val = max(1, int(n_total * args.val_split))
        n_train = n_total - n_val
        ds_train, ds_val = random_split(ds_all, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return dl_train, dl_val

# ADD:
def _nowstamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _make_loggers(save_dir, run_name):
    """Create CSV (epoch) and JSONL (step) writers. Returns dict with handles and run dir."""
    run_dir = Path(save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # epoch CSV
    csv_f = open(run_dir / "metrics_epoch.csv", "a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "epoch", "split", "loss", "mIoU", "mDice","MASD","NSD","lr", "epoch_secs"
    ])
    if csv_f.tell() == 0:
        csv_w.writeheader()

    # per-step JSONL (only written if --log-interval > 0)
    jsonl_f = open(run_dir / "metrics_step.jsonl", "a")

    # save args snapshot once (created in main)
    return {"dir": run_dir, "csv_f": csv_f, "csv_w": csv_w, "jsonl_f": jsonl_f}

def _log_epoch(loggers, epoch, split, loss, miou, mdice, masd,nsd, lr, epoch_secs):
    row = {
        "epoch": int(epoch),
        "split": split,
        "loss": None if loss is None else float(loss),
        "mIoU": None if miou is None else float(miou),
        "mDice": None if mdice is None else float(mdice),
        "MASD": None if masd is None else float(masd),
        "NSD": None if nsd is None else float(nsd),
        "lr": None if lr is None else float(lr),
        "epoch_secs": None if epoch_secs is None else float(epoch_secs),
    }
    loggers["csv_w"].writerow(row)
    loggers["csv_f"].flush()

def _log_step(loggers, payload: dict):
    loggers["jsonl_f"].write(json.dumps(payload) + "\n")
    loggers["jsonl_f"].flush()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
        # ADD:
    args.run_name = args.run_name or _nowstamp()
    run_root = Path(args.save_dir) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    logs = _make_loggers(args.save_dir, args.run_name)

    # snapshot args to JSON for reproducibility
    with open(run_root / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    best_ckpt = run_root / "best.ckpt"   # move best.ckpt into run dir


    dl_train, dl_val = build_loaders(args)
    
    ####### !!!!Change Network here
    # net = TriPathUNetStacked(in_ch=3, num_classes=args.num_classes, base_ch=32) 
    # net = SegFormer(num_classes=args.num_classes, variant="b2")
    net = TriPathDC(in_ch=3, num_classes=args.num_classes, base_ch=32)
    #net = UNet(in_ch=3, num_classes=args.num_classes, base_ch=64) 
    net.to(device)
    print(f"Model params: {count_trainable_parameters(net)/1e6:.2f}M")

    ignore_index = 255 if not args.ignore_bg else 0
    criterion = DiceCELoss(num_classes=args.num_classes, ignore_index=255, ce_weight=1.0, dice_weight=1.0)
    #criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(10, args.epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_miou = 0.0
    best_ckpt = Path(args.save_dir) / "best_TriPath.ckpt"          #####!!! Change the best file name here. 
    max_grad_norm = 5.0  # or any value you want, e.g. 1.0–10.0

    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        cur_lr = opt.param_groups[0]["lr"]

        net.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs} [train]")
        running_loss = 0.0
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = net(imgs)
                masks = masks.clamp(min=0, max=args.num_classes - 1)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())
            if args.log_interval > 0 and (pbar.n % args.log_interval == 0):
                _log_step(logs, {
                    "phase": "train",
                    "epoch": int(epoch),
                    "step_in_epoch": int(pbar.n),
                    "loss": float(loss.item()),
                    "lr": float(cur_lr),
                    "batch_size": int(imgs.size(0))
                })

        sched.step()
        train_loss = running_loss / len(dl_train.dataset)
        # epoch-level train logging (miou/mdice unknown here)
        prelim_secs = time.time() - epoch_start
        _log_epoch(logs, epoch, "train", train_loss, None, None,None,None, cur_lr, prelim_secs)

        # Validation
        net.eval()
        cm = torch.zeros(args.num_classes, args.num_classes, device=device)
        val_loss = 0.0
        # NEW: accumulators for epoch-level surface metrics
        masd_sum, nsd_sum = 0.0, 0.0
        masd_count, nsd_count = 0, 0
        dice_sum, iou_sum = 0.0,0.0
        dice_count, iou_count = 0,0
        with torch.no_grad():
            for imgs, masks, _ in tqdm(dl_val, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = net(imgs)
                masks = masks.clamp(min=0, max=args.num_classes - 1)
                val_loss += criterion(logits, masks).item() * imgs.size(0)
                preds = logits.argmax(1)
                cm += compute_confusion_matrix(preds, masks, args.num_classes)
                batch_masd, batch_nsd, _ = compute_surface_metrics_batch(
                    preds, masks, num_classes=args.num_classes, tau=2, class_exclude=0,
                pixel_ignore=255)

                if np.isfinite(batch_masd):
                    masd_sum += float(batch_masd)
                    masd_count +=1 

                nsd_sum += float(batch_nsd)
                nsd_count  += 1  

                batch_dice = dice_from_preds(preds, masks, num_classes=args.num_classes,
                                            ignore_index=0)
                batch_iou  = iou_from_preds(preds, masks, num_classes=args.num_classes,
                                            ignore_index=0)
                dice_sum += batch_dice
                iou_sum  += batch_iou
                dice_count += 1
                iou_count += 1                

        val_loss /= len(dl_val.dataset)
        # miou = iou_from_cm(cm, ignore_index=ignore_index).item()
        # mdice = dice_from_logits(logits, masks, args.num_classes, ignore_index=ignore_index).item()  # last batch dice (approximate)
       
        masd_epoch = (masd_sum / max(masd_count, 1))
        nsd_epoch  = (nsd_sum  / max(nsd_count,  1))
        mdice = dice_sum / dice_count
        miou  = iou_sum  / iou_count
        
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"mIoU={miou:.4f} mDice~={mdice:.4f} MASD={masd_epoch:.4f} NSD={nsd_epoch:.4f}")

        
        # epoch-level val logging (finalize epoch time)
        epoch_secs = time.time() - epoch_start
        _log_epoch(logs, epoch, "val", val_loss, miou, mdice, masd_epoch, nsd_epoch, cur_lr, epoch_secs)

        # Save latest
        torch.save({
            "epoch": epoch,
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "args": vars(args),
            "miou": miou,
            "masd": masd_epoch,
            "nsd_surface_dice": nsd_epoch,         
        }, run_root / "latest.ckpt")


        # Save best
        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model": net.state_dict(),
                "miou": miou,
                "args": vars(args),
            }, best_ckpt)
            print(f"** New best mIoU: {best_miou:.4f}. Saved to {best_ckpt}")

    print(f"Training done. Best mIoU={best_miou:.4f}. Best ckpt: {best_ckpt}")
    logs["csv_f"].close()
    logs["jsonl_f"].close()
    print(f"Logs saved to {logs['dir']}")
    with open(run_root / "training_summary.txt", "a") as f:
        f.write(
            f"New best mIoU: {best_miou:.4f}. "
            f"Model params: {count_trainable_parameters(net)/1e6:.2f}M "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"mIoU={miou:.4f} mDice~={mdice:.4f}"
            f"MASD={masd_epoch:.4f} NSD={nsd_epoch:.4f}\n"
        )


if __name__ == "__main__":
    main()

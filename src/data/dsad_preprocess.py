
from pathlib import Path
import argparse, re, shutil
from PIL import Image

IMG_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}
MASK_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

def parse_set(s: str):
    if not s: return set()
    parts = re.split(r"[,\s]+", s.strip().strip("{}"))
    return {p for p in parts if p}

def parse_id_set(s: str):
    if not s: return set()
    parts = re.split(r"[,\s]+", s.strip().strip("{}"))
    ids = set()
    for p in parts:
        p = p.strip()
        if p.isdigit():
            ids.add(int(p))
    return ids

def is_image(fn: Path): return fn.suffix.lower() in IMG_EXTS
def is_mask(fn: Path):  return fn.suffix.lower() in MASK_EXTS

def find_source_folders(src_root: Path):
    # immediate subfolders only
    return [p for p in src_root.iterdir() if p.is_dir()]

def folder_matches(folder: Path, id_set, name_set):
    name = folder.name
    # exact name match
    if name in name_set: return True
    # numeric id inside name
    m = re.search(r"(\d+)", name)
    if m and int(m.group(1)) in id_set: return True
    # if folder name itself is a number
    if name.isdigit() and int(name) in id_set: return True
    return False

def index_from_filename(fname: str, img_prefix: str, mask_prefix: str):
    base = Path(fname).stem.lower()
    # try mask first
    m = re.match(rf"^{re.escape(mask_prefix.lower())}[_-]?(\d+)$", base)
    if m: return int(m.group(1))
    m = re.match(rf"^{re.escape(img_prefix.lower())}[_-]?(\d+)$", base)
    if m: return int(m.group(1))
    # try generic trailing digits
    m = re.search(r"(\d+)$", base)
    return int(m.group(1)) if m else None

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def save_mask_as_png(src: Path, dst_png: Path, overwrite: bool):
    if dst_png.exists() and not overwrite:
        return
    img = Image.open(src)
    if img.mode != "L":
        # take first channel or convert; avoid palette loss surprises
        img = img.convert("L")
    img.save(dst_png)

def copy_image(src: Path, dst: Path, overwrite: bool):
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def move_file(src: Path, dst: Path, overwrite: bool):
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def main():
    ap = argparse.ArgumentParser("Flatten DSAD folders into images/ and masks/ with unique names")
    ap.add_argument("--src-root", required=True, type=str, help="Root containing many folders (each has image*/mask* files)")
    ap.add_argument("--train", default="", type=str, help="Comma/space list of numeric folder IDs, e.g., '4,5,8'")
    ap.add_argument("--val",   default="", type=str, help="Comma/space list of numeric folder IDs, e.g., '2,7,11'")
    ap.add_argument("--train-names", default="", type=str, help="Optional exact folder names for train")
    ap.add_argument("--val-names",   default="", type=str, help="Optional exact folder names for val")
    ap.add_argument("--img-prefix", default="image", type=str, help="Image filename prefix")
    ap.add_argument("--mask-prefix", default="mask",  type=str, help="Mask filename prefix")
    ap.add_argument("--image-out", required=True, type=str, help="Target images directory")
    ap.add_argument("--mask-out",  required=True, type=str, help="Target masks directory")
    ap.add_argument("--lists-out", default="./splits", type=str, help="Where to write train.txt and val.txt")
    ap.add_argument("--stem_format", default="{folder}_{index:06d}", type=str, help="New filename stem format")
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    img_out = Path(args.image_out); mask_out = Path(args.mask_out)
    ensure_dir(img_out); ensure_dir(mask_out)
    lists_out = Path(args.lists_out); ensure_dir(lists_out)

    train_ids = parse_id_set(args.train)
    val_ids   = parse_id_set(args.val)
    train_names = parse_set(args.train_names)
    val_names   = parse_set(args.val_names)

    # Select folders
    folders = find_source_folders(src_root)
    train_folders = [f for f in folders if folder_matches(f, train_ids, train_names)]
    val_folders   = [f for f in folders if folder_matches(f, val_ids,   val_names)]

    # Process a split
    def process_split(split_folders, split_name):
        stems = []
        op = move_file if args.move else copy_image
        for folder in sorted(split_folders):
            # Build index maps inside this folder
            imgs_by_idx, masks_by_idx = {}, {}
            for p in folder.iterdir():
                if not p.is_file(): continue
                idx = index_from_filename(p.name, args.img_prefix, args.mask_prefix)
                if idx is None: continue
                if is_image(p) and p.name.lower().startswith(args.img_prefix.lower()):
                    imgs_by_idx[idx] = p
                elif is_mask(p) and p.name.lower().startswith(args.mask_prefix.lower()):
                    masks_by_idx[idx] = p
            # Pair by common indices
            common = sorted(set(imgs_by_idx.keys()) & set(masks_by_idx.keys()))
            for idx in common:
                src_img = imgs_by_idx[idx]
                src_msk = masks_by_idx[idx]
                stem = args.stem_format.format(folder=folder.name, index=idx)
                # image: keep original extension
                img_dst = img_out / f"{stem}{src_img.suffix.lower()}"
                # mask: force PNG 8-bit
                msk_dst = mask_out / f"{stem}.png"
                op(src_img, img_dst, args.overwrite)
                save_mask_as_png(src_msk, msk_dst, args.overwrite)
                stems.append(stem)
        # write list file (stems only)
        list_path = lists_out / f"{split_name}.txt"
        list_path.write_text("\n".join(sorted(stems)) + ("\n" if stems else ""))
        return list_path, len(stems)

    train_list, n_train = process_split(train_folders, "train")
    val_list,   n_val   = process_split(val_folders,   "val")

    print(f"Train: {n_train} pairs -> {train_list}")
    print(f"Val:   {n_val} pairs -> {val_list}")
    print("Done.")

if __name__ == "__main__":
    main()

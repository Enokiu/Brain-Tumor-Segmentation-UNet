# build_masks_from_via.py
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
import warnings

def _as_region_list(regions):
    """regions can be a list or a dict."""
    if isinstance(regions, dict):
        return list(regions.values())
    return regions or []

def regions_to_mask(h, w, regions):
    """Builds binary mask (0/1) from VIA regions (polygon + ellipse)."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for r in _as_region_list(regions):
        sa = r.get("shape_attributes", {}) or {}
        name = sa.get("name", "")
        if name == "polygon":
            xs = sa.get("all_points_x", []) or []
            ys = sa.get("all_points_y", []) or []
            if xs and ys and len(xs) == len(ys):
                draw.polygon(list(zip(xs, ys)), outline=1, fill=1)
        elif name == "ellipse":
            cx = sa.get("cx"); cy = sa.get("cy")
            rx = sa.get("rx"); ry = sa.get("ry")
            theta = sa.get("theta", 0.0) or 0.0  
            if None not in (cx, cy, rx, ry):
                # Approximate an ellipse as a polygon
                pts = []
                for k in range(120):
                    t = 2*np.pi*k/120.0
                    x = rx*np.cos(t); y = ry*np.sin(t)
                    xr = x*np.cos(theta) - y*np.sin(theta)
                    yr = x*np.sin(theta) + y*np.cos(theta)
                    pts.append((cx + xr, cy + yr))
                draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

def build_masks_for_split(split_dir: Path, ann_file: Path, mask_dir: Path, overwrite=False):
    if not ann_file.exists():
        warnings.warn(f"No annotations found: {ann_file}")
        return {"saved":0, "skipped_exist":0, "missing_img":0}

    data = json.loads(ann_file.read_text(encoding="utf-8"))
    recs = [v for v in data.values() if "filename" in v]

    mask_dir.mkdir(parents=True, exist_ok=True)

    saved = skipped_exist = missing_img = 0
    for rec in recs:
        filename = rec.get("filename")
        if not filename: 
            continue

        # search for image file in split_dir
        img_path = next(iter((split_dir).rglob(filename)), None)
        if not img_path or not img_path.exists():
            missing_img += 1
            continue

        out_path = mask_dir / (Path(filename).stem + ".png")
        if out_path.exists() and not overwrite:
            skipped_exist += 1
            continue

        # Read image size, render mask, save (0/255)
        img = Image.open(img_path).convert("L")
        h, w = np.asarray(img).shape
        mask01 = regions_to_mask(h, w, rec.get("regions", []))
        Image.fromarray((mask01*255).astype(np.uint8)).save(out_path)
        saved += 1

    print(f"✔ {split_dir.name}: saved {saved} | skipped_exist {skipped_exist} | missing_img {missing_img}")
    return {"saved":saved, "skipped_exist":skipped_exist, "missing_img":missing_img}

# Main function to build masks for all splits
def build_all_masks(
        root="brain_tumor_dataset/dataset",
        splits=("train","val","test"),
        ann_name="annotations.json",
        mask_subdir="masks",
        overwrite=False
    ):

    root = Path(root)
    totals = {"saved":0,"skipped_exist":0,"missing_img":0}
    for s in splits:
        split_dir = root / s
        ann_file  = split_dir / ann_name
        mask_dir  = split_dir / mask_subdir
        stats = build_masks_for_split(split_dir, ann_file, mask_dir, overwrite=overwrite)
        for k in totals: totals[k] += stats[k]
    print(f"==> Total: saved {totals['saved']} | skipped_exist {totals['skipped_exist']} | missing_img {totals['missing_img']}")

from pathlib import Path
import json, shutil, random
import numpy as np
from PIL import Image, ImageDraw

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def scan_images(root: Path):
    """Walk root and map filename -> absolute path (works if names are mostly unique)."""
    m = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in EXTS:
            m.setdefault(p.name, p.resolve())
    return m

def load_via_records(json_paths):
    """Read one or more VIA JSON files and return a list of records (dicts)."""
    recs = []
    for jp in map(Path, json_paths):
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        recs.extend([v for v in data.values() if "filename" in v])
    return recs

def regions_to_mask_area(h, w, regions):
    """Return total mask area (pixels) for VIA polygons/ellipses only."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for r in regions:
        sa = r.get("shape_attributes", {})
        name = sa.get("name")
        if name == "polygon":
            xs, ys = sa.get("all_points_x", []), sa.get("all_points_y", [])
            if xs and ys and len(xs) == len(ys):
                draw.polygon(list(zip(xs, ys)), outline=1, fill=1)
        elif name == "ellipse":
            cx, cy = sa.get("cx"), sa.get("cy")
            rx, ry = sa.get("rx"), sa.get("ry")
            theta  = sa.get("theta", 0.0) or 0.0
            pts = []
            for k in range(120):
                t = 2*np.pi*k/120.0
                x = rx*np.cos(t); y = ry*np.sin(t)
                xr = x*np.cos(theta) - y*np.sin(theta)
                yr = x*np.sin(theta) + y*np.cos(theta)
                pts.append((cx + xr, cy + yr))
            draw.polygon(pts, outline=1, fill=1)
    return int(np.array(mask, dtype=np.uint8).sum())

def make_unique_name(dst_dir: Path, name: str):
    """If name exists in dst_dir, append __1, __2, ... before extension."""
    stem, ext = Path(name).stem, Path(name).suffix
    k, new = 1, name
    while (dst_dir / new).exists():
        new = f"{stem}__{k}{ext}"; k += 1
    return new

def via_entry_for(dst_img: Path, rec: dict):
    """Build a VIA entry with adjusted filename+size key."""
    size = dst_img.stat().st_size
    key = f"{dst_img.name}{size}"
    return key, {
        "filename": dst_img.name,
        "size": size,
        "regions": rec.get("regions", []),
        "file_attributes": rec.get("file_attributes", {}),
    }

def integer_targets(n, ratios):
    """Integer targets per split that sum to n (largest remainders)."""
    r = np.array(ratios, dtype=float)
    r = r / r.sum()
    ideal = r * n
    base  = np.floor(ideal).astype(int)
    rem   = n - base.sum()
    order = np.argsort(-(ideal - base))  # assign leftover to biggest fractional parts
    for i in range(rem):
        base[order[i % len(base)]] += 1
    return base.tolist()

# Main function to split dataset into size buckets and then into train/val/test splits
def split_size_bucket_then_split(
    src_root,
    json_paths,
    out_root="brain_tumor_dataset/dataset",
    ratios=(0.8, 0.1, 0.1),        # train/val/test targets overall
    q_small=0.50,                  # <= q_small  -> small
    q_large=0.90,                  # > q_large   -> large  (rest = medium)
    train_boost_large=0.05,        # give train +5% extra of LARGE (taken from val/test)
    seed=42,
    buckets_debug_root=None,       
):
    """
    1) Compute tumor area per annotated image.
    2) Bucket into small/medium/large using quantiles.
    3) Distribute EACH bucket across train/val/test using ratios, but with an
       optional 'train_boost_large' applied to the LARGE bucket.
    4) Copy images into out_root/<split>/ and write VIA 'annotations.json' per split.
    """
    rng = random.Random(seed)
    src_root  = Path(src_root)
    out_root  = Path(out_root);  out_root.mkdir(parents=True, exist_ok=True)
    img_index = scan_images(src_root)
    records   = load_via_records(json_paths)

    # Only annotated images with an existing file
    records = [r for r in records if r.get("filename") in img_index and len(r.get("regions", [])) > 0]
    if not records:
        raise RuntimeError("No annotated records with matching images found.")

    # Tumor area per image
    sizes = []
    for r in records:
        img_path = img_index[r["filename"]]
        img = Image.open(img_path).convert("L")
        h, w = np.asarray(img).shape
        sizes.append(regions_to_mask_area(h, w, r.get("regions", [])))
    sizes = np.array(sizes)

    # Bucket thresholds
    qs = np.quantile(sizes, [0.0, q_small, q_large, 1.0])
    small_thr, large_thr = qs[1], qs[2]

    # Assign bucket id per sample
    buckets = {"small": [], "medium": [], "large": []}
    for i, s in enumerate(sizes):
        if s <= small_thr:
            buckets["small"].append(i)
        elif s > large_thr:
            buckets["large"].append(i)
        else:
            buckets["medium"].append(i)

    # write debug bucket folders
    if buckets_debug_root:
        dbg = Path(buckets_debug_root); dbg.mkdir(parents=True, exist_ok=True)
        for bname, idxs in buckets.items():
            bdir = dbg / bname
            bdir.mkdir(parents=True, exist_ok=True)
            sub_json = {}
            for i in idxs:
                rec = records[i]
                src_img = img_index[rec["filename"]]
                dst_name = make_unique_name(bdir, rec["filename"])
                dst_img  = bdir / dst_name
                shutil.copy2(src_img, dst_img)
                key, entry = via_entry_for(dst_img, rec)
                sub_json[key] = entry
            with (bdir / "annotations.json").open("w", encoding="utf-8") as f:
                json.dump(sub_json, f, indent=2)

    # Split per bucket
    split_names = ["train", "val", "test"]
    chosen = {s: [] for s in split_names}

    # Base ratios
    base = np.array(ratios, dtype=float)

    for bname, idxs in buckets.items():
        rng.shuffle(idxs)
        r = base.copy()
        if bname == "large" and train_boost_large != 0:
            # shift from val/test equally to train (keep non-negative)
            delta = float(train_boost_large)
            r[0] += delta
            leftover = max(1e-9, r[1] + r[2])  # avoid divide-by-zero
            r[1] -= delta * (r[1] / leftover)
            r[2] -= delta * (r[2] / leftover)
            r = np.clip(r, 0, None)
        counts = integer_targets(len(idxs), r)
        start = 0
        for s, need in zip(split_names, counts):
            chosen[s].extend(idxs[start:start+need])
            start += need

    # Copy & write VIA per split
    for s in split_names:
        split_dir = out_root / s
        split_dir.mkdir(parents=True, exist_ok=True)
        sub_json = {}
        for i in chosen[s]:
            rec = records[i]
            src_img = img_index[rec["filename"]]
            dst_name = make_unique_name(split_dir, rec["filename"])
            dst_img  = split_dir / dst_name
            shutil.copy2(src_img, dst_img)
            key, entry = via_entry_for(dst_img, rec)
            sub_json[key] = entry
        with (split_dir / "annotations.json").open("w", encoding="utf-8") as f:
            json.dump(sub_json, f, indent=2)

    # Summary
    print(f"Buckets (<= {int(small_thr)} = small, > {int(large_thr)} = large):")
    for b in ["small", "medium", "large"]:
        print(f"  {b:6s}: {len(buckets[b])}")
    for s in split_names:
        # bucket counts inside each split
        counts = {"small":0, "medium":0, "large":0}
        for i in chosen[s]:
            if i in buckets["small"]:  counts["small"]  += 1
            elif i in buckets["large"]: counts["large"] += 1
            else:                       counts["medium"] += 1
        print(f"{s:5s} | total {len(chosen[s]):4d}  | small {counts['small']:4d}  "
              f"medium {counts['medium']:4d}  large {counts['large']:4d}  "
              f"| thr_small {int(small_thr)}, thr_large {int(large_thr)}")

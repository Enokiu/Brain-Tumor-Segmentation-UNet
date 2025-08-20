from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def regions_to_area(h, w, regions):
    """Return mask area (pixel count) from VIA regions (polygon + ellipse)."""
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for r in regions:
        sa = r.get("shape_attributes", {})
        name = sa.get("name")
        if name == "polygon":
            xs, ys = sa.get("all_points_x", []), sa.get("all_points_y", [])
            if xs and ys and len(xs) == len(ys):
                d.polygon(list(zip(xs, ys)), outline=1, fill=1)
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
            d.polygon(pts, outline=1, fill=1)
    return int(np.array(m, dtype=np.uint8).sum())

def collect_areas(base_dir, splits=("train","val","test")):
    """Read annotations.json in each split and return list of areas per split."""
    base = Path(base_dir)
    areas = {s: [] for s in splits}
    for s in splits:
        jp = base/s/f"annotations.json"
        if not jp.exists():
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        for rec in data.values():
            imgp = next((base/s).rglob(rec["filename"]), None)
            if not imgp or not imgp.exists():
                continue
            arr = np.array(Image.open(imgp).convert("L"))
            h, w = arr.shape
            areas[s].append(regions_to_area(h, w, rec.get("regions", [])))
    return areas

def plot_tumor_size_distribution(
    base_dir="brain_tumor_dataset/dataset",
    splits=("train","val","test"),
    thr_small=None, thr_large=None,     
    auto_q=(0.60, 0.85),                
    mode="grouped"                     
    ):
    areas = collect_areas(base_dir, splits)
    all_areas = np.array(sum(areas.values(), []))
    assert len(all_areas) > 0, "no masks found"

    # thresholds
    if thr_small is None or thr_large is None:
        thr_small = float(np.quantile(all_areas, auto_q[0]))
        thr_large = float(np.quantile(all_areas, auto_q[1]))

    counts = {}  # split -> (small, medium, large)
    for s in splits:
        a = np.array(areas[s])
        small  = int((a <= thr_small).sum())
        large  = int((a >  thr_large).sum())
        medium = int(len(a) - small - large)
        counts[s] = (small, medium, large)
        print(f"{s:5s} | total {len(a):4d} | small {small:3d} | medium {medium:3d} | large {large:3d} "
              f"(thr_small={int(thr_small)}, thr_large={int(thr_large)})")

    # plot
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(splits))
    if mode == "grouped":
        w = 0.25
        b1 = ax.bar(x - w, [counts[s][0] for s in splits], width=w, label="small")
        b2 = ax.bar(x      , [counts[s][1] for s in splits], width=w, label="medium")
        b3 = ax.bar(x + w, [counts[s][2] for s in splits], width=w, label="large")
        for bars in (b1,b2,b3):
            for r in bars:
                ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.5, str(int(r.get_height())),
                        ha="center", va="bottom", fontsize=9)
    else:  
        small = [counts[s][0] for s in splits]
        medium= [counts[s][1] for s in splits]
        large = [counts[s][2] for s in splits]
        b1 = ax.bar(x, small, label="small")
        b2 = ax.bar(x, medium, bottom=small, label="medium")
        b3 = ax.bar(x, large, bottom=np.array(small)+np.array(medium), label="large")
        totals = np.array(small)+np.array(medium)+np.array(large)
        for i, t in enumerate(totals):
            ax.text(i, t+0.5, str(int(t)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("Image count")          
    ax.set_title("Tumor size distribution per split")
    ax.legend()
    plt.tight_layout(); plt.show()


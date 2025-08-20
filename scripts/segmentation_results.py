from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# colors for overlay
purple     = (0.60, 0.20, 0.80)   # GT
light_blue = (0.30, 0.80, 1.00)   # Pred

# binary dice and IoU metrics
def dice_bin(pred, target, thr=0.5, eps=1e-6):
    pred = (pred > thr).float()
    num  = 2 * (pred * target).sum(dim=(1,2,3))
    den  = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (num/den).mean().item()

def iou_bin(pred, target, thr=0.5, eps=1e-6):
    pred = (pred > thr).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter + eps
    return (inter/union).mean().item()

# overlay helper
def overlay_rgba(mask, color, alpha=0.35):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3]  = (mask > 0).astype(np.float32) * alpha
    return rgba

def report(model, test_loader, tag="unet_base", save_root="reports",
                       device=None, max_samples=6, thr=0.5, show=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    out_dir = Path(save_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    shown, dices, ious = 0, [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            p = model(x)

            dice = dice_bin(p, y, thr=thr)
            iou  = iou_bin(p, y, thr=thr)
            dices.append(dice); ious.append(iou)

            img       = x[0].detach().cpu().squeeze().numpy()
            true_mask = (y[0].detach().cpu().squeeze().numpy() > 0.5)
            pred_mask = (p[0].detach().cpu().squeeze().numpy() > thr)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(img, cmap="gray")
            axes[1].imshow(overlay_rgba(true_mask.astype(np.uint8), purple, 0.35))
            axes[1].contour(true_mask, colors=[purple], linewidths=1)
            axes[1].set_title("Ground Truth (purple)"); axes[1].axis("off")
            axes[2].imshow(img, cmap="gray")
            axes[2].imshow(overlay_rgba(pred_mask.astype(np.uint8), light_blue, 0.35))
            axes[2].contour(pred_mask, colors=[light_blue], linewidths=1)
            axes[2].set_title(f"Prediction (blue)\nDice {dice:.3f} | IoU {iou:.3f}")
            axes[2].axis("off")
            plt.tight_layout()

            if show:
                plt.show()
            fig.savefig(out_dir / f"{tag}_pred_{shown:03d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            shown += 1
            if shown >= max_samples:
                break

    avg_dice = float(np.mean(dices)) if dices else 0.0
    avg_iou  = float(np.mean(ious)) if ious else 0.0
    print(f"[{tag}] avg_dice {avg_dice:.4f} | avg_iou {avg_iou:.4f} → saved in {out_dir}")
    return {"avg_dice": avg_dice, "avg_iou": avg_iou}

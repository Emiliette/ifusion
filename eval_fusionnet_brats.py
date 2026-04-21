import argparse
import csv
import os
import time
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from util.brats_2d_dataset import BraTS2DPngDataset
from util.fusion_losses import grad_mag, ssim_torch
from util.fusion_objective_metrics import (
    fsim01,
    mutual_information_u8,
    psnr01,
    q_cv_chen_varshney,
    q_ncie_wang,
    q_s_piella,
    vifp,
)
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


def _entropy_u8(img_u8: np.ndarray) -> float:
    hist = np.bincount(img_u8.reshape(-1), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * (np.log2(p + 1e-12))).sum())


def save_uint8_png(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate trained FlexibleFusionNet on BraTS and export metrics CSV")
    p.add_argument("--brats_2d_root", type=str, required=True, help="2D PNG dataset root (root/<mod>/<case>_z###.png).")
    p.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints_brats", "latest.pth"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feat_channels", type=int, default=32)

    p.add_argument("--modalities", type=str, nargs="+", default=["t1n", "t1c", "t2w", "t2f"])
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--qcv_window", type=int, default=16)
    p.add_argument("--qcv_alpha", type=int, default=5)
    p.add_argument("--skip_fsim", action="store_true", help="Skip FSIM (slower).")
    p.add_argument("--limit_slices", type=int, default=0)

    p.add_argument("--out_csv", type=str, default="./fusionnet_metrics.csv")
    p.add_argument("--save_fused", action="store_true", help="Also save fused PNGs under --out_dir/recon/")
    p.add_argument("--out_dir", type=str, default="./result_fusionnet_eval/")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"BraTS 2D root: {os.path.abspath(args.brats_2d_root)}")

    modalities = [m.strip().lower() for m in args.modalities]
    if len(modalities) < 2 or len(modalities) > 4:
        raise SystemExit("--modalities must have 2..4 items.")
    for m in modalities:
        if m not in {"t1n", "t1c", "t2w", "t2f"}:
            raise SystemExit("Modalities must be among: t1n t1c t2w t2f")

    modality_to_id = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
    mod_ids = torch.tensor([modality_to_id[m] for m in modalities], dtype=torch.long, device=device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    feat_channels = int(ckpt_args.get("feat_channels", args.feat_channels))
    model = FlexibleFusionNet(feat_channels=feat_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    dataset_2d = BraTS2DPngDataset(
        args.brats_2d_root,
        modalities=modalities,
        scale=int(args.scale),
        limit_slices=int(args.limit_slices),
    )
    logger.info(f"Aligned 2D slices: {len(dataset_2d)}")

    if args.save_fused:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "recon"), exist_ok=True)

    metric_cols = []
    for m in modalities:
        metric_cols.extend([f"MI_{m}", f"PSNR_{m}", f"SSIM_{m}", f"VIF_{m}", f"FSIM_{m}"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    t0 = time.time()
    rows = []

    scale = int(args.scale)
    qcv_window = int(args.qcv_window)
    skipped_too_small = 0

    for xk_t, case_id, z in tqdm(dataset_2d, desc="Slices"):
        # xk_t: (K,1,H,W) float01
        h, w = int(xk_t.shape[-2]), int(xk_t.shape[-1])
        h_m = h - h % scale
        w_m = w - w % scale
        if h_m < 3 or w_m < 3:
            skipped_too_small += 1
            continue
        xk_t = xk_t[..., :h_m, :w_m]
        x = xk_t.squeeze(1).numpy()  # (K,Hm,Wm)

        with torch.no_grad():
            fused01_t = model(xk_t.unsqueeze(0).to(device), mod_ids=mod_ids)  # (1,1,H,W)
        fused01 = fused01_t.detach().cpu().squeeze(0).squeeze(0).numpy()
        fused_u8 = np.clip(fused01 * 255.0, 0, 255).astype(np.uint8)

        en = _entropy_u8(fused_u8)
        sd = float(np.std(fused01))
        ag = float(grad_mag(fused01_t).mean().detach().cpu())

        mi_vals = []
        psnr_vals = []
        ssim_vals = []
        vif_vals = []
        fsim_vals = []
        per_mod = {}
        fused_cpu = torch.from_numpy(fused01).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W) cpu
        for mi_m, m in enumerate(modalities):
            src01 = x[mi_m]
            m01 = torch.from_numpy(src01).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W) cpu
            src_u8 = np.clip(src01 * 255.0, 0, 255).astype(np.uint8)
            mi = mutual_information_u8(fused_u8, src_u8, bins=256, normalize_log_base=256)
            psnr = psnr01(fused01, src01)
            ssim = float(ssim_torch(fused_cpu, m01).item())
            vif = vifp(src01, fused01)
            fsim = 0.0 if args.skip_fsim else fsim01(src01, fused01)
            mi_vals.append(mi)
            psnr_vals.append(psnr)
            ssim_vals.append(ssim)
            vif_vals.append(vif)
            fsim_vals.append(fsim)
            per_mod[f"MI_{m}"] = mi
            per_mod[f"PSNR_{m}"] = psnr
            per_mod[f"SSIM_{m}"] = ssim
            per_mod[f"VIF_{m}"] = vif
            per_mod[f"FSIM_{m}"] = fsim

        qncie = q_ncie_wang([np.clip(x[i] * 255.0, 0, 255).astype(np.uint8) for i in range(len(modalities))], fused_u8)
        qs = q_s_piella([x[i] for i in range(len(modalities))], fused01)
        h_q = h_m - h_m % qcv_window
        w_q = w_m - w_m % qcv_window
        if h_q >= qcv_window and w_q >= qcv_window:
            qcv = q_cv_chen_varshney(
                [x[i][:h_q, :w_q] for i in range(len(modalities))],
                fused01[:h_q, :w_q],
                window_size=qcv_window,
                alpha=int(args.qcv_alpha),
            )
        else:
            qcv = float("nan")

        row = {
            "case_id": str(case_id),
            "slice": int(z),
            "K": int(len(modalities)),
            "EN": en,
            "MI_mean": float(np.mean(mi_vals)) if mi_vals else 0.0,
            "PSNR_mean": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
            "SSIM_mean": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
            "VIF_mean": float(np.mean(vif_vals)) if vif_vals else 0.0,
            "FSIM_mean": float(np.mean(fsim_vals)) if fsim_vals else 0.0,
            "Q_NCIE": qncie,
            "Q_S": qs,
            "Q_CV": qcv,
            "SD": sd,
            "AG": ag,
            **per_mod,
        }
        rows.append(row)

        if args.save_fused:
            out_path = os.path.join(args.out_dir, "recon", f"{case_id}_z{int(z):03d}.png")
            save_uint8_png(out_path, fused_u8)

    # Write CSV + mean summary row.
    cols = [
        "case_id",
        "slice",
        "K",
        "EN",
        "MI_mean",
        "PSNR_mean",
        "SSIM_mean",
        "VIF_mean",
        "FSIM_mean",
        "Q_NCIE",
        "Q_S",
        "Q_CV",
        "SD",
        "AG",
    ] + metric_cols
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

        if rows:
            mean_row = {"case_id": "MEAN", "slice": "", "K": rows[0]["K"]}
            for c in cols:
                if c in {"case_id", "slice", "K"}:
                    continue
                vals = [float(r.get(c, 0.0)) for r in rows if r.get(c, "") != ""]
                mean_row[c] = float(np.mean(vals)) if vals else 0.0
            w.writerow(mean_row)

    if skipped_too_small:
        logger.info(f"Skipped {skipped_too_small} slices due to too-small size after crop to --scale={scale}")
    logger.info(f"Done. Wrote {len(rows)} rows to {args.out_csv} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

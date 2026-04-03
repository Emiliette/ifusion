import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from util.brats_dataset import AXIS_TO_SLICE, choose_slice_indices, discover_brats_cases, normalize_volume_to_float01, _load_nii
from util.fusion_losses import grad_mag, ssim_torch
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


def _entropy_u8(img_u8: np.ndarray) -> float:
    hist = np.bincount(img_u8.reshape(-1), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * (np.log2(p + 1e-12))).sum())


def _mutual_information_u8(x_u8: np.ndarray, y_u8: np.ndarray, bins: int = 64) -> float:
    x = (x_u8.astype(np.int64) * bins) // 256
    y = (y_u8.astype(np.int64) * bins) // 256
    joint = np.bincount((x.reshape(-1) * bins + y.reshape(-1)), minlength=bins * bins).astype(np.float64).reshape(bins, bins)
    pxy = joint / (joint.sum() + 1e-12)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py + 1e-12
    nz = pxy > 0
    return float((pxy[nz] * np.log2((pxy[nz] / denom[nz]) + 1e-12)).sum())


def _psnr01(x01: torch.Tensor, y01: torch.Tensor) -> float:
    mse = torch.mean((x01 - y01) ** 2).item()
    return float(10.0 * np.log10(1.0 / (mse + 1e-12)))


def save_uint8_png(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate trained FlexibleFusionNet on BraTS and export metrics CSV")
    p.add_argument("--brats_root", type=str, default=os.path.join("Dataset-BraTS", "BraTS2024-BraTS-GLI-TrainingData"))
    p.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints_brats", "latest.pth"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feat_channels", type=int, default=32)

    p.add_argument("--modalities", type=str, nargs="+", default=["t1n", "t1c", "t2w", "t2f"])
    p.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"])
    p.add_argument("--slice_mode", type=str, default="nonzero", choices=["all", "nonzero", "seg_nonzero"])
    p.add_argument("--min_nonzero_frac", type=float, default=0.01)
    p.add_argument("--p_low", type=float, default=0.5)
    p.add_argument("--p_high", type=float, default=99.5)
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--limit_cases", type=int, default=0)
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

    modalities = [m.strip().lower() for m in args.modalities]
    if len(modalities) < 2 or len(modalities) > 4:
        raise SystemExit("--modalities must have 2..4 items.")
    for m in modalities:
        if m not in {"t1n", "t1c", "t2w", "t2f"}:
            raise SystemExit("Modalities must be among: t1n t1c t2w t2f")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    feat_channels = int(ckpt_args.get("feat_channels", args.feat_channels))
    model = FlexibleFusionNet(feat_channels=feat_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    cases_map = discover_brats_cases(args.brats_root)
    case_ids = sorted(cases_map.keys())
    if args.limit_cases and int(args.limit_cases) > 0:
        case_ids = case_ids[: int(args.limit_cases)]

    if args.save_fused:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "recon"), exist_ok=True)

    metric_cols = []
    for m in modalities:
        metric_cols.extend([f"MI_{m}", f"PSNR_{m}", f"SSIM_{m}"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    t0 = time.time()
    rows = []

    slicer = AXIS_TO_SLICE[args.axis]
    for case_id in tqdm(case_ids, desc="Cases"):
        case = cases_map[case_id]
        files = case.paths
        if any(m not in files for m in modalities):
            continue

        vols01: Dict[str, np.ndarray] = {}
        for m in modalities:
            vols01[m] = normalize_volume_to_float01(_load_nii(files[m]), p_low=float(args.p_low), p_high=float(args.p_high))

        seg_vol = _load_nii(files["seg"]) if ("seg" in files and args.slice_mode == "seg_nonzero") else None
        ref_vol = seg_vol if args.slice_mode == "seg_nonzero" else vols01[modalities[0]]
        slice_indices = choose_slice_indices(
            ref_vol,
            axis=args.axis,
            slice_mode=args.slice_mode,
            seg_reference=seg_vol,
            min_nonzero_frac=float(args.min_nonzero_frac),
        )
        if args.limit_slices and int(args.limit_slices) > 0:
            slice_indices = slice_indices[: int(args.limit_slices)]

        for z in slice_indices:
            slices = []
            for m in modalities:
                sl = slicer(vols01[m], int(z))  # float01
                slices.append(np.ascontiguousarray(sl))

            h, w = slices[0].shape
            scale = int(args.scale)
            h = h - h % scale
            w = w - w % scale
            x = np.stack([s[:h, :w] for s in slices], axis=0)  # (K,H,W)
            x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(2).to(device)  # (1,K,1,H,W)

            with torch.no_grad():
                fused01_t = model(x_t)  # (1,1,H,W)
            fused01 = fused01_t.detach().cpu().squeeze(0).squeeze(0).numpy()
            fused_u8 = np.clip(fused01 * 255.0, 0, 255).astype(np.uint8)

            en = _entropy_u8(fused_u8)
            sd = float(np.std(fused01))
            ag = float(grad_mag(fused01_t).mean().detach().cpu())

            mi_vals = []
            psnr_vals = []
            ssim_vals = []
            per_mod = {}
            for mi_m, m in enumerate(modalities):
                m01 = torch.from_numpy(x[mi_m]).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W) cpu
                mi = _mutual_information_u8(fused_u8, np.clip(x[mi_m] * 255.0, 0, 255).astype(np.uint8))
                psnr = _psnr01(torch.from_numpy(fused01).float().unsqueeze(0).unsqueeze(0), m01)
                ssim = float(ssim_torch(torch.from_numpy(fused01).float().unsqueeze(0).unsqueeze(0), m01).item())
                mi_vals.append(mi)
                psnr_vals.append(psnr)
                ssim_vals.append(ssim)
                per_mod[f"MI_{m}"] = mi
                per_mod[f"PSNR_{m}"] = psnr
                per_mod[f"SSIM_{m}"] = ssim

            row = {
                "case_id": case_id,
                "slice": int(z),
                "K": int(len(modalities)),
                "EN": en,
                "MI_mean": float(np.mean(mi_vals)) if mi_vals else 0.0,
                "PSNR_mean": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
                "SSIM_mean": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
                "SD": sd,
                "AG": ag,
                **per_mod,
            }
            rows.append(row)

            if args.save_fused:
                out_path = os.path.join(args.out_dir, "recon", f"{case_id}_z{int(z):03d}.png")
                save_uint8_png(out_path, fused_u8)

    # Write CSV + mean summary row.
    cols = ["case_id", "slice", "K", "EN", "MI_mean", "PSNR_mean", "SSIM_mean", "SD", "AG"] + metric_cols
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

    logger.info(f"Done. Wrote {len(rows)} rows to {args.out_csv} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


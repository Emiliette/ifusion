import argparse
import csv
import os
import time
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from util.brats_2d_dataset import BraTS2DPngDataset
from util.fusion_objective_metrics import (
    avg_gradient_u8,
    entropy_u8,
    fsim01,
    mutual_information_u8,
    normalize1_u8,
    psnr_u8,
    q_cv_chen_varshney,
    q_ncie_wang,
    q_s_piella,
    ssim_u8,
    std_u8,
    vifp_u8,
)
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


ALL_BRATS_MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def save_uint8_png(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate trained FlexibleFusionNet on BraTS across all 4C2, 4C3, 4C4 modality subsets.")
    p.add_argument("--brats_2d_root", type=str, required=True, help="2D PNG dataset root (root/<mod>/<case>_z###.png).")
    p.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints_brats", "latest.pth"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feat_channels", type=int, default=32)

    p.add_argument("--modalities", type=str, nargs="+", default=list(ALL_BRATS_MODALITIES))
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--qcv_window", type=int, default=16)
    p.add_argument("--qcv_alpha", type=int, default=5)
    p.add_argument("--skip_fsim", action="store_true", help="Skip FSIM (slower).")
    p.add_argument("--limit_slices", type=int, default=0)

    p.add_argument("--out_csv", type=str, default="./fusionnet_metrics.csv")
    p.add_argument("--save_fused", action="store_true", help="Also save fused PNGs under --out_dir/recon/")
    p.add_argument("--out_dir", type=str, default="./result_fusionnet_eval/")
    return p.parse_args(argv)


def _validate_modalities(modalities: Sequence[str]) -> list[str]:
    normalized = [m.strip().lower() for m in modalities]
    if len(normalized) != 4:
        raise SystemExit("--modalities must contain exactly 4 BraTS modalities to evaluate all 11 subsets.")
    if len(set(normalized)) != 4:
        raise SystemExit("--modalities must be unique.")
    for m in normalized:
        if m not in ALL_BRATS_MODALITIES:
            raise SystemExit("Modalities must be among: t1n t1c t2w t2f")
    return normalized


def _build_subset_specs(modalities: Sequence[str], device: torch.device) -> list[dict[str, object]]:
    modality_to_id = {m: i for i, m in enumerate(ALL_BRATS_MODALITIES)}
    specs: list[dict[str, object]] = []
    for subset_size in (2, 3, 4):
        for subset in combinations(modalities, subset_size):
            subset_indices = [modalities.index(m) for m in subset]
            subset_mod_ids = [modality_to_id[m] for m in subset]
            specs.append(
                {
                    "name": "+".join(subset),
                    "modalities": list(subset),
                    "indices": subset_indices,
                    "mod_ids": torch.tensor(subset_mod_ids, dtype=torch.long, device=device),
                }
            )
    return specs


def _make_metric_cols(modalities: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for m in modalities:
        cols.extend([f"MI_{m}", f"PSNR_{m}", f"SSIM_{m}", f"VIF_{m}", f"FSIM_{m}"])
    return cols


def _summarize_rows(rows: Sequence[dict[str, object]], cols: Sequence[str], subset_name: str) -> dict[str, object]:
    mean_row: dict[str, object] = {
        "case_id": "MEAN",
        "slice": "",
        "subset": subset_name,
        "K": rows[0]["K"],
    }
    for c in cols:
        if c in {"case_id", "slice", "subset", "K"}:
            continue
        vals = [float(r.get(c, 0.0)) for r in rows if r.get(c, "") != ""]
        mean_row[c] = float(np.mean(vals)) if vals else 0.0
    return mean_row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"BraTS 2D root: {os.path.abspath(args.brats_2d_root)}")

    modalities = _validate_modalities(args.modalities)
    subset_specs = _build_subset_specs(modalities, device)
    logger.info(f"Evaluating {len(subset_specs)} modality subsets: {', '.join(str(spec['name']) for spec in subset_specs)}")

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

    metric_cols = _make_metric_cols(modalities)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    t0 = time.time()
    rows: list[dict[str, object]] = []

    scale = int(args.scale)
    qcv_window = int(args.qcv_window)
    skipped_too_small = 0

    for x4_t, case_id, z in tqdm(dataset_2d, desc="Slices"):
        # x4_t: (4,1,H,W) float01
        h, w = int(x4_t.shape[-2]), int(x4_t.shape[-1])
        h_m = h - h % scale
        w_m = w - w % scale
        if h_m < 3 or w_m < 3:
            skipped_too_small += 1
            continue
        x4_t = x4_t[..., :h_m, :w_m]
        x4 = x4_t.squeeze(1).numpy()  # (4,Hm,Wm)

        for spec in subset_specs:
            subset_indices = list(spec["indices"])
            subset_modalities = list(spec["modalities"])
            mod_ids = spec["mod_ids"]
            subset_name = str(spec["name"])

            subset_t = x4_t.index_select(dim=0, index=torch.tensor(subset_indices, dtype=torch.long))
            subset_x = x4[subset_indices]

            with torch.no_grad():
                fused01_t = model(subset_t.unsqueeze(0).to(device), mod_ids=mod_ids)  # (1,1,H,W)
            fused01 = fused01_t.detach().cpu().squeeze(0).squeeze(0).numpy()
            fused_u8 = normalize1_u8(fused01)

            en = entropy_u8(fused_u8)
            sd = std_u8(fused_u8)
            ag = avg_gradient_u8(fused_u8)

            mi_vals = []
            psnr_vals = []
            ssim_vals = []
            vif_vals = []
            fsim_vals = []
            per_mod = {c: "" for c in metric_cols}

            for mi_m, m in enumerate(subset_modalities):
                src01 = subset_x[mi_m]
                src_u8 = normalize1_u8(src01)
                mi = mutual_information_u8(fused_u8, src_u8, bins=256, normalize_log_base=256)
                psnr = psnr_u8(fused_u8, src_u8)
                ssim = ssim_u8(fused_u8, src_u8)
                vif = vifp_u8(src_u8, fused_u8)
                fsim = 0.0 if args.skip_fsim else fsim01(src_u8.astype(np.float64) / 255.0, fused_u8.astype(np.float64) / 255.0)
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

            src_u8_list = [normalize1_u8(subset_x[i]) for i in range(len(subset_modalities))]
            src01_list = [src_u8.astype(np.float64) / 255.0 for src_u8 in src_u8_list]
            qncie = q_ncie_wang(src_u8_list, fused_u8)
            qs = q_s_piella(src01_list, fused_u8.astype(np.float64) / 255.0)
            h_q = h_m - h_m % qcv_window
            w_q = w_m - w_m % qcv_window
            if h_q >= qcv_window and w_q >= qcv_window:
                qcv = q_cv_chen_varshney(
                    [src01[:h_q, :w_q] for src01 in src01_list],
                    (fused_u8.astype(np.float64) / 255.0)[:h_q, :w_q],
                    window_size=qcv_window,
                    alpha=int(args.qcv_alpha),
                )
            else:
                qcv = float("nan")

            row = {
                "case_id": str(case_id),
                "slice": int(z),
                "subset": subset_name,
                "K": int(len(subset_modalities)),
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
                out_path = os.path.join(args.out_dir, "recon", subset_name, f"{case_id}_z{int(z):03d}.png")
                save_uint8_png(out_path, fused_u8)

    cols = [
        "case_id",
        "slice",
        "subset",
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
            rows_by_subset: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                rows_by_subset.setdefault(str(row["subset"]), []).append(row)
            for subset_name in sorted(rows_by_subset):
                w.writerow(_summarize_rows(rows_by_subset[subset_name], cols, subset_name))
            w.writerow(_summarize_rows(rows, cols, "ALL"))

    if skipped_too_small:
        logger.info(f"Skipped {skipped_too_small} slices due to too-small size after crop to --scale={scale}")
    logger.info(f"Done. Wrote {len(rows)} rows across {len(subset_specs)} subsets to {args.out_csv} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
import os
import re
import time
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from util.atlas_2d_dataset import Atlas2DImageDataset, DEFAULT_MODALITIES, DEFAULT_RESIZE
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


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def save_uint8_png(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def _safe_case_tag(case_id: str) -> str:
    tag = str(case_id).strip()
    tag = re.sub(r"[\\/:*?\"<>|]+", "_", tag)
    tag = tag.replace(",", "_").replace("'", "")
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"_+", "_", tag).strip("._")
    return tag or "case"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate trained FlexibleFusionNet on Atlas and export metrics CSV")
    p.add_argument("--atlas_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=os.path.join("/storage/student5/ifusion/Intern_ImageFusion/checkpoints_brats_6_26/best_26.pt"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feat_channels", type=int, default=32)

    p.add_argument("--modalities", type=str, nargs="+", default=list(DEFAULT_MODALITIES))
    p.add_argument(
        "--disease_ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional Atlas disease folders to evaluate. Outer quotes around names are tolerated.",
    )
    p.add_argument(
        "--prefer_checkpoint_modalities",
        action="store_true",
        help="Use checkpoint modalities only when they are compatible with the Atlas folder layout.",
    )
    p.add_argument("--allow_missing_modalities", action="store_true", help="Allow SPECT/PET mixed Atlas layout.")
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--resize", type=int, default=DEFAULT_RESIZE)
    p.add_argument("--qcv_window", type=int, default=16)
    p.add_argument("--qcv_alpha", type=int, default=5)
    p.add_argument("--skip_fsim", action="store_true", help="Skip FSIM (slower).")
    p.add_argument("--limit_slices", type=int, default=0)

    p.add_argument("--out_csv", type=str, default="./fusionnet_metrics_atlas.csv")
    p.add_argument("--save_fused", action="store_true", help="Also save fused PNGs under --out_dir/recon/")
    p.add_argument("--out_dir", type=str, default="./result_fusionnet_eval_atlas/")
    args = p.parse_args(argv)

    # Default behavior for Atlas: allow missing modalities (mix SPECT+PET).
    if not bool(args.allow_missing_modalities):
        args.allow_missing_modalities = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Atlas root: {os.path.abspath(args.atlas_root)}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}
    feat_channels = int(ckpt_args.get("feat_channels", args.feat_channels))

    cli_modalities = [str(m).strip() for m in args.modalities]
    ckpt_modalities = []
    if "modalities" in ckpt_args and isinstance(ckpt_args["modalities"], list) and ckpt_args["modalities"]:
        ckpt_modalities = [str(m).strip() for m in ckpt_args["modalities"]]

    modalities = cli_modalities
    if args.prefer_checkpoint_modalities and ckpt_modalities:
        atlas_disease_dirs = [
            os.path.join(args.atlas_root, d)
            for d in os.listdir(args.atlas_root)
            if os.path.isdir(os.path.join(args.atlas_root, d))
        ]
        atlas_mod_names = set()
        for disease_dir in atlas_disease_dirs:
            for name in os.listdir(disease_dir):
                full = os.path.join(disease_dir, name)
                if os.path.isdir(full):
                    atlas_mod_names.add(_normalize_name(name))
        if ckpt_modalities and all(_normalize_name(m) in atlas_mod_names for m in ckpt_modalities):
            modalities = ckpt_modalities
            logger.info(f"Using modalities from checkpoint: {modalities}")
        else:
            logger.info(
                f"Checkpoint modalities {ckpt_modalities} do not match Atlas modalities {sorted(atlas_mod_names)}; "
                f"using CLI/default modalities instead: {cli_modalities}"
            )
    else:
        if ckpt_modalities:
            logger.info(f"Ignoring checkpoint modalities for Atlas eval: {ckpt_modalities}")
        logger.info(f"Using CLI/default modalities: {cli_modalities}")

    if len(modalities) < 2:
        raise SystemExit("--modalities must have at least 2 items.")

    model = FlexibleFusionNet(feat_channels=feat_channels, num_modalities=len(modalities)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    dataset_2d = Atlas2DImageDataset(
        args.atlas_root,
        modalities=modalities,
        scale=int(args.scale),
        resize=int(args.resize),
        disease_ids=args.disease_ids,
        allow_missing_modalities=bool(args.allow_missing_modalities),
        limit_slices=int(args.limit_slices),
    )
    logger.info(f"Aligned Atlas slices: {len(dataset_2d)}")

    case_ids = sorted({str(it.case_id) for it in dataset_2d.indices})
    logger.info(f"Atlas cases: {', '.join(case_ids)}")

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
    skipped_too_few = 0

    for xm_t, present_t, case_id, z in tqdm(dataset_2d, desc="Slices"):
        # xm_t: (M,1,H,W) float01
        # present_t: (M,) bool
        if int(present_t.sum().item()) < 2:
            skipped_too_few += 1
            continue

        h, w = int(xm_t.shape[-2]), int(xm_t.shape[-1])
        h_m = h - h % scale
        w_m = w - w % scale
        if h_m < 3 or w_m < 3:
            skipped_too_small += 1
            continue
        xm_t = xm_t[..., :h_m, :w_m]
        x = xm_t.squeeze(1).numpy()  # (M,Hm,Wm)

        avail = torch.nonzero(present_t, as_tuple=False).flatten()
        mod_ids = torch.tensor([int(a) for a in avail.tolist()], dtype=torch.long, device=device)

        with torch.no_grad():
            fused01_t = model(xm_t.index_select(dim=0, index=avail).unsqueeze(0).to(device), mod_ids=mod_ids)  # (1,1,H,W)
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

        for mi_m, m in enumerate(modalities):
            if not bool(present_t[mi_m].item()):
                continue
            src01 = x[mi_m]
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

        src_u8_list = [normalize1_u8(x[i]) for i in range(len(modalities)) if bool(present_t[i].item())]
        src01_list = [src_u8.astype(np.float64) / 255.0 for src_u8 in src_u8_list]
        qncie = q_ncie_wang(src_u8_list, fused_u8)
        qs = q_s_piella(src01_list, fused_u8.astype(np.float64) / 255.0)

        h_q = h_m - h_m % qcv_window
        w_q = w_m - w_m % qcv_window
        if h_q >= qcv_window and w_q >= qcv_window:
            qcv = q_cv_chen_varshney(
                [s[:h_q, :w_q] for s in src01_list],
                (fused_u8.astype(np.float64) / 255.0)[:h_q, :w_q],
                window_size=qcv_window,
                alpha=int(args.qcv_alpha),
            )
        else:
            qcv = float("nan")

        row = {
            "case_id": str(case_id),
            "slice": int(z),
            "K_present": int(present_t.sum().item()),
            "K_total": int(len(modalities)),
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
            tag = _safe_case_tag(case_id)
            out_path = os.path.join(args.out_dir, "recon", f"{tag}_z{int(z):03d}.png")
            save_uint8_png(out_path, fused_u8)

    cols = [
        "case_id",
        "slice",
        "K_present",
        "K_total",
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
            mean_row = {"case_id": "MEAN", "slice": "", "K_present": "", "K_total": rows[0]["K_total"]}
            for c in cols:
                if c in {"case_id", "slice", "K_present", "K_total"}:
                    continue
                vals = [float(r.get(c, 0.0)) for r in rows if r.get(c, "") != ""]
                mean_row[c] = float(np.mean(vals)) if vals else 0.0
            w.writerow(mean_row)

    if skipped_too_few:
        logger.info(f"Skipped {skipped_too_few} slices with <2 modalities present.")
    if skipped_too_small:
        logger.info(f"Skipped {skipped_too_small} slices due to too-small size after crop to --scale={scale}")
    logger.info(f"Done. Wrote {len(rows)} rows to {args.out_csv} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

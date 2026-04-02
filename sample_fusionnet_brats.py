import argparse
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from util.brats_dataset import AXIS_TO_SLICE, discover_brats_cases, normalize_volume_to_float01, _load_nii, choose_slice_indices
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


def save_uint8_png(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Inference with trained FlexibleFusionNet on BraTS (2-4 modalities)")
    p.add_argument("--brats_root", type=str, default=os.path.join("Dataset-BraTS", "BraTS2024-BraTS-GLI-TrainingData"))
    p.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints_brats", "latest.pth"))
    p.add_argument("--save_dir", type=str, default="./result_fusionnet_brats/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--modalities", type=str, nargs="+", default=["t1n", "t1c", "t2w", "t2f"])
    p.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"])
    p.add_argument("--slice_mode", type=str, default="nonzero", choices=["all", "nonzero", "seg_nonzero"])
    p.add_argument("--min_nonzero_frac", type=float, default=0.01)
    p.add_argument("--p_low", type=float, default=0.5)
    p.add_argument("--p_high", type=float, default=99.5)
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--limit_cases", type=int, default=0)
    p.add_argument("--limit_slices", type=int, default=0)
    p.add_argument("--feat_channels", type=int, default=32, help="Must match training (or inferred from checkpoint args if present).")
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

    os.makedirs(args.save_dir, exist_ok=True)
    recon_dir = os.path.join(args.save_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)

    cases_map = discover_brats_cases(args.brats_root)
    case_ids = sorted(cases_map.keys())
    if args.limit_cases and int(args.limit_cases) > 0:
        case_ids = case_ids[: int(args.limit_cases)]

    slicer = AXIS_TO_SLICE[args.axis]
    total = 0
    t0 = time.time()

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
                sl = slicer(vols01[m], int(z))  # (H,W) float01
                slices.append(np.ascontiguousarray(sl))

            h, w = slices[0].shape
            scale = int(args.scale)
            h = h - h % scale
            w = w - w % scale
            x = np.stack([s[:h, :w] for s in slices], axis=0)  # (K,H,W)
            x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(2).to(device)  # (1,K,1,H,W)

            with torch.no_grad():
                fused01 = model(x_t).detach().cpu().squeeze(0).squeeze(0).numpy()  # (H,W) in [0,1]

            fused_u8 = np.clip(fused01 * 255.0, 0, 255).astype(np.uint8)
            out_path = os.path.join(recon_dir, f"{case_id}_z{int(z):03d}.png")
            save_uint8_png(out_path, fused_u8)
            total += 1

    logger.info(f"Done. Wrote {total} fused slices to {recon_dir} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
import re
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.unet import create_model
from util.logger import get_logger
from util.pytorch_colors import rgb_to_ycbcr

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency `nibabel`. Install with: pip install nibabel") from e


_BRATS_RE = re.compile(r"^(?P<case>.+)-(?P<mod>t1n|t1c|t2w|t2f|seg)\.nii(\.gz)?$")


AXIS_TO_SLICE = {
    "axial": lambda vol, i: vol[:, :, i],
    "coronal": lambda vol, i: vol[:, i, :],
    "sagittal": lambda vol, i: vol[i, :, :],
}


def load_yaml(file_path: str) -> dict:
    with open(file_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _iter_nii_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                yield os.path.join(dirpath, name)


def discover_cases(in_root: str) -> Dict[str, Dict[str, str]]:
    cases: Dict[str, Dict[str, str]] = {}
    for path in _iter_nii_files(in_root):
        m = _BRATS_RE.match(os.path.basename(path))
        if not m:
            continue
        case_id = m.group("case")
        mod = m.group("mod")
        cases.setdefault(case_id, {})[mod] = path
    return cases


def _load_nii(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} for {path}")
    return data


def normalize_volume_to_float01(
    vol: np.ndarray,
    *,
    p_low: float = 0.5,
    p_high: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    v = vol.astype(np.float32, copy=False)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = v[np.abs(v) > 0]
    if nonzero.size == 0:
        return np.zeros_like(v, dtype=np.float32)

    lo = float(np.percentile(nonzero, p_low))
    hi = float(np.percentile(nonzero, p_high))
    if abs(hi - lo) < eps:
        return np.zeros_like(v, dtype=np.float32)

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    return v.astype(np.float32)


def choose_slice_indices(
    reference_vol: np.ndarray,
    *,
    axis: str,
    slice_mode: str,
    seg_reference: Optional[np.ndarray],
    min_nonzero_frac: float,
) -> List[int]:
    axis = axis.lower()
    slicer = AXIS_TO_SLICE[axis]
    num_slices = reference_vol.shape[{"sagittal": 0, "coronal": 1, "axial": 2}[axis]]

    if slice_mode == "all":
        return list(range(num_slices))

    if slice_mode == "nonzero":
        keep: List[int] = []
        for i in range(num_slices):
            sl = slicer(reference_vol, i)
            frac = float(np.mean(np.abs(sl) > 0))
            if frac >= min_nonzero_frac:
                keep.append(i)
        return keep

    if slice_mode == "seg_nonzero":
        if seg_reference is None:
            raise ValueError("slice_mode=seg_nonzero requires seg volume available.")
        keep = []
        for i in range(num_slices):
            sl = slicer(seg_reference, i)
            if np.any(sl > 0):
                keep.append(i)
        return keep

    raise ValueError(f"Unknown slice_mode: {slice_mode}")


def save_uint8_png(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlexiD-Fuse sampling directly from BraTS NIfTI volumes")
    parser.add_argument("--model_config", type=str, default="configs/model_config_imagenet.yaml")
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_config.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default="dit_b4", choices=["dit_b4", "unet"])
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model weights (.pth).")

    parser.add_argument(
        "--brats_root",
        type=str,
        default=os.path.join("Dataset-BraTS", "BraTS2024-BraTS-GLI-TrainingData"),
        help="Root folder searched recursively for BraTS NIfTI files.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["t1n", "t1c", "t2w", "t2f"],
        help="Modalities to fuse (2 to 4): t1n t1c t2w t2f",
    )
    parser.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--slice_mode", type=str, default="nonzero", choices=["all", "nonzero", "seg_nonzero"])
    parser.add_argument("--min_nonzero_frac", type=float, default=0.01)
    parser.add_argument("--p_low", type=float, default=0.5)
    parser.add_argument("--p_high", type=float, default=99.5)
    parser.add_argument("--scale", type=int, default=30, help="Crop to make H/W divisible by this scale.")

    parser.add_argument("--save_dir", type=str, default="./result_brats/")
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.001)
    parser.add_argument("--fusion_objective", type=str, default="metrics", choices=["edge", "metrics"])
    parser.add_argument("--w_en", type=float, default=1.0)
    parser.add_argument("--w_mi", type=float, default=1.0)
    parser.add_argument("--w_psnr", type=float, default=1.0)
    parser.add_argument("--w_ssim", type=float, default=1.0)
    parser.add_argument("--w_sd", type=float, default=1.0)
    parser.add_argument("--w_ag", type=float, default=1.0)
    parser.add_argument("--limit_cases", type=int, default=0)
    parser.add_argument("--limit_slices", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()

    modalities = [m.strip().lower() for m in args.modalities]
    if len(modalities) < 2 or len(modalities) > 4:
        raise SystemExit("--modalities must have 2..4 items.")
    if any(m not in {"t1n", "t1c", "t2w", "t2f"} for m in modalities):
        raise SystemExit("Modalities must be among: t1n t1c t2w t2f")

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Device set to {device_str}.")

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)

    if args.arch == "dit_b4":
        try:
            from guided_diffusion.models_DiT_Mamba import DiT_B_4
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "Failed to import DiT_B_4 (Mamba). Install `mamba_ssm` / `causal_conv1d`, "
                "or run with --arch unet."
            ) from e
        model = DiT_B_4().to(device)
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded pretrained weights from {args.model_path}")
    else:
        model = create_model(**model_config).to(device)
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded pretrained weights from {args.model_path}")
    model.eval()

    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model)

    os.makedirs(args.save_dir, exist_ok=True)
    recon_dir = os.path.join(args.save_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)

    cases = discover_cases(args.brats_root)
    case_ids = sorted(cases.keys())
    if args.limit_cases and args.limit_cases > 0:
        case_ids = case_ids[: args.limit_cases]

    slicer = AXIS_TO_SLICE[args.axis]
    total_done = 0

    for case_id in tqdm(case_ids, desc="Cases"):
        files = cases[case_id]
        if any(m not in files for m in modalities):
            continue

        # Load and normalize volumes.
        vols01: Dict[str, np.ndarray] = {}
        for m in modalities:
            vols01[m] = normalize_volume_to_float01(
                _load_nii(files[m]), p_low=float(args.p_low), p_high=float(args.p_high)
            )

        seg_vol = _load_nii(files["seg"]) if ("seg" in files and args.slice_mode == "seg_nonzero") else None
        ref_vol = seg_vol if (args.slice_mode == "seg_nonzero") else vols01[modalities[0]]
        slice_indices = choose_slice_indices(
            ref_vol,
            axis=args.axis,
            slice_mode=args.slice_mode,
            seg_reference=seg_vol,
            min_nonzero_frac=float(args.min_nonzero_frac),
        )
        if args.limit_slices and args.limit_slices > 0:
            slice_indices = slice_indices[: args.limit_slices]

        for z in slice_indices:
            mods_slices = []
            for m in modalities:
                sl = slicer(vols01[m], int(z))  # float01, (H, W)
                sl = np.ascontiguousarray(sl)
                # Match the legacy pipeline scaling (0..1 -> -1..1).
                sl = sl * 2.0 - 1.0
                mods_slices.append(sl)

            # Crop to divisible size.
            h, w = mods_slices[0].shape
            h = h - h % int(args.scale)
            w = w - w % int(args.scale)
            mods_t = [
                torch.from_numpy(ms[:h, :w]).float().unsqueeze(0).unsqueeze(0).to(device) for ms in mods_slices
            ]

            seed = 3407
            torch.manual_seed(seed)
            x_start = torch.randn((1, 3, h, w), device=device)

            with torch.no_grad():
                sample = sample_fn(
                    x_start=x_start,
                    record=False,
                    modalities=mods_t,
                    save_root=args.save_dir,
                    img_index=f"{case_id}_z{int(z):03d}",
                    lamb=float(args.lamb),
                    rho=float(args.rho),
                    fusion_objective=args.fusion_objective,
                    metric_weights={
                        "EN": float(args.w_en),
                        "MI": float(args.w_mi),
                        "PSNR": float(args.w_psnr),
                        "SSIM": float(args.w_ssim),
                        "SD": float(args.w_sd),
                        "AG": float(args.w_ag),
                    },
                )

            # Save fused luminance (Y channel) as PNG, similar to sample.py.
            rgb = sample.detach().cpu()
            y = (rgb_to_ycbcr(rgb)[:, 0, :, :]).squeeze(0).numpy()
            y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-12)
            y_u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
            out_path = os.path.join(recon_dir, f"{case_id}_z{int(z):03d}.png")
            save_uint8_png(out_path, y_u8)
            total_done += 1

    logger.info(f"Done. Wrote {total_done} fused slices to {recon_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

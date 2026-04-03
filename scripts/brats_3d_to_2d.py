import argparse
import csv
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency `nibabel`.\n"
        "Install it with: pip install nibabel\n"
        "Or add it to your environment requirements."
    ) from e


AXIS_TO_SLICE = {
    "axial": lambda vol, i: vol[:, :, i],
    "coronal": lambda vol, i: vol[:, i, :],
    "sagittal": lambda vol, i: vol[i, :, :],
}


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    by_modality: Dict[str, str]


def _iter_nii_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                yield os.path.join(dirpath, name)


_BRATS_RE = re.compile(r"^(?P<case>.+)-(?P<mod>t1n|t1c|t2w|t2f|seg)\.nii(\.gz)?$")


def discover_brats_cases(in_root: str) -> Dict[str, CaseFiles]:
    """
    Discovers BraTS-style files by pattern:
      <case_id>-<modality>.nii(.gz)
    where modality in {t1n, t1c, t2w, t2f, seg}.
    """
    cases: Dict[str, Dict[str, str]] = {}
    for path in _iter_nii_files(in_root):
        m = _BRATS_RE.match(os.path.basename(path))
        if not m:
            continue
        case_id = m.group("case")
        mod = m.group("mod")
        cases.setdefault(case_id, {})[mod] = path
    return {cid: CaseFiles(case_id=cid, by_modality=mods) for cid, mods in cases.items()}


def normalize_volume_to_uint8(
    vol: np.ndarray,
    *,
    method: str,
    p_low: float,
    p_high: float,
    eps: float = 1e-6,
) -> np.ndarray:
    if method == "none":
        v = vol.astype(np.float32, copy=False)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if abs(vmax - vmin) < eps:
            return np.zeros_like(v, dtype=np.uint8)
        v = (v - vmin) / (vmax - vmin)
        return np.clip(v * 255.0, 0, 255).astype(np.uint8)

    if method != "percentile":
        raise ValueError(f"Unknown normalization method: {method}")

    v = vol.astype(np.float32, copy=False)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = v[np.abs(v) > 0]
    if nonzero.size == 0:
        return np.zeros_like(v, dtype=np.uint8)

    lo = float(np.percentile(nonzero, p_low))
    hi = float(np.percentile(nonzero, p_high))
    if abs(hi - lo) < eps:
        return np.zeros_like(v, dtype=np.uint8)

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    return np.clip(v * 255.0, 0, 255).astype(np.uint8)


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


def save_slice_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {img.shape} for {path}")
    Image.fromarray(img, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BraTS 3D NIfTI volumes (.nii/.nii.gz) into 2D PNG slices."
    )
    parser.add_argument(
        "--in_root",
        type=str,
        default=os.path.join("Dataset-BraTS", "BraTS2024-BraTS-GLI-TrainingData"),
        help="Input root folder (searched recursively for BraTS NIfTI files).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=os.path.join("Dataset", "BraTS_2D"),
        help="Output root folder for 2D PNG slices.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["t1n", "t1c", "t2w", "t2f"],
        help="Modalities to export (choose among: t1n t1c t2w t2f seg).",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Slicing axis.",
    )
    parser.add_argument(
        "--slice_mode",
        type=str,
        default="nonzero",
        choices=["all", "nonzero", "seg_nonzero"],
        help="Which slices to keep.",
    )
    parser.add_argument(
        "--min_nonzero_frac",
        type=float,
        default=0.01,
        help="For slice_mode=nonzero: keep slices with >= this fraction of non-zero pixels.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="percentile",
        choices=["percentile", "none"],
        help="Intensity normalization for MR modalities (seg is always saved as labels).",
    )
    parser.add_argument(
        "--p_low",
        type=float,
        default=0.5,
        help="Lower percentile for --normalize=percentile.",
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=99.5,
        help="Upper percentile for --normalize=percentile.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="modalities",
        choices=["modalities", "flexid"],
        help="Output folder layout: modalities/<mod>/... or flexid/{vi,ir,3}/....",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs (default: skip if exists).",
    )
    parser.add_argument(
        "--limit_cases",
        type=int,
        default=0,
        help="For quick tests: only process N cases (0 = no limit).",
    )
    return parser.parse_args(argv)


def _layout_dirs(layout: str, modalities: List[str]) -> Dict[str, str]:
    if layout == "modalities":
        return {m: m for m in modalities}
    if layout == "flexid":
        if len(modalities) > 3:
            warnings.warn(
                "--layout flexid only supports up to 3 slots (vi/ir/3). "
                "Falling back to --layout modalities for 4+ modalities."
            )
            return {m: m for m in modalities}
        if len(modalities) < 2:
            raise ValueError("--layout flexid requires at least 2 modalities.")
        mapping = {modalities[0]: "vi", modalities[1]: "ir"}
        if len(modalities) == 3:
            mapping[modalities[2]] = "3"
        return mapping
    raise ValueError(f"Unknown layout: {layout}")


def _load_nii(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} for {path}")
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    modalities = [m.lower() for m in args.modalities]
    unknown = [m for m in modalities if m not in {"t1n", "t1c", "t2w", "t2f", "seg"}]
    if unknown:
        raise SystemExit(f"Unknown modalities: {unknown}")

    slicer = AXIS_TO_SLICE[args.axis]
    out_dirs = _layout_dirs(args.layout, modalities)

    cases = discover_brats_cases(args.in_root)
    case_ids = sorted(cases.keys())
    if args.limit_cases and args.limit_cases > 0:
        case_ids = case_ids[: args.limit_cases]

    os.makedirs(args.out_root, exist_ok=True)
    manifest_path = os.path.join(args.out_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f_manifest:
        writer = csv.writer(f_manifest)
        writer.writerow(["case_id", "slice_index", "axis", "modality", "out_path"])

        for case_id in tqdm(case_ids, desc="Cases"):
            case = cases[case_id]

            missing = [m for m in modalities if m not in case.by_modality]
            if missing:
                # Skip incomplete cases silently (BraTS val/test may not have seg).
                continue

            ref_mod = "seg" if (args.slice_mode == "seg_nonzero") else modalities[0]
            ref_vol = _load_nii(case.by_modality[ref_mod])
            seg_vol = _load_nii(case.by_modality["seg"]) if ("seg" in case.by_modality) else None

            slice_indices = choose_slice_indices(
                ref_vol,
                axis=args.axis,
                slice_mode=args.slice_mode,
                seg_reference=seg_vol,
                min_nonzero_frac=float(args.min_nonzero_frac),
            )

            # Pre-normalize MR volumes for consistent scaling across slices.
            vol_u8_by_mod: Dict[str, np.ndarray] = {}
            for mod in modalities:
                vol = _load_nii(case.by_modality[mod])
                if mod == "seg":
                    vol_u8_by_mod[mod] = np.clip(np.rint(vol), 0, 255).astype(np.uint8)
                else:
                    vol_u8_by_mod[mod] = normalize_volume_to_uint8(
                        vol, method=args.normalize, p_low=float(args.p_low), p_high=float(args.p_high)
                    )

            for i in slice_indices:
                for mod in modalities:
                    sl = slicer(vol_u8_by_mod[mod], i)
                    # Ensure 2D contiguous array for cv2.
                    sl = np.ascontiguousarray(sl)
                    out_dir = os.path.join(args.out_root, out_dirs[mod])
                    out_name = f"{case_id}_z{int(i):03d}.png"
                    out_path = os.path.join(out_dir, out_name)
                    if (not args.overwrite) and os.path.exists(out_path):
                        continue
                    save_slice_png(out_path, sl)
                    writer.writerow([case_id, int(i), args.axis, mod, out_path])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

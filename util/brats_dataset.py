import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

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


@dataclass(frozen=True)
class BraTSCase:
    case_id: str
    paths: Dict[str, str]


def _iter_nii_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                yield os.path.join(dirpath, name)


def discover_brats_cases(in_root: str) -> Dict[str, BraTSCase]:
    cases: Dict[str, Dict[str, str]] = {}
    for path in _iter_nii_files(in_root):
        m = _BRATS_RE.match(os.path.basename(path))
        if not m:
            continue
        case_id = m.group("case")
        mod = m.group("mod")
        cases.setdefault(case_id, {})[mod] = path
    return {cid: BraTSCase(case_id=cid, paths=mods) for cid, mods in cases.items()}


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


class BraTSSliceDataset(Dataset):
    """
    Returns aligned 2D slices for the 4 BraTS modalities as:
      x: FloatTensor (4, 1, H, W) in [0,1]
    """

    def __init__(
        self,
        brats_root: str,
        *,
        modalities: Sequence[str] = ("t1n", "t1c", "t2w", "t2f"),
        axis: str = "axial",
        slice_mode: str = "nonzero",
        min_nonzero_frac: float = 0.01,
        p_low: float = 0.5,
        p_high: float = 99.5,
        scale: int = 30,
        limit_cases: int = 0,
        seed: int = 3407,
    ):
        self.axis = axis
        self.slice_mode = slice_mode
        self.min_nonzero_frac = float(min_nonzero_frac)
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.scale = int(scale)
        self.modalities = [m.strip().lower() for m in modalities]
        if set(self.modalities) != {"t1n", "t1c", "t2w", "t2f"}:
            raise ValueError("BraTSSliceDataset currently expects modalities = t1n t1c t2w t2f (all four).")

        rng = random.Random(int(seed))
        cases = discover_brats_cases(brats_root)
        case_ids = sorted(cases.keys())
        if limit_cases and limit_cases > 0:
            case_ids = case_ids[: int(limit_cases)]

        kept: List[BraTSCase] = []
        for cid in case_ids:
            c = cases[cid]
            if all(m in c.paths for m in self.modalities):
                kept.append(c)
        rng.shuffle(kept)
        self.cases = kept
        if not self.cases:
            raise RuntimeError("No BraTS cases found with all 4 modalities present.")

    def __len__(self) -> int:
        # Infinite-ish sampling by reusing cases; length is arbitrary but stable.
        return max(1024, len(self.cases) * 16)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        case = self.cases[idx % len(self.cases)]

        vols01: Dict[str, np.ndarray] = {}
        for m in self.modalities:
            vols01[m] = normalize_volume_to_float01(_load_nii(case.paths[m]), p_low=self.p_low, p_high=self.p_high)

        ref_vol = vols01[self.modalities[0]]
        seg_vol = _load_nii(case.paths["seg"]) if ("seg" in case.paths and self.slice_mode == "seg_nonzero") else None
        if self.slice_mode == "seg_nonzero":
            ref_vol = seg_vol

        slice_indices = choose_slice_indices(
            ref_vol,
            axis=self.axis,
            slice_mode=self.slice_mode,
            seg_reference=seg_vol,
            min_nonzero_frac=self.min_nonzero_frac,
        )
        if not slice_indices:
            # fallback
            slice_indices = [ref_vol.shape[2] // 2]
        z = int(slice_indices[random.randint(0, len(slice_indices) - 1)])

        slicer = AXIS_TO_SLICE[self.axis]
        slices = []
        for m in self.modalities:
            sl = slicer(vols01[m], z)  # (H,W) float01
            slices.append(np.ascontiguousarray(sl))

        h, w = slices[0].shape
        h = h - h % self.scale
        w = w - w % self.scale
        x = np.stack([s[:h, :w] for s in slices], axis=0)  # (4,H,W)
        x_t = torch.from_numpy(x).float().unsqueeze(1)  # (4,1,H,W)
        return x_t, case.case_id, z


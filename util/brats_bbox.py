from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from util.brats_dataset import AXIS_TO_SLICE, discover_brats_cases, _load_nii


_FUSED_RE = re.compile(r"^(?P<case>.+)_z(?P<z>\d+)\.png$")


def parse_fused_slice_name(name: str) -> Tuple[str, int]:
    """
    Expected: <case_id>_z###.png
    Returns: (case_id, z)
    """
    m = _FUSED_RE.match(os.path.basename(name))
    if not m:
        raise ValueError(f"Unrecognized fused slice name: {name}")
    return m.group("case"), int(m.group("z"))


def bbox_xyxy_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    mask: (H,W) integer/boolean.
    Returns bbox (x0,y0,x1,y1) in pixel coords, with x1/y1 exclusive.
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def xyxy_px_to_norm(x0: int, y0: int, x1: int, y1: int, *, w: int, h: int) -> Tuple[float, float, float, float]:
    # Normalize to [0,1] using image width/height, with x1/y1 exclusive.
    x0n = float(x0) / float(w)
    y0n = float(y0) / float(h)
    x1n = float(x1) / float(w)
    y1n = float(y1) / float(h)
    return x0n, y0n, x1n, y1n


def xyxy_norm_to_px(x0: float, y0: float, x1: float, y1: float, *, w: int, h: int) -> Tuple[int, int, int, int]:
    x0p = int(round(x0 * w))
    y0p = int(round(y0 * h))
    x1p = int(round(x1 * w))
    y1p = int(round(y1 * h))
    x0p = max(0, min(w, x0p))
    y0p = max(0, min(h, y0p))
    x1p = max(0, min(w, x1p))
    y1p = max(0, min(h, y1p))
    if x1p < x0p:
        x0p, x1p = x1p, x0p
    if y1p < y0p:
        y0p, y1p = y1p, y0p
    return x0p, y0p, x1p, y1p


def bbox_iou_xyxy_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a,b: (...,4) in normalized xyxy with x1/y1 exclusive.
    Returns IoU (...,)
    """
    ax0, ay0, ax1, ay1 = a.unbind(dim=-1)
    bx0, by0, bx1, by1 = b.unbind(dim=-1)

    ix0 = torch.maximum(ax0, bx0)
    iy0 = torch.maximum(ay0, by0)
    ix1 = torch.minimum(ax1, bx1)
    iy1 = torch.minimum(ay1, by1)

    iw = torch.clamp(ix1 - ix0, min=0.0)
    ih = torch.clamp(iy1 - iy0, min=0.0)
    inter = iw * ih

    aw = torch.clamp(ax1 - ax0, min=0.0)
    ah = torch.clamp(ay1 - ay0, min=0.0)
    bw = torch.clamp(bx1 - bx0, min=0.0)
    bh = torch.clamp(by1 - by0, min=0.0)
    union = aw * ah + bw * bh - inter + 1e-12
    return inter / union


@dataclass(frozen=True)
class BBoxManifestRow:
    fused_relpath: str
    case_id: str
    z: int
    has_tumor: int
    x0: float
    y0: float
    x1: float
    y1: float


def read_bbox_manifest_csv(path: str) -> List[BBoxManifestRow]:
    rows: List[BBoxManifestRow] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                BBoxManifestRow(
                    fused_relpath=str(r["fused_relpath"]),
                    case_id=str(r["case_id"]),
                    z=int(r["z"]),
                    has_tumor=int(r["has_tumor"]),
                    x0=float(r["x0"]),
                    y0=float(r["y0"]),
                    x1=float(r["x1"]),
                    y1=float(r["y1"]),
                )
            )
    return rows


def split_manifest_by_case(
    rows: Sequence[BBoxManifestRow],
    *,
    val_frac: float = 0.2,
    seed: int = 3407,
) -> Tuple[List[int], List[int]]:
    by_case: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_case.setdefault(r.case_id, []).append(i)

    case_ids = sorted(by_case.keys())
    rng = random.Random(int(seed))
    rng.shuffle(case_ids)
    n_val = int(round(len(case_ids) * float(val_frac)))
    val_cases = set(case_ids[:n_val])

    train_idx: List[int] = []
    val_idx: List[int] = []
    for cid, idxs in by_case.items():
        (val_idx if cid in val_cases else train_idx).extend(idxs)
    return train_idx, val_idx


class BraTSFusedBBoxDataset(Dataset):
    """
    Loads fused 2D slices (PNG) and bbox targets built from BraTS seg labels.

    Returns:
      x: FloatTensor (1,H,W) in [0,1]
      has_tumor: FloatTensor () in {0,1}
      bbox: FloatTensor (4,) normalized xyxy (x1/y1 exclusive). If has_tumor=0, bbox is zeros.
    """

    def __init__(
        self,
        manifest_rows: Sequence[BBoxManifestRow],
        *,
        fused_root: str,
        indices: Optional[Sequence[int]] = None,
        augment: bool = False,
        seed: int = 3407,
    ) -> None:
        self.rows = list(manifest_rows)
        self.fused_root = fused_root
        self.indices = list(indices) if indices is not None else list(range(len(self.rows)))
        self.augment = bool(augment)
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.indices)

    def _maybe_aug(self, img: torch.Tensor, bbox: torch.Tensor, has_tumor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return img, bbox
        # img: (1,H,W)
        do_h = self.rng.random() < 0.5
        do_v = self.rng.random() < 0.5
        if do_h:
            img = torch.flip(img, dims=[2])
            if has_tumor.item() > 0.5:
                x0, y0, x1, y1 = bbox.tolist()
                bbox = torch.tensor([1.0 - x1, y0, 1.0 - x0, y1], dtype=bbox.dtype)
        if do_v:
            img = torch.flip(img, dims=[1])
            if has_tumor.item() > 0.5:
                x0, y0, x1, y1 = bbox.tolist()
                bbox = torch.tensor([x0, 1.0 - y1, x1, 1.0 - y0], dtype=bbox.dtype)
        return img, bbox

    def __getitem__(self, i: int):
        row = self.rows[self.indices[i]]
        path = os.path.join(self.fused_root, row.fused_relpath)
        img_u8 = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        x = torch.from_numpy(img_u8).float().unsqueeze(0) / 255.0  # (1,H,W)

        has_tumor = torch.tensor(float(row.has_tumor), dtype=torch.float32)
        bbox = torch.tensor([row.x0, row.y0, row.x1, row.y1], dtype=torch.float32)
        x, bbox = self._maybe_aug(x, bbox, has_tumor)
        bbox = torch.clamp(bbox, 0.0, 1.0)
        return x, has_tumor, bbox, row.case_id, int(row.z)


def collate_bbox(batch):
    xs, has_ts, bbs, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)  # (N,1,H,W)
    has_t = torch.stack(has_ts, dim=0)  # (N,)
    bb = torch.stack(bbs, dim=0)  # (N,4)
    return x, has_t, bb, list(case_ids), list(zs)


def build_bbox_manifest(
    *,
    fused_dir: str,
    brats_root: str,
    axis: str = "axial",
    out_csv: str,
) -> int:
    """
    Creates a CSV mapping fused slice PNGs to bbox targets derived from BraTS seg labels.
    """
    axis = axis.lower()
    if axis not in AXIS_TO_SLICE:
        raise ValueError(f"Unknown axis: {axis}")

    cases = discover_brats_cases(brats_root)
    slicer = AXIS_TO_SLICE[axis]

    fused_paths: List[str] = []
    for name in os.listdir(fused_dir):
        if name.lower().endswith(".png"):
            fused_paths.append(os.path.join(fused_dir, name))
    fused_paths.sort()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    seg_cache: Dict[str, np.ndarray] = {}
    wrote = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fused_relpath", "case_id", "z", "has_tumor", "x0", "y0", "x1", "y1"],
        )
        writer.writeheader()

        for fp in fused_paths:
            name = os.path.basename(fp)
            case_id, z = parse_fused_slice_name(name)
            if case_id not in cases:
                continue
            case = cases[case_id]
            if "seg" not in case.paths:
                continue

            seg_vol = seg_cache.get(case_id)
            if seg_vol is None:
                seg_vol = _load_nii(case.paths["seg"]).astype(np.int16, copy=False)
                seg_cache[case_id] = seg_vol

            seg_sl = slicer(seg_vol, int(z))
            img = Image.open(fp).convert("L")
            w, h = img.size
            if seg_sl.shape[0] < h or seg_sl.shape[1] < w:
                raise RuntimeError(f"Seg slice smaller than fused image for {name}: seg={seg_sl.shape}, img={(h,w)}")
            seg_sl = np.ascontiguousarray(seg_sl[:h, :w])

            bb_px = bbox_xyxy_from_mask(seg_sl)
            if bb_px is None:
                row = {
                    "fused_relpath": os.path.relpath(fp, fused_dir).replace("\\", "/"),
                    "case_id": case_id,
                    "z": int(z),
                    "has_tumor": 0,
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 0.0,
                    "y1": 0.0,
                }
            else:
                x0, y0, x1, y1 = bb_px
                x0n, y0n, x1n, y1n = xyxy_px_to_norm(x0, y0, x1, y1, w=w, h=h)
                row = {
                    "fused_relpath": os.path.relpath(fp, fused_dir).replace("\\", "/"),
                    "case_id": case_id,
                    "z": int(z),
                    "has_tumor": 1,
                    "x0": float(x0n),
                    "y0": float(y0n),
                    "x1": float(x1n),
                    "y1": float(y1n),
                }
            writer.writerow(row)
            wrote += 1
    return wrote


from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_SLICE_RE = re.compile(r"^(?P<case>.+)_z(?P<z>\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class SliceIndex:
    name: str
    case_id: str
    z: int


def _parse_slice_name(name: str) -> SliceIndex:
    m = _SLICE_RE.match(os.path.basename(name))
    if not m:
        raise ValueError(f"Unrecognized slice filename (expected *_z###.png): {name}")
    return SliceIndex(name=os.path.basename(name), case_id=m.group("case"), z=int(m.group("z")))


def _list_png_names(folder: str) -> List[str]:
    return [n for n in os.listdir(folder) if n.lower().endswith(".png")]


def _remap_brats_labels(seg_u8: np.ndarray, *, remap_4_to_3: bool) -> np.ndarray:
    seg = seg_u8.astype(np.int64, copy=False)
    if remap_4_to_3:
        seg = np.where(seg == 4, 3, seg)
    return seg.astype(np.int64, copy=False)


class BraTSFused2DSegDataset(Dataset):
    """
    Loads paired (fused_image, seg_mask) 2D PNG slices.

    Expected layout:
      fused_dir/<case>_z###.png
      seg_dir/<case>_z###.png

    Returns:
      x: FloatTensor (1,H,W) in [0,1]
      y: LongTensor (H,W) with class ids (optionally remapped 4->3)
      case_id: str
      z: int
    """

    def __init__(
        self,
        fused_dir: str,
        seg_dir: str,
        *,
        scale: int = 1,
        limit_slices: int = 0,
        remap_4_to_3: bool = True,
    ) -> None:
        self.fused_dir = str(fused_dir)
        self.seg_dir = str(seg_dir)
        self.scale = int(scale)
        self.remap_4_to_3 = bool(remap_4_to_3)

        if not os.path.isdir(self.fused_dir):
            raise RuntimeError(f"Missing fused_dir: {self.fused_dir}")
        if not os.path.isdir(self.seg_dir):
            raise RuntimeError(f"Missing seg_dir: {self.seg_dir}")

        fused_names = set(_list_png_names(self.fused_dir))
        seg_names = set(_list_png_names(self.seg_dir))
        common = sorted(fused_names & seg_names)
        if not common:
            raise RuntimeError(f"No aligned fused+seg PNGs found. fused_dir={self.fused_dir} seg_dir={self.seg_dir}")

        indices = [_parse_slice_name(n) for n in common]
        indices.sort(key=lambda s: (s.case_id, s.z))
        if limit_slices and int(limit_slices) > 0:
            indices = indices[: int(limit_slices)]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        it = self.indices[int(idx)]
        img_u8 = np.array(Image.open(os.path.join(self.fused_dir, it.name)).convert("L"), dtype=np.uint8)
        seg_u8 = np.array(Image.open(os.path.join(self.seg_dir, it.name)).convert("L"), dtype=np.uint8)

        h, w = img_u8.shape
        h2 = min(h, seg_u8.shape[0])
        w2 = min(w, seg_u8.shape[1])
        if self.scale > 1:
            h2 = h2 - (h2 % self.scale)
            w2 = w2 - (w2 % self.scale)
        if h2 < 1 or w2 < 1:
            raise RuntimeError(f"Invalid slice size after crop: {(h2, w2)} for {it.name}")

        img01 = (img_u8[:h2, :w2].astype(np.float32) / 255.0).astype(np.float32, copy=False)
        seg = _remap_brats_labels(seg_u8[:h2, :w2], remap_4_to_3=self.remap_4_to_3)

        x = torch.from_numpy(img01).unsqueeze(0)  # (1,H,W)
        y = torch.from_numpy(seg).long()  # (H,W)
        return x, y, it.case_id, int(it.z)


def intersect_fused_dirs(fused_dir: str, seg_dir: str, *, out_txt: str) -> None:
    """
    Utility for debugging: writes the common filenames into out_txt.
    """
    fused_names = set(_list_png_names(fused_dir))
    seg_names = set(_list_png_names(seg_dir))
    common = sorted(fused_names & seg_names)
    os.makedirs(os.path.dirname(os.path.abspath(out_txt)) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for n in common:
            f.write(n + "\n")


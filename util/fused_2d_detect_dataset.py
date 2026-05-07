from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

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


class BraTSFused2DDetectDataset(Dataset):
    """
    Slice-level tumor detection from fused 2D PNG slices.

    Expected layout:
      fused_dir/<case>_z###.png
      seg_dir/<case>_z###.png  (used only to build labels)

    Label:
      y = 1 if seg has any non-zero pixel (tumor present), else 0.

    Returns:
      x: FloatTensor (1,H,W) in [0,1]
      y: FloatTensor () in {0,1} (for BCEWithLogitsLoss)
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
    ) -> None:
        self.fused_dir = str(fused_dir)
        self.seg_dir = str(seg_dir)
        self.scale = int(scale)

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

        # Precompute labels to speed up training and to report class balance.
        labels = []
        for it in self.indices:
            seg_u8 = np.array(Image.open(os.path.join(self.seg_dir, it.name)).convert("L"), dtype=np.uint8)
            labels.append(1.0 if bool(np.any(seg_u8 > 0)) else 0.0)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        it = self.indices[int(idx)]
        img_u8 = np.array(Image.open(os.path.join(self.fused_dir, it.name)).convert("L"), dtype=np.uint8)
        h, w = img_u8.shape
        if self.scale > 1:
            h = h - (h % self.scale)
            w = w - (w % self.scale)
        if h < 1 or w < 1:
            raise RuntimeError(f"Invalid slice size after crop: {(h, w)} for {it.name}")

        img01 = (img_u8[:h, :w].astype(np.float32) / 255.0).astype(np.float32, copy=False)
        x = torch.from_numpy(img01).unsqueeze(0)  # (1,H,W)
        y = torch.tensor(self.labels[int(idx)], dtype=torch.float32)  # scalar in {0,1}
        return x, y, it.case_id, int(it.z)


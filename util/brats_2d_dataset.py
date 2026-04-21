from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_SLICE_RE = re.compile(r"^(?P<case>.+)_z(?P<z>\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Slice2DIndex:
    name: str
    case_id: str
    z: int


def _parse_slice_name(name: str) -> Slice2DIndex:
    m = _SLICE_RE.match(os.path.basename(name))
    if not m:
        raise ValueError(f"Unrecognized slice filename (expected *_z###.png): {name}")
    return Slice2DIndex(name=os.path.basename(name), case_id=m.group("case"), z=int(m.group("z")))


def _list_png_names(folder: str) -> List[str]:
    names = []
    for n in os.listdir(folder):
        if n.lower().endswith(".png"):
            names.append(n)
    return names


class BraTS2DPngDataset(Dataset):
    """
    Loads aligned 2D PNG slices exported from BraTS volumes, in a per-modality folder structure:
      root/<modality>/<case>_z###.png

    Returns:
      x: FloatTensor (M,1,H,W) in [0,1]
      case_id: str
      z: int
    """

    def __init__(
        self,
        root: str,
        *,
        modalities: Sequence[str] = ("t1n", "t1c", "t2w", "t2f"),
        scale: int = 30,
        limit_slices: int = 0,
    ) -> None:
        self.root = str(root)
        self.modalities = [m.strip().lower() for m in modalities]
        if len(self.modalities) < 2 or len(self.modalities) > 4:
            raise ValueError("BraTS2DPngDataset expects 2..4 modalities.")
        self.scale = int(scale)

        by_mod: Dict[str, str] = {}
        for m in self.modalities:
            p = os.path.join(self.root, m)
            if not os.path.isdir(p):
                raise RuntimeError(f"Missing modality folder: {p}")
            by_mod[m] = p
        self.by_mod = by_mod

        # Build an aligned slice list by intersecting filenames across modality folders.
        name_sets = []
        for m in self.modalities:
            name_sets.append(set(_list_png_names(self.by_mod[m])))
        common = set.intersection(*name_sets) if name_sets else set()
        if not common:
            raise RuntimeError(f"No aligned PNG slices found under {self.root} for modalities={self.modalities}")

        indices = [_parse_slice_name(n) for n in common]
        indices.sort(key=lambda s: (s.case_id, s.z))
        if limit_slices and int(limit_slices) > 0:
            indices = indices[: int(limit_slices)]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        it = self.indices[int(idx)]
        slices = []
        for m in self.modalities:
            path = os.path.join(self.by_mod[m], it.name)
            img_u8 = np.array(Image.open(path).convert("L"), dtype=np.uint8)
            slices.append(img_u8.astype(np.float32) / 255.0)  # (H,W) float01

        h, w = slices[0].shape
        h = h - h % self.scale
        w = w - w % self.scale
        x = np.stack([s[:h, :w] for s in slices], axis=0)  # (M,H,W)
        x_t = torch.from_numpy(x).float().unsqueeze(1)  # (M,1,H,W)
        return x_t, it.case_id, int(it.z)


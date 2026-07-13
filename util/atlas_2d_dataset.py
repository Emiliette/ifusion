from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_MODALITIES = ("MR-T1", "MR-T2", "SPECT", "PET")
DEFAULT_RESIZE = 240

_NUMERIC_STEM_RE = re.compile(r"^(?P<z>\d+)$")


@dataclass(frozen=True)
class AtlasSliceIndex:
    case_id: str
    z: int
    paths: Tuple[Optional[str], ...]
    present: Tuple[bool, ...]


def _normalize_dir_key(name: str) -> str:
    text = str(name).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text.lower()


def _resolve_child_dir(parent: str, child_name: str) -> str:
    want = _normalize_dir_key(child_name)
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if os.path.isdir(full) and _normalize_dir_key(name) == want:
            return full
    raise RuntimeError(f"Missing folder '{child_name}' under: {parent}")


def _resolve_existing_dir_names(parent: str, requested_names: Sequence[str]) -> List[str]:
    actual_dirs = [name for name in os.listdir(parent) if os.path.isdir(os.path.join(parent, name))]
    norm_to_actual: Dict[str, str] = {}
    for name in sorted(actual_dirs):
        norm_to_actual.setdefault(_normalize_dir_key(name), name)

    resolved: List[str] = []
    missing: List[str] = []
    for name in requested_names:
        actual = norm_to_actual.get(_normalize_dir_key(name))
        if actual is None:
            missing.append(str(name))
        elif actual not in resolved:
            resolved.append(actual)

    if missing:
        raise RuntimeError(
            f"Unknown Atlas folder(s): {missing}. Available folders under {parent}: {sorted(actual_dirs)}"
        )
    return resolved


def _list_subdirs(folder: str) -> List[str]:
    out = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            out.append(name)
    out.sort()
    return out


def _list_image_paths(folder: str) -> List[str]:
    out = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in (".png", ".gif", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            out.append(os.path.join(folder, name))
    return out


def _parse_numeric_stem(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _NUMERIC_STEM_RE.match(stem)
    if not match:
        raise ValueError(f"Unrecognized slice filename (expected NNN.<ext>): {path}")
    return int(match.group("z"))


def _build_z_to_path(folder: str) -> Dict[int, str]:
    z_to_path: Dict[int, str] = {}
    for path in _list_image_paths(folder):
        z_to_path[_parse_numeric_stem(path)] = path
    return z_to_path


class Atlas2DImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        *,
        modalities: Sequence[str],
        scale: int = 30,
        resize: int = DEFAULT_RESIZE,
        disease_ids: Optional[Sequence[str]] = None,
        allow_missing_modalities: bool = False,
        limit_slices: int = 0,
    ) -> None:
        self.root = str(root)
        self.modalities = [modality.strip() for modality in modalities if str(modality).strip()]
        if len(self.modalities) < 2:
            raise ValueError("Atlas2DImageDataset expects at least 2 modalities.")
        self.scale = int(scale)
        self.resize = int(resize)
        self.allow_missing_modalities = bool(allow_missing_modalities)

        if not os.path.isdir(self.root):
            raise RuntimeError(f"Missing Atlas root directory: {self.root}")

        all_diseases = [name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))]
        all_diseases.sort()
        diseases = _resolve_existing_dir_names(self.root, disease_ids) if disease_ids is not None else all_diseases
        if not diseases:
            raise RuntimeError(f"No disease folders found under: {self.root}")

        indices: List[AtlasSliceIndex] = []
        for disease_id in diseases:
            disease_dir = os.path.join(self.root, disease_id)
            mod_dirs: List[Optional[str]] = []
            for modality in self.modalities:
                try:
                    mod_dirs.append(_resolve_child_dir(disease_dir, modality))
                except RuntimeError:
                    mod_dirs.append(None)

            present_mods = [path is not None for path in mod_dirs]
            if sum(present_mods) < 2:
                continue
            if not self.allow_missing_modalities and not all(present_mods):
                continue

            has_direct_images = False
            subcase_sets: List[set[str]] = []
            for path, present in zip(mod_dirs, present_mods):
                if not present or path is None:
                    continue
                if _list_image_paths(path):
                    has_direct_images = True
                subcase_sets.append(set(_list_subdirs(path)))

            if has_direct_images:
                case_names = ["."]
            else:
                common = set.intersection(*subcase_sets) if subcase_sets else set()
                if not common:
                    continue
                case_names = sorted(common)

            for case_name in case_names:
                case_id = disease_id if case_name == "." else f"{disease_id}/{case_name}"
                z_to_paths: List[Dict[int, str]] = []
                for path in mod_dirs:
                    if path is None:
                        z_to_paths.append({})
                        continue
                    folder = path if case_name == "." else os.path.join(path, case_name)
                    if not os.path.isdir(folder):
                        z_to_paths.append({})
                        continue
                    try:
                        z_to_paths.append(_build_z_to_path(folder))
                    except ValueError:
                        z_to_paths.append({})

                z_counts: Dict[int, int] = {}
                for z_to_path in z_to_paths:
                    for z in z_to_path.keys():
                        z_counts[z] = z_counts.get(z, 0) + 1

                for z in sorted([z for z, count in z_counts.items() if count >= 2]):
                    paths: List[Optional[str]] = []
                    present: List[bool] = []
                    for z_to_path in z_to_paths:
                        path = z_to_path.get(z)
                        paths.append(path)
                        present.append(path is not None)
                    indices.append(
                        AtlasSliceIndex(case_id=case_id, z=int(z), paths=tuple(paths), present=tuple(present))
                    )

        if not indices:
            raise RuntimeError(
                f"No aligned Atlas slices found under {self.root}. "
                f"Expected Atlas/<disease>/<modality>/[<case>/]NNN.<ext> for modalities={self.modalities}"
            )

        if limit_slices and int(limit_slices) > 0:
            indices = indices[: int(limit_slices)]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        item = self.indices[int(idx)]
        slices: List[Optional[np.ndarray]] = []
        present: List[bool] = []
        for path in item.paths:
            if path is None:
                slices.append(None)
                present.append(False)
                continue

            image = Image.open(path).convert("L")
            if int(self.resize) > 0:
                image = image.resize((int(self.resize), int(self.resize)), resample=Image.BILINEAR)
            image_u8 = np.array(image, dtype=np.uint8)
            slices.append(image_u8.astype(np.float32) / 255.0)
            present.append(True)

        first = next((slice_ for slice_ in slices if slice_ is not None), None)
        if first is None:
            raise RuntimeError("Corrupt sample: no modality present.")

        height, width = first.shape
        if int(self.scale) > 1:
            height = height - height % self.scale
            width = width - width % self.scale

        filled = []
        for slice_ in slices:
            if slice_ is None:
                filled.append(np.zeros((height, width), dtype=np.float32))
                continue
            if slice_.shape[0] != height or slice_.shape[1] != width:
                image = Image.fromarray((slice_ * 255.0).clip(0, 255).astype(np.uint8), mode="L")
                image = image.resize((int(width), int(height)), resample=Image.BILINEAR)
                slice_ = np.array(image, dtype=np.uint8).astype(np.float32) / 255.0
            filled.append(slice_[:height, :width].astype(np.float32))

        x = np.stack(filled, axis=0)
        x_t = torch.from_numpy(x).float().unsqueeze(1)
        present_t = torch.tensor(present, dtype=torch.bool)
        return x_t, present_t, item.case_id, int(item.z)

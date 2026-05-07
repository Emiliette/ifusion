import argparse
import csv
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from util.fusion_losses import fusion_loss_source_guided
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


DEFAULT_ATLAS_ROOT = "Atlas"
DEFAULT_MODALITIES = ("MR-T1", "MR-T2", "SPECT", "PET")
DEFAULT_EPOCHS = 100
DEFAULT_RESIZE = 240


_NUMERIC_STEM_RE = re.compile(r"^(?P<z>\d+)$")


@dataclass(frozen=True)
class AtlasSliceIndex:
    case_id: str
    z: int
    paths: Tuple[Optional[str], ...]  # len == num_modalities
    present: Tuple[bool, ...]  # len == num_modalities


def _resolve_child_dir(parent: str, child_name: str) -> str:
    want = str(child_name).strip().lower()
    for n in os.listdir(parent):
        full = os.path.join(parent, n)
        if os.path.isdir(full) and n.strip().lower() == want:
            return full
    raise RuntimeError(f"Missing folder '{child_name}' under: {parent}")


def _list_subdirs(folder: str) -> List[str]:
    out = []
    for n in os.listdir(folder):
        p = os.path.join(folder, n)
        if os.path.isdir(p):
            out.append(n)
    out.sort()
    return out


def _list_image_paths(folder: str) -> List[str]:
    out = []
    for n in os.listdir(folder):
        ext = os.path.splitext(n)[1].lower()
        if ext in (".png", ".gif", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            out.append(os.path.join(folder, n))
    return out


def _parse_numeric_stem(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _NUMERIC_STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognized slice filename (expected NNN.<ext>): {path}")
    return int(m.group("z"))


def _build_z_to_path(folder: str) -> Dict[int, str]:
    z_to_path: Dict[int, str] = {}
    for p in _list_image_paths(folder):
        z = _parse_numeric_stem(p)
        z_to_path[z] = p
    return z_to_path


class Atlas2DImageDataset(Dataset):
    """
    Atlas folder expected structure (as you described):
      Atlas/<disease>/<modality>/[<case>/]NNN.(png|gif|...)

    Example modalities: MR-T1, MR-T2, SPECT, PET.

    Returns:
      x: FloatTensor (M,1,H,W) in [0,1] (missing modalities are zero-filled if allow_missing_modalities=True)
      present: BoolTensor (M,) modality availability mask
      case_id: str  (e.g. "DiseaseA" or "DiseaseA/case1")
      z: int
    """

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
        self.modalities = [m.strip() for m in modalities if str(m).strip()]
        if len(self.modalities) < 2:
            raise ValueError("Atlas2DImageDataset expects at least 2 modalities.")
        self.scale = int(scale)
        self.resize = int(resize)
        self.allow_missing_modalities = bool(allow_missing_modalities)

        if not os.path.isdir(self.root):
            raise RuntimeError(f"Missing Atlas root directory: {self.root}")

        all_diseases = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        all_diseases.sort()
        if disease_ids is not None:
            want = {str(d) for d in disease_ids}
            diseases = [d for d in all_diseases if d in want]
        else:
            diseases = all_diseases
        if not diseases:
            raise RuntimeError(f"No disease folders found under: {self.root}")

        indices: List[AtlasSliceIndex] = []
        for disease_id in diseases:
            disease_dir = os.path.join(self.root, disease_id)

            # Resolve modality dirs; allow some to be absent if requested.
            mod_dirs: List[Optional[str]] = []
            for m in self.modalities:
                try:
                    mod_dirs.append(_resolve_child_dir(disease_dir, m))
                except RuntimeError:
                    mod_dirs.append(None)

            present_mods = [p is not None for p in mod_dirs]
            if sum(present_mods) < 2:
                continue
            if not self.allow_missing_modalities and not all(present_mods):
                continue

            # Decide whether this disease has per-case subfolders (case1, case2, ...) under each modality.
            # If any present modality folder contains image files directly -> treat as single-case.
            has_direct_images = False
            subcase_sets: List[set[str]] = []
            for p, ok in zip(mod_dirs, present_mods):
                if not ok or p is None:
                    continue
                if _list_image_paths(p):
                    has_direct_images = True
                subcase_sets.append(set(_list_subdirs(p)))

            if has_direct_images:
                case_names = ["."]
            else:
                # Intersect subcase names across present modalities.
                common = set.intersection(*subcase_sets) if subcase_sets else set()
                if not common:
                    continue
                case_names = sorted(common)

            for case_name in case_names:
                case_id = disease_id if case_name == "." else f"{disease_id}/{case_name}"
                z_to_paths: List[Dict[int, str]] = []
                for p in mod_dirs:
                    if p is None:
                        z_to_paths.append({})
                        continue
                    folder = p if case_name == "." else os.path.join(p, case_name)
                    if not os.path.isdir(folder):
                        z_to_paths.append({})
                        continue
                    try:
                        z_to_paths.append(_build_z_to_path(folder))
                    except ValueError:
                        z_to_paths.append({})

                # Keep z where at least 2 modalities provide this slice.
                z_counts: Dict[int, int] = {}
                for d in z_to_paths:
                    for z in d.keys():
                        z_counts[z] = z_counts.get(z, 0) + 1
                zs = sorted([z for z, c in z_counts.items() if c >= 2])
                for z in zs:
                    paths: List[Optional[str]] = []
                    present: List[bool] = []
                    for d in z_to_paths:
                        p = d.get(z)
                        paths.append(p)
                        present.append(p is not None)
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
        it = self.indices[int(idx)]
        slices: List[Optional[np.ndarray]] = []
        present: List[bool] = []
        for p in it.paths:
            if p is None:
                slices.append(None)
                present.append(False)
                continue
            img = Image.open(p).convert("L")
            if int(self.resize) > 0:
                img = img.resize((int(self.resize), int(self.resize)), resample=Image.BILINEAR)
            img_u8 = np.array(img, dtype=np.uint8)
            slices.append(img_u8.astype(np.float32) / 255.0)
            present.append(True)

        first = next((s for s in slices if s is not None), None)
        if first is None:
            raise RuntimeError("Corrupt sample: no modality present.")

        h, w = first.shape
        if int(self.scale) > 1:
            h = h - h % self.scale
            w = w - w % self.scale
        filled = []
        for s in slices:
            if s is None:
                filled.append(np.zeros((h, w), dtype=np.float32))
            else:
                if s.shape[0] != h or s.shape[1] != w:
                    # Safety: if any modality deviates, resize/crop to target (h,w).
                    # (Should rarely happen when --resize is set.)
                    pil = Image.fromarray((s * 255.0).clip(0, 255).astype(np.uint8), mode="L")
                    pil = pil.resize((int(w), int(h)), resample=Image.BILINEAR)
                    s = (np.array(pil, dtype=np.uint8).astype(np.float32) / 255.0)
                filled.append(s[:h, :w].astype(np.float32))
        x = np.stack(filled, axis=0)  # (M,H,W)
        x_t = torch.from_numpy(x).float().unsqueeze(1)  # (M,1,H,W)
        present_t = torch.tensor(present, dtype=torch.bool)
        return x_t, present_t, it.case_id, int(it.z)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train fusion model on Atlas 2D slices")
    parser.add_argument("--atlas_root", type=str, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--modalities", type=str, nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument(
        "--allow_missing_modalities",
        action="store_true",
        help="Allow missing modalities (useful for mixing SPECT+PET in one run). Missing inputs are zero-filled and masked.",
    )
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE, help="Resize slices to NxN before cropping.")
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--val_fraction", type=float, default=0.25, help="Disease-level validation split fraction.")
    parser.add_argument("--split_seed", type=int, default=3407)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--feat_channels", type=int, default=32)
    parser.add_argument("--lam_ssim", type=float, default=0.5)
    parser.add_argument("--lam_grad", type=float, default=0.3)
    parser.add_argument("--lam_l1", type=float, default=0.2)

    parser.add_argument("--val_batches", type=int, default=0, help="0 means full validation set.")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Patience in epochs. 0 disables.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_warmup", type=int, default=0, help="Ignore early stop for first N epochs.")
    parser.add_argument("--lr_reduce_patience", type=int, default=0, help="Patience in epochs. 0 disables.")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--lr_reduce_min_lr", type=float, default=1e-6)

    parser.add_argument("--out_dir", type=str, default="./checkpoints_atlas/")
    args = parser.parse_args(argv)
    if not hasattr(args, "allow_missing_modalities") or not bool(args.allow_missing_modalities):
        # Default behavior: run in mixed SPECT+PET mode (Atlas typically has diseases with either SPECT or PET).
        args.allow_missing_modalities = True
    return args


def _collate(batch):
    xs, presents, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)
    present = torch.stack(presents, dim=0)
    return x, present, list(case_ids), list(zs)


def _sample_train_subset_from_available(avail: torch.Tensor) -> torch.Tensor:
    # avail: 1D long tensor of available modality indices
    k = int(torch.randint(low=2, high=int(avail.numel()) + 1, size=(1,)).item())
    return avail[torch.randperm(int(avail.numel()), device=avail.device)[:k]]


def save_checkpoint(
    path: str,
    *,
    model: FlexibleFusionNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    args: argparse.Namespace,
    epoch: int,
    step: int,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def run_validation_full_modalities(
    *,
    model: FlexibleFusionNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    mod_ids_full: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "ssim": 0.0, "loss_ssim": 0.0, "loss_grad": 0.0, "loss_l1": 0.0}
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (xm, _present, _case_ids, _zs) in enumerate(loader, start=1):
            if int(args.val_batches) > 0 and batch_idx > int(args.val_batches):
                break
            xm = xm.to(device)
            fused = model(xm, mod_ids=mod_ids_full)
            _, logs = fusion_loss_source_guided(
                fused,
                xm,
                lam_ssim=float(args.lam_ssim),
                lam_grad=float(args.lam_grad),
                lam_l1=float(args.lam_l1),
            )
            for key in sums:
                sums[key] += float(logs[key])
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Validation produced 0 batches. Check your Atlas dataset and --val_batches.")
    return {key: value / num_batches for key, value in sums.items()}


def run_validation_masked(
    *,
    model: FlexibleFusionNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    mod_ids_full: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "ssim": 0.0, "loss_ssim": 0.0, "loss_grad": 0.0, "loss_l1": 0.0}
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (xm, present, _case_ids, _zs) in enumerate(loader, start=1):
            if int(args.val_batches) > 0 and batch_idx > int(args.val_batches):
                break
            xm = xm.to(device)
            present = present.to(device)
            for i in range(xm.shape[0]):
                avail = torch.nonzero(present[i], as_tuple=False).flatten()
                if int(avail.numel()) < 2:
                    continue
                xk = xm[i : i + 1].index_select(dim=1, index=avail)
                fused = model(xk, mod_ids=mod_ids_full.index_select(dim=0, index=avail))
                _, logs = fusion_loss_source_guided(
                    fused,
                    xk,
                    lam_ssim=float(args.lam_ssim),
                    lam_grad=float(args.lam_grad),
                    lam_l1=float(args.lam_l1),
                )
                for key in sums:
                    sums[key] += float(logs[key])
                num_samples += 1

    if num_samples == 0:
        raise RuntimeError("Validation produced 0 usable samples (need >=2 modalities per sample).")
    return {key: value / num_samples for key, value in sums.items()}


def _split_diseases(root: str, *, val_fraction: float, split_seed: int) -> Tuple[List[str], List[str]]:
    diseases = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    diseases.sort()
    if not diseases:
        raise RuntimeError(f"No disease folders found under: {root}")

    frac = float(val_fraction)
    if not (0.0 < frac < 1.0):
        raise RuntimeError("--val_fraction must be in (0,1)")

    g = torch.Generator()
    g.manual_seed(int(split_seed))
    perm = torch.randperm(len(diseases), generator=g).tolist()
    n_val = max(1, int(round(len(diseases) * frac)))
    val_idx = set(perm[:n_val])
    train = [c for i, c in enumerate(diseases) if i not in val_idx]
    val = [c for i, c in enumerate(diseases) if i in val_idx]
    if not train or not val:
        raise RuntimeError(f"Bad split produced train={len(train)} val={len(val)}. Adjust --val_fraction.")
    return train, val


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()

    if int(args.epochs) <= 0:
        raise SystemExit("--epochs must be > 0")
    if not str(args.atlas_root).strip():
        raise SystemExit("--atlas_root is required")
    if not os.path.isdir(args.atlas_root):
        raise SystemExit(f"--atlas_root not found: {args.atlas_root}")

    args.modalities = [m.strip() for m in args.modalities if str(m).strip()]
    if len(args.modalities) < 2:
        raise SystemExit("--modalities must have at least 2 entries")

    args.fuse_mode = "attn"
    args.no_modality_emb = False

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    train_disease_ids, val_disease_ids = _split_diseases(
        args.atlas_root, val_fraction=float(args.val_fraction), split_seed=int(args.split_seed)
    )
    logger.info(f"Atlas diseases | train={len(train_disease_ids)} val={len(val_disease_ids)}")

    train_dataset = Atlas2DImageDataset(
        args.atlas_root,
        modalities=args.modalities,
        scale=int(args.scale),
        resize=int(args.resize),
        disease_ids=train_disease_ids,
        allow_missing_modalities=bool(args.allow_missing_modalities),
    )
    val_dataset = Atlas2DImageDataset(
        args.atlas_root,
        modalities=args.modalities,
        scale=int(args.scale),
        resize=int(args.resize),
        disease_ids=val_disease_ids,
        allow_missing_modalities=bool(args.allow_missing_modalities),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=_collate,
    )

    steps_per_epoch = len(train_loader)
    logger.info(
        f"Train slices: {len(train_dataset)} | Val slices: {len(val_dataset)} | "
        f"Steps per epoch: {steps_per_epoch} | Epochs: {int(args.epochs)}"
    )

    num_modalities = len(args.modalities)
    model = FlexibleFusionNet(
        feat_channels=int(args.feat_channels),
        fuse_mode="attn",
        use_modality_emb=True,
        num_modalities=int(num_modalities),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    scheduler = None
    if int(args.lr_reduce_patience) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_reduce_factor),
            patience=int(args.lr_reduce_patience),
            min_lr=float(args.lr_reduce_min_lr),
        )

    mod_ids_full = torch.arange(num_modalities, dtype=torch.long, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    train_csv = os.path.join(args.out_dir, "train_log.csv")
    val_csv = os.path.join(args.out_dir, "val_log.csv")
    with open(train_csv, "w", newline="", encoding="utf-8") as f_train, open(val_csv, "w", newline="", encoding="utf-8") as f_val:
        train_writer = csv.DictWriter(
            f_train,
            fieldnames=["epoch", "step", "loss", "ssim", "loss_ssim", "loss_grad", "loss_l1", "lr"],
        )
        val_writer = csv.DictWriter(
            f_val,
            fieldnames=["epoch", "step", "val_loss", "val_ssim", "val_loss_ssim", "val_loss_grad", "val_loss_l1", "lr"],
        )
        train_writer.writeheader()
        val_writer.writeheader()

        global_step = 0
        best_val_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0
        started_at = time.time()

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            train_sums = {"loss": 0.0, "ssim": 0.0, "loss_ssim": 0.0, "loss_grad": 0.0, "loss_l1": 0.0}

            for batch_idx, (xm, present, _case_ids, _zs) in enumerate(train_loader, start=1):
                global_step += 1
                xm = xm.to(device)
                present = present.to(device)

                if bool(args.allow_missing_modalities) and not bool(torch.all(present)):
                    losses = []
                    logs_sums = {"loss": 0.0, "ssim": 0.0, "loss_ssim": 0.0, "loss_grad": 0.0, "loss_l1": 0.0}
                    for i in range(xm.shape[0]):
                        avail = torch.nonzero(present[i], as_tuple=False).flatten()
                        if int(avail.numel()) < 2:
                            continue
                        sel = _sample_train_subset_from_available(avail)
                        xk = xm[i : i + 1].index_select(dim=1, index=sel)
                        fused = model(xk, mod_ids=mod_ids_full.index_select(dim=0, index=sel))
                        loss_i, logs_i = fusion_loss_source_guided(
                            fused,
                            xk,
                            lam_ssim=float(args.lam_ssim),
                            lam_grad=float(args.lam_grad),
                            lam_l1=float(args.lam_l1),
                        )
                        losses.append(loss_i)
                        for key in logs_sums:
                            logs_sums[key] += float(logs_i[key])
                    if not losses:
                        continue
                    loss = torch.stack(losses, dim=0).mean()
                    logs = {k: v / len(losses) for k, v in logs_sums.items()}
                else:
                    avail = torch.arange(num_modalities, device=device)
                    sel = _sample_train_subset_from_available(avail)
                    xk = xm.index_select(dim=1, index=sel)
                    fused = model(xk, mod_ids=mod_ids_full.index_select(dim=0, index=sel))
                    loss, logs = fusion_loss_source_guided(
                        fused,
                        xk,
                        lam_ssim=float(args.lam_ssim),
                        lam_grad=float(args.lam_grad),
                        lam_l1=float(args.lam_l1),
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                for key in train_sums:
                    train_sums[key] += float(logs[key])

                if batch_idx % int(args.log_every) == 0 or batch_idx == steps_per_epoch:
                    logger.info(
                        f"Epoch {epoch}/{int(args.epochs)} | Batch {batch_idx}/{steps_per_epoch} | Step {global_step} | "
                        f"loss {float(logs['loss']):.4f} | SSIM {float(logs['ssim']):.4f} | "
                        f"Lssim {float(logs['loss_ssim']):.4f} | Lgrad {float(logs['loss_grad']):.4f} | Ll1 {float(logs['loss_l1']):.4f}"
                    )

            train_logs = {key: value / steps_per_epoch for key, value in train_sums.items()}
            lr_now = float(optimizer.param_groups[0]["lr"])
            train_writer.writerow(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": train_logs["loss"],
                    "ssim": train_logs["ssim"],
                    "loss_ssim": train_logs["loss_ssim"],
                    "loss_grad": train_logs["loss_grad"],
                    "loss_l1": train_logs["loss_l1"],
                    "lr": lr_now,
                }
            )
            f_train.flush()

            if bool(args.allow_missing_modalities):
                val_logs = run_validation_masked(
                    model=model,
                    loader=val_loader,
                    device=device,
                    args=args,
                    mod_ids_full=mod_ids_full,
                )
            else:
                val_logs = run_validation_full_modalities(
                    model=model,
                    loader=val_loader,
                    device=device,
                    args=args,
                    mod_ids_full=mod_ids_full,
                )

            val_writer.writerow(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "val_loss": val_logs["loss"],
                    "val_ssim": val_logs["ssim"],
                    "val_loss_ssim": val_logs["loss_ssim"],
                    "val_loss_grad": val_logs["loss_grad"],
                    "val_loss_l1": val_logs["loss_l1"],
                    "lr": lr_now,
                }
            )
            f_val.flush()

            logger.info(
                f"Epoch {epoch}/{int(args.epochs)} finished | "
                f"train_loss {train_logs['loss']:.4f} | val_loss {val_logs['loss']:.4f} | "
                f"val_ssim {val_logs['ssim']:.4f} | elapsed {(time.time() - started_at) / 60.0:.1f} min"
            )

            improved = float(val_logs["loss"]) < float(best_val_loss) - float(args.early_stop_min_delta)
            if improved:
                best_val_loss = float(val_logs["loss"])
                best_epoch = int(epoch)
                bad_epochs = 0
                save_checkpoint(
                    os.path.join(args.out_dir, "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    epoch=epoch,
                    step=global_step,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                )
                logger.info(f"Best checkpoint updated at epoch {epoch} (val_loss={best_val_loss:.4f})")
            else:
                bad_epochs += 1

            epoch_ckpt = os.path.join(args.out_dir, f"epoch_{epoch:03d}.pth")
            save_checkpoint(
                epoch_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                step=global_step,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )
            save_checkpoint(
                os.path.join(args.out_dir, "latest.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                step=global_step,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )

            if scheduler is not None:
                scheduler.step(float(val_logs["loss"]))
                logger.info(f"LR now: {float(optimizer.param_groups[0]['lr']):g}")

            if epoch <= int(args.early_stop_warmup):
                bad_epochs = 0
            if int(args.early_stop_patience) > 0 and epoch > int(args.early_stop_warmup):
                if bad_epochs >= int(args.early_stop_patience):
                    logger.info(
                        f"Early stopping at epoch {epoch}: no val_loss improvement for {bad_epochs} epochs "
                        f"(best {best_val_loss:.4f} at epoch {best_epoch})."
                    )
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

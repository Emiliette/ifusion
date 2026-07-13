import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from guided_diffusion.flex_em_nmodal import em_fuse_refine
from util.brats_2d_dataset import BraTS2DPngDataset
from util.fusion_losses import fusion_loss_source_guided
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


DEFAULT_TRAIN_ROOT = os.path.join("/storage/student5/ifusion/Intern_ImageFusion/scripts/Dataset/BraTS_2D_Training")
DEFAULT_VAL_ROOT = "/storage/student5/ifusion/Intern_ImageFusion/scripts/Dataset/BraTS_2D_Validation"
DEFAULT_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
DEFAULT_EPOCHS = 100


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Train BraTS 2D fusion model with flexible number of input modalities per step (range + schedule) + optional EM refinement"
    )
    p.add_argument(
        "--train_2d_root",
        type=str,
        default=DEFAULT_TRAIN_ROOT,
        help="Train root with subfolders per modality: <root>/<t1n|t1c|t2w|t2f>/<case>_z###.png",
    )
    p.add_argument(
        "--val_2d_root",
        type=str,
        default=DEFAULT_VAL_ROOT,
        help="Validation root in the same structure as --train_2d_root (required for val/early-stop/LR scheduling).",
    )
    p.add_argument("--modalities", type=str, nargs="+", default=list(DEFAULT_MODALITIES))
    # Kept for backward-compatibility with train_brats_5.py helpers; BraTS2DPngDataset assumes aligned modalities.
    p.add_argument(
        "--allow_missing_modalities",
        action="store_true",
        help="(Ignored for BraTS2DPngDataset) Kept for compatibility with shared K-range helpers.",
    )
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--val_batches", type=int, default=0, help="0 means full validation set.")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--feat_channels", type=int, default=32)
    p.add_argument("--lam_ssim", type=float, default=0.5)
    p.add_argument("--lam_grad", type=float, default=0.3)
    p.add_argument("--lam_l1", type=float, default=0.2)

    p.add_argument(
        "--em_steps",
        type=int,
        default=0,
        help="Number of EM refinement steps applied on top of fused_init. 0 disables EM refinement.",
    )
    p.add_argument("--em_eta", type=float, default=0.2)
    p.add_argument("--em_rho", type=float, default=0.02)
    p.add_argument("--em_alpha_grad", type=float, default=1.0)
    p.add_argument("--em_beta_residual", type=float, default=1.0)
    p.add_argument(
        "--loss_init_weight",
        type=float,
        default=0.0,
        help="Optional auxiliary loss weight for fused_init (helps gradients reach FlexibleFusionNet when EM is enabled).",
    )

    # Flexible-K sampling controls (static range).
    p.add_argument("--min_modalities_per_sample", type=int, default=0, help="Min K sampled per step. 0=auto.")
    p.add_argument(
        "--max_modalities_per_sample",
        type=int,
        default=0,
        help="Max K sampled per step. 0=all available for that sample.",
    )

    # Optional schedule for (k_min, k_max) over epochs.
    p.add_argument(
        "--k_schedule",
        type=str,
        default="none",
        choices=("none", "linear"),
        help="If not 'none', linearly interpolate K range from start->end over epochs.",
    )
    p.add_argument("--k_min_start", type=int, default=0, help="0=auto.")
    p.add_argument("--k_min_end", type=int, default=0, help="0=auto (usually 2).")
    p.add_argument("--k_max_start", type=int, default=0, help="0=auto (usually num_modalities).")
    p.add_argument("--k_max_end", type=int, default=0, help="0=auto (usually num_modalities).")
    p.add_argument(
        "--k_epoch_cover",
        action="store_true",
        help="Ensure each epoch covers all K in [k_min,k_max] by cycling K per batch (then repeating).",
    )
    p.add_argument(
        "--k_epoch_shuffle",
        action="store_const",
        const=True,
        default=None,
        help="When --k_epoch_cover is enabled, shuffle the K cycle order per epoch (seeded). Default: enabled.",
    )
    p.add_argument(
        "--k_epoch_no_shuffle",
        dest="k_epoch_shuffle",
        action="store_const",
        const=False,
        default=None,
        help="Disable K shuffle when --k_epoch_cover is enabled.",
    )

    # Lightweight preprocessing / augmentation toggles.
    p.add_argument(
        "--preproc",
        type=str,
        default="percentile",
        choices=("none", "percentile"),
        help="Optional per-image preprocessing. 'percentile' does percentile contrast stretching.",
    )
    p.add_argument("--preproc_p_low", type=float, default=0.01, help="Lower quantile for percentile stretch.")
    p.add_argument("--preproc_p_high", type=float, default=0.99, help="Upper quantile for percentile stretch.")
    p.add_argument("--aug_prob", type=float, default=0.5, help="Augment probability per modality image (0 disables).")
    p.add_argument("--aug_gamma", type=float, default=0.25, help="Gamma jitter range: [1-g, 1+g].")
    p.add_argument("--aug_brightness", type=float, default=0.06, help="Brightness jitter range: +/- value.")
    p.add_argument("--aug_noise_std", type=float, default=0.02, help="Gaussian noise std max (0 disables).")

    p.add_argument("--early_stop_patience", type=int, default=0, help="Patience in epochs. 0 disables.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--early_stop_warmup", type=int, default=0, help="Ignore early stop for first N epochs.")
    p.add_argument("--lr_reduce_patience", type=int, default=0, help="Patience in epochs. 0 disables.")
    p.add_argument("--lr_reduce_factor", type=float, default=0.5)
    p.add_argument("--lr_reduce_min_lr", type=float, default=1e-6)

    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    p.add_argument("--out_dir", type=str, default="./checkpoints_brats_6_26/")
    return p.parse_args(argv)


def _collate(batch):
    xs, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)
    return x, list(case_ids), list(zs)


def _sample_subset_from_available(
    avail: torch.Tensor,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    # avail: 1D long tensor of available modality indices.
    n = int(avail.numel())
    if n < 2:
        raise ValueError("Need at least 2 available modalities to sample a subset.")
    min_k = max(2, int(min_k))
    max_k = int(max_k)
    if max_k <= 0:
        max_k = n
    max_k = max(2, min(max_k, n))
    min_k = min(min_k, max_k)
    k = int(torch.randint(low=min_k, high=max_k + 1, size=(1,), device=avail.device).item())
    return avail[torch.randperm(n, device=avail.device)[:k]]


def _split_cases(root: str, *, val_fraction: float, split_seed: int) -> Tuple[List[str], List[str]]:
    cases = set()
    for n in os.listdir(root):
        folder = os.path.join(root, n)
        if not os.path.isdir(folder):
            continue
        try:
            case_id, _z = _parse_slice_dir_name(n)
        except ValueError:
            continue
        cases.add(case_id)
    cases = sorted(cases)
    if not cases:
        raise RuntimeError(f"No slice folders found under: {root}")

    frac = float(val_fraction)
    if not (0.0 < frac < 1.0):
        raise RuntimeError("--val_fraction must be in (0,1)")

    g = torch.Generator()
    g.manual_seed(int(split_seed))
    perm = torch.randperm(len(cases), generator=g).tolist()
    n_val = max(1, int(round(len(cases) * frac)))
    val_idx = set(perm[:n_val])
    train = [c for i, c in enumerate(cases) if i not in val_idx]
    val = [c for i, c in enumerate(cases) if i in val_idx]
    if not train or not val:
        raise RuntimeError(f"Bad split produced train={len(train)} val={len(val)}. Adjust --val_fraction.")
    return train, val


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
    k_min: int,
    k_max: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "k_min": int(k_min),
        "k_max": int(k_max),
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
        for batch_idx, (xm, _case_ids, _zs) in enumerate(loader, start=1):
            if int(args.val_batches) > 0 and batch_idx > int(args.val_batches):
                break
            xm = xm.to(device)
            fused_init = model(xm, mod_ids=mod_ids_full)
            fused = fused_init
            if int(args.em_steps) > 0:
                fused, _aux = em_fuse_refine(
                    fused_init,
                    xm,
                    steps=int(args.em_steps),
                    eta=float(args.em_eta),
                    rho=float(args.em_rho),
                    alpha_grad=float(args.em_alpha_grad),
                    beta_residual=float(args.em_beta_residual),
                )
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
        raise RuntimeError("Validation produced 0 batches. Check your dataset and --val_batches.")
    return {key: value / num_batches for key, value in sums.items()}


def _default_k_range(num_modalities: int, *, allow_missing_modalities: bool) -> Tuple[int, int]:
    k_min = 2
    k_max = int(num_modalities)
    if int(num_modalities) == 3:
        k_max = 3
    elif int(num_modalities) == 4:
        k_max = 4
    _ = bool(allow_missing_modalities)
    return int(k_min), int(k_max)


def _resolve_epoch_k_range(
    *,
    epoch: int,
    epochs_total: int,
    args: argparse.Namespace,
    num_modalities: int,
) -> Tuple[int, int]:
    def _norm_k_min(k: int) -> int:
        return max(2, int(k))

    def _norm_k_max(k: int) -> int:
        if int(k) == 0:
            return 0
        return max(2, min(int(k), int(num_modalities)))

    default_min, default_max = _default_k_range(int(num_modalities), allow_missing_modalities=bool(args.allow_missing_modalities))

    if str(args.k_schedule).lower() == "none":
        k_min = int(args.min_modalities_per_sample) if int(args.min_modalities_per_sample) > 0 else int(default_min)
        k_max = int(args.max_modalities_per_sample) if int(args.max_modalities_per_sample) != 0 else int(default_max)
        k_min = _norm_k_min(k_min)
        k_max = _norm_k_max(k_max)
        if k_max != 0 and k_min > k_max:
            k_min = k_max
        return int(k_min), int(k_max)

    e = int(epoch)
    t = 0.0
    if int(epochs_total) > 1:
        t = float(e - 1) / float(int(epochs_total) - 1)
    t = max(0.0, min(1.0, t))

    kmin_s = int(args.k_min_start) if int(args.k_min_start) > 0 else int(default_min)
    kmin_e = int(args.k_min_end) if int(args.k_min_end) > 0 else int(default_min)
    kmax_s = int(args.k_max_start) if int(args.k_max_start) != 0 else int(default_max)
    kmax_e = int(args.k_max_end) if int(args.k_max_end) != 0 else int(default_max)

    k_min = int(round((1.0 - t) * float(kmin_s) + t * float(kmin_e)))
    if int(kmax_s) == 0 or int(kmax_e) == 0:
        k_max = 0
    else:
        k_max = int(round((1.0 - t) * float(kmax_s) + t * float(kmax_e)))

    k_min = _norm_k_min(k_min)
    k_max = _norm_k_max(k_max)
    if k_max != 0 and k_min > k_max:
        k_min = k_max
    return int(k_min), int(k_max)


def _epoch_k_targets(
    *,
    steps_per_epoch: int,
    k_min: int,
    k_max: int,
    seed: int,
    epoch: int,
    device: torch.device,
    shuffle: bool,
) -> List[int]:
    if int(steps_per_epoch) <= 0:
        return []
    k_min_i = max(2, int(k_min))
    k_max_i = max(k_min_i, int(k_max))
    base = list(range(k_min_i, k_max_i + 1))
    out = (base * ((int(steps_per_epoch) + len(base) - 1) // len(base)))[: int(steps_per_epoch)]
    if shuffle and len(out) > 1:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + int(epoch) * 1009)
        perm = torch.randperm(len(out), generator=g).tolist()
        out = [out[i] for i in perm]
    return out


@dataclass(frozen=True)
class _AugParams:
    p: float
    gamma: float
    brightness: float
    noise_std: float


def _percentile_stretch01(x01: torch.Tensor, *, p_low: float, p_high: float) -> torch.Tensor:
    flat = x01.reshape(-1)
    lo = torch.quantile(flat, float(p_low))
    hi = torch.quantile(flat, float(p_high))
    denom = (hi - lo).clamp_min(1e-6)
    y = (x01 - lo) / denom
    return torch.clamp(y, 0.0, 1.0)


def _apply_preproc_and_aug(
    xm: torch.Tensor,
    *,
    args: argparse.Namespace,
    aug: _AugParams,
) -> torch.Tensor:
    # xm: (B,M,1,H,W)
    out = xm
    if str(args.preproc).lower() == "percentile":
        p_low = float(args.preproc_p_low)
        p_high = float(args.preproc_p_high)
        if not (0.0 <= p_low < p_high <= 1.0):
            raise ValueError("--preproc_p_low/high must satisfy 0<=low<high<=1")
        b, m, _, _, _ = out.shape
        out2 = out.clone()
        for bi in range(int(b)):
            for mi in range(int(m)):
                out2[bi, mi, 0] = _percentile_stretch01(out2[bi, mi, 0], p_low=p_low, p_high=p_high)
        out = out2

    if float(aug.p) > 0.0:
        b, m, _, _, _ = out.shape
        out2 = out.clone()
        for bi in range(int(b)):
            for mi in range(int(m)):
                if float(torch.rand((), device=out.device).item()) > float(aug.p):
                    continue
                img = out2[bi, mi, 0]
                if float(aug.gamma) > 0.0:
                    g = float(aug.gamma)
                    gamma = float(torch.empty((), device=out.device).uniform_(1.0 - g, 1.0 + g).item())
                    img = torch.clamp(img, 0.0, 1.0).pow(gamma)
                if float(aug.brightness) > 0.0:
                    bmax = float(aug.brightness)
                    delta = float(torch.empty((), device=out.device).uniform_(-bmax, bmax).item())
                    img = img + delta
                if float(aug.noise_std) > 0.0:
                    smax = float(aug.noise_std)
                    sigma = float(torch.empty((), device=out.device).uniform_(0.0, smax).item())
                    img = img + sigma * torch.randn_like(img)
                out2[bi, mi, 0] = torch.clamp(img, 0.0, 1.0)
        out = out2

    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()

    if int(args.epochs) <= 0:
        raise SystemExit("--epochs must be > 0")
    if not str(args.train_2d_root).strip():
        raise SystemExit("--train_2d_root is required")
    if not os.path.isdir(args.train_2d_root):
        raise SystemExit(f"--train_2d_root not found: {args.train_2d_root}")
    if not str(args.val_2d_root).strip():
        raise SystemExit("--val_2d_root is required for this script")
    if not os.path.isdir(args.val_2d_root):
        raise SystemExit(f"--val_2d_root not found: {args.val_2d_root}")

    args.modalities = [str(m).strip().lower() for m in args.modalities if str(m).strip()]
    if len(args.modalities) < 2:
        raise SystemExit("--modalities must have at least 2 entries")
    if len(set(args.modalities)) != len(args.modalities):
        raise SystemExit("--modalities contains duplicates")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Train root: {os.path.abspath(args.train_2d_root)}")
    logger.info(f"Val root: {os.path.abspath(args.val_2d_root)}")

    train_dataset = BraTS2DPngDataset(args.train_2d_root, modalities=args.modalities, scale=int(args.scale))
    val_dataset = BraTS2DPngDataset(args.val_2d_root, modalities=args.modalities, scale=int(args.scale))
    logger.info(f"Train slices: {len(train_dataset)} | Val slices: {len(val_dataset)}")

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
        f"Steps per epoch: {steps_per_epoch} | Epochs: {int(args.epochs)} | "
        f"Modalities ({len(args.modalities)}): {', '.join(args.modalities)}"
    )

    num_modalities = len(args.modalities)
    mod_ids_full = torch.arange(num_modalities, dtype=torch.long, device=device)

    model = FlexibleFusionNet(
        feat_channels=int(args.feat_channels),
        fuse_mode="attn",
        use_modality_emb=True,
        num_modalities=num_modalities,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    scheduler = None

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0

    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)

        logger.info(
            f"Resumed from epoch {ckpt['epoch']} "
            f"(next epoch: {start_epoch})"
        )

    if int(args.lr_reduce_patience) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_reduce_factor),
            patience=int(args.lr_reduce_patience),
            min_lr=float(args.lr_reduce_min_lr),
        )

    os.makedirs(args.out_dir, exist_ok=True)
    train_csv = os.path.join(args.out_dir, "train_log.csv")
    val_csv = os.path.join(args.out_dir, "val_log.csv")

    aug = _AugParams(
        p=float(args.aug_prob),
        gamma=float(args.aug_gamma),
        brightness=float(args.aug_brightness),
        noise_std=float(args.aug_noise_std),
    )

    with open(train_csv, "w", newline="", encoding="utf-8") as f_train, open(val_csv, "w", newline="", encoding="utf-8") as f_val:
        train_writer = csv.DictWriter(
            f_train,
            fieldnames=[
                "epoch",
                "step",
                "loss",
                "ssim",
                "loss_ssim",
                "loss_grad",
                "loss_l1",
                "lr",
                "k_min",
                "k_max",
                "k_schedule",
                "preproc",
                "aug_prob",
            ],
        )
        val_writer = csv.DictWriter(
            f_val,
            fieldnames=[
                "epoch",
                "step",
                "val_loss",
                "val_ssim",
                "val_loss_ssim",
                "val_loss_grad",
                "val_loss_l1",
                "lr",
                "k_min",
                "k_max",
            ],
        )
        train_writer.writeheader()
        val_writer.writeheader()

        global_step = 0
        best_val_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0
        started_at = time.time()

        for epoch in range(start_epoch, int(args.epochs) + 1):
            k_min, k_max = _resolve_epoch_k_range(
                epoch=int(epoch), epochs_total=int(args.epochs), args=args, num_modalities=int(num_modalities)
            )
            k_targets: Optional[List[int]] = None
            if bool(args.k_epoch_cover):
                if int(k_max) == 0:
                    raise SystemExit(
                        "--k_epoch_cover requires an explicit finite k_max (set --max_modalities_per_sample or --k_max_*)."
                    )
                shuffle = True if args.k_epoch_shuffle is None else bool(args.k_epoch_shuffle)
                k_targets = _epoch_k_targets(
                    steps_per_epoch=int(steps_per_epoch),
                    k_min=int(k_min),
                    k_max=int(k_max),
                    seed=int(args.seed),
                    epoch=int(epoch),
                    device=device,
                    shuffle=shuffle,
                )

            model.train()
            train_sums = {"loss": 0.0, "ssim": 0.0, "loss_ssim": 0.0, "loss_grad": 0.0, "loss_l1": 0.0}

            for batch_idx, (xm, _case_ids, _zs) in enumerate(train_loader, start=1):
                global_step += 1
                k_target = int(k_targets[batch_idx - 1]) if k_targets is not None else None
                xm = xm.to(device)

                xm = _apply_preproc_and_aug(xm, args=args, aug=aug)

                avail = torch.arange(num_modalities, device=device)
                if k_target is None:
                    sel = _sample_subset_from_available(avail, min_k=int(k_min), max_k=int(k_max))
                else:
                    k_eff = min(int(k_target), int(avail.numel()))
                    k_eff = max(2, int(k_eff))
                    sel = avail[torch.randperm(int(avail.numel()), device=avail.device)[: int(k_eff)]]
                xk = xm.index_select(dim=1, index=sel)
                fused_init = model(xk, mod_ids=mod_ids_full.index_select(dim=0, index=sel))
                fused = fused_init
                if int(args.em_steps) > 0:
                    fused, _aux = em_fuse_refine(
                        fused_init,
                        xk,
                        steps=int(args.em_steps),
                        eta=float(args.em_eta),
                        rho=float(args.em_rho),
                        alpha_grad=float(args.em_alpha_grad),
                        beta_residual=float(args.em_beta_residual),
                    )
                loss_refined, logs = fusion_loss_source_guided(
                    fused,
                    xk,
                    lam_ssim=float(args.lam_ssim),
                    lam_grad=float(args.lam_grad),
                    lam_l1=float(args.lam_l1),
                )
                loss = loss_refined
                if float(args.loss_init_weight) > 0.0:
                    loss_init, _logs_init = fusion_loss_source_guided(
                        fused_init,
                        xk,
                        lam_ssim=float(args.lam_ssim),
                        lam_grad=float(args.lam_grad),
                        lam_l1=float(args.lam_l1),
                    )
                    loss = loss + float(args.loss_init_weight) * loss_init

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                for key in train_sums:
                    train_sums[key] += float(logs[key])

                if batch_idx % int(args.log_every) == 0 or batch_idx == steps_per_epoch:
                    logger.info(
                        f"Epoch {epoch}/{int(args.epochs)} | Batch {batch_idx}/{steps_per_epoch} | Step {global_step} | "
                        f"k=[{k_min},{k_max if k_max != 0 else 'all'}]"
                        + (f" | k_target {k_target}" if k_target is not None else "")
                        + " | "
                        f"loss {logs['loss']:.4f} | SSIM {logs['ssim']:.4f} | "
                        f"Lssim {logs['loss_ssim']:.4f} | Lgrad {logs['loss_grad']:.4f} | Ll1 {logs['loss_l1']:.4f}"
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
                    "k_min": int(k_min),
                    "k_max": int(k_max),
                    "k_schedule": str(args.k_schedule),
                    "preproc": str(args.preproc),
                    "aug_prob": float(args.aug_prob),
                }
            )
            f_train.flush()

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
                    "k_min": int(k_min),
                    "k_max": int(k_max),
                }
            )
            f_val.flush()

            save_checkpoint(
                os.path.join(args.out_dir, "last.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                step=global_step,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                k_min=int(k_min),
                k_max=int(k_max),
            )

            improved = float(best_val_loss) - float(val_logs["loss"]) > float(args.early_stop_min_delta)
            if improved:
                best_val_loss = float(val_logs["loss"])
                best_epoch = int(epoch)
                bad_epochs = 0
                save_checkpoint(
                    os.path.join(args.out_dir, "best_26.pt"),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    epoch=epoch,
                    step=global_step,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    k_min=int(k_min),
                    k_max=int(k_max),
                )
            else:
                bad_epochs += 1

            elapsed = time.time() - started_at
            logger.info(
                f"Epoch {epoch}/{int(args.epochs)} done | "
                f"k=[{k_min},{k_max if k_max != 0 else 'all'}] | "
                f"train_loss {train_logs['loss']:.4f} | val_loss {val_logs['loss']:.4f} | "
                f"best {best_val_loss:.4f} @ epoch {best_epoch} | bad_epochs {bad_epochs} | "
                f"elapsed {elapsed/60.0:.1f} min"
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

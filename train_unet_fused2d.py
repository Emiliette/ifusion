import argparse
import csv
import os
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from util.fused_2d_seg_dataset import BraTSFused2DSegDataset
from util.logger import get_logger
from util.unet_seg import UNet2D


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Train 2D U-Net segmentation on fused BraTS PNG slices")
    p.add_argument("--train_fused_dir", type=str, required=True, help="Folder with fused PNGs: <case>_z###.png")
    p.add_argument("--train_seg_dir", type=str, required=True, help="Folder with seg PNGs: <case>_z###.png")
    p.add_argument("--val_fused_dir", type=str, required=True)
    p.add_argument("--val_seg_dir", type=str, required=True)

    p.add_argument("--num_classes", type=int, default=4, help="Default 4 for BraTS labels {0,1,2,3} (with 4->3 remap).")
    p.add_argument("--remap_4_to_3", action="store_true", help="Remap label 4 -> 3 (recommended).")
    p.add_argument("--no_remap_4_to_3", dest="remap_4_to_3", action="store_false")
    p.set_defaults(remap_4_to_3=True)

    p.add_argument("--scale", type=int, default=30, help="Crop H/W to multiples of this value.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--base_channels", type=int, default=32)

    p.add_argument("--lam_dice", type=float, default=1.0)
    p.add_argument("--lam_ce", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="./checkpoints_unet_fused2d/")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args(argv)


def _collate(batch):
    xs, ys, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)  # (N,1,H,W)
    y = torch.stack(ys, dim=0)  # (N,H,W)
    return x, y, list(case_ids), list(zs)


def dice_score_from_logits(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns per-class Dice for classes [0..C-1] as a (C,) tensor.
    """
    if logits.dim() != 4:
        raise ValueError("logits must be (N,C,H,W)")
    n, c, h, w = logits.shape
    if c != int(num_classes):
        raise ValueError(f"Expected num_classes={num_classes}, got logits C={c}")
    if y.shape != (n, h, w):
        raise ValueError(f"Expected y shape {(n,h,w)}, got {tuple(y.shape)}")

    pred = torch.argmax(logits, dim=1)  # (N,H,W)
    dice = []
    for cls in range(int(num_classes)):
        p = (pred == cls).float()
        t = (y == cls).float()
        inter = torch.sum(p * t)
        denom = torch.sum(p) + torch.sum(t)
        dice.append((2.0 * inter + eps) / (denom + eps))
    return torch.stack(dice, dim=0)


def soft_dice_loss(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss over foreground classes (1..C-1).
    """
    probs = torch.softmax(logits, dim=1)  # (N,C,H,W)
    y_oh = F.one_hot(y.clamp_min(0), num_classes=int(num_classes)).permute(0, 3, 1, 2).float()  # (N,C,H,W)
    losses = []
    for cls in range(1, int(num_classes)):
        p = probs[:, cls]
        t = y_oh[:, cls]
        inter = torch.sum(p * t)
        denom = torch.sum(p) + torch.sum(t)
        d = (2.0 * inter + eps) / (denom + eps)
        losses.append(1.0 - d)
    if not losses:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return torch.mean(torch.stack(losses, dim=0))


def save_checkpoint(
    path: str,
    *,
    model: UNet2D,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    best_val_dice: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_dice": float(best_val_dice),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def run_one_epoch(
    *,
    model: UNet2D,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    num_classes: int,
    lam_dice: float,
    lam_ce: float,
    log_every: int,
    logger,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sums: Dict[str, float] = {"loss": 0.0, "loss_ce": 0.0, "loss_dice": 0.0, "dice_fg_mean": 0.0}
    steps = 0

    for batch_idx, (x, y, _case_ids, _zs) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss_ce = F.cross_entropy(logits, y)
        loss_dice = soft_dice_loss(logits, y, num_classes=num_classes)
        loss = float(lam_ce) * loss_ce + float(lam_dice) * loss_dice

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            dice = dice_score_from_logits(logits, y, num_classes=num_classes)
            dice_fg = dice[1:] if num_classes > 1 else dice
            dice_fg_mean = float(torch.mean(dice_fg).item()) if dice_fg.numel() else 0.0

        sums["loss"] += float(loss.item())
        sums["loss_ce"] += float(loss_ce.item())
        sums["loss_dice"] += float(loss_dice.item())
        sums["dice_fg_mean"] += dice_fg_mean
        steps += 1

        if is_train and (batch_idx % int(log_every) == 0):
            logger.info(
                f"Batch {batch_idx}/{len(loader)} | loss {loss.item():.4f} | ce {loss_ce.item():.4f} | "
                f"dice_loss {loss_dice.item():.4f} | dice_fg {dice_fg_mean:.4f}"
            )

    if steps == 0:
        raise RuntimeError("0 steps in epoch (empty loader).")
    return {k: v / steps for k, v in sums.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    train_ds = BraTSFused2DSegDataset(
        args.train_fused_dir,
        args.train_seg_dir,
        scale=int(args.scale),
        remap_4_to_3=bool(args.remap_4_to_3),
    )
    val_ds = BraTSFused2DSegDataset(
        args.val_fused_dir,
        args.val_seg_dir,
        scale=int(args.scale),
        remap_4_to_3=bool(args.remap_4_to_3),
    )
    logger.info(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=_collate,
    )

    model = UNet2D(in_channels=1, num_classes=int(args.num_classes), base_channels=int(args.base_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_csv = os.path.join(args.out_dir, "train_log.csv")
    val_csv = os.path.join(args.out_dir, "val_log.csv")
    with open(train_csv, "w", newline="", encoding="utf-8") as f_train, open(val_csv, "w", newline="", encoding="utf-8") as f_val:
        train_writer = csv.DictWriter(f_train, fieldnames=["epoch", "loss", "loss_ce", "loss_dice", "dice_fg_mean", "lr"])
        val_writer = csv.DictWriter(f_val, fieldnames=["epoch", "val_loss", "val_loss_ce", "val_loss_dice", "val_dice_fg_mean", "lr"])
        train_writer.writeheader()
        val_writer.writeheader()

        best_val_dice = -1.0
        started = time.time()
        for epoch in range(1, int(args.epochs) + 1):
            lr_now = float(optimizer.param_groups[0]["lr"])
            logger.info(f"Epoch {epoch}/{int(args.epochs)} | lr {lr_now:g}")

            train_logs = run_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                num_classes=int(args.num_classes),
                lam_dice=float(args.lam_dice),
                lam_ce=float(args.lam_ce),
                log_every=int(args.log_every),
                logger=logger,
            )
            val_logs = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                num_classes=int(args.num_classes),
                lam_dice=float(args.lam_dice),
                lam_ce=float(args.lam_ce),
                log_every=int(args.log_every),
                logger=logger,
            )

            train_writer.writerow(
                {
                    "epoch": epoch,
                    "loss": train_logs["loss"],
                    "loss_ce": train_logs["loss_ce"],
                    "loss_dice": train_logs["loss_dice"],
                    "dice_fg_mean": train_logs["dice_fg_mean"],
                    "lr": lr_now,
                }
            )
            val_writer.writerow(
                {
                    "epoch": epoch,
                    "val_loss": val_logs["loss"],
                    "val_loss_ce": val_logs["loss_ce"],
                    "val_loss_dice": val_logs["loss_dice"],
                    "val_dice_fg_mean": val_logs["dice_fg_mean"],
                    "lr": lr_now,
                }
            )
            f_train.flush()
            f_val.flush()

            logger.info(
                f"Train: loss {train_logs['loss']:.4f} dice_fg {train_logs['dice_fg_mean']:.4f} | "
                f"Val: loss {val_logs['loss']:.4f} dice_fg {val_logs['dice_fg_mean']:.4f}"
            )

            save_checkpoint(
                os.path.join(args.out_dir, "latest.pth"),
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                best_val_dice=best_val_dice,
            )
            if float(val_logs["dice_fg_mean"]) > best_val_dice:
                best_val_dice = float(val_logs["dice_fg_mean"])
                save_checkpoint(
                    os.path.join(args.out_dir, "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    best_val_dice=best_val_dice,
                )
                logger.info(f"New best val dice_fg_mean: {best_val_dice:.4f}")

        logger.info(f"Done in {time.time() - started:.1f}s. Best val dice_fg_mean={best_val_dice:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


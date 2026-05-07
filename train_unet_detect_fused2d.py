import argparse
import csv
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from util.fused_2d_detect_dataset import BraTSFused2DDetectDataset
from util.logger import get_logger
from util.unet_detect import UNetDetector2D


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Train U-Net style detector on fused BraTS 2D PNG slices (binary detect, no segmentation)")
    p.add_argument("--train_fused_dir", type=str, required=True, help="Folder with fused PNGs: <case>_z###.png")
    p.add_argument("--train_seg_dir", type=str, required=True, help="Folder with seg PNGs (to build labels): <case>_z###.png")
    p.add_argument("--val_fused_dir", type=str, required=True)
    p.add_argument("--val_seg_dir", type=str, required=True)

    p.add_argument("--scale", type=int, default=30, help="Crop H/W to multiples of this value.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--pos_weight", type=float, default=0.0, help="If >0, used as BCE pos_weight to handle imbalance.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on sigmoid(prob).")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="./checkpoints_unet_detect_fused2d/")
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args(argv)


def _collate(batch):
    xs, ys, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)  # (N,1,H,W)
    y = torch.stack([t.view(1) for t in ys], dim=0).squeeze(1)  # (N,)
    return x, y, list(case_ids), list(zs)


def _metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, *, threshold: float) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= float(threshold)).float()
    y = y.float()

    tp = torch.sum((pred == 1) & (y == 1)).item()
    tn = torch.sum((pred == 0) & (y == 0)).item()
    fp = torch.sum((pred == 1) & (y == 0)).item()
    fn = torch.sum((pred == 0) & (y == 1)).item()
    total = tp + tn + fp + fn + 1e-12

    acc = (tp + tn) / total
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1)}


def save_checkpoint(
    path: str,
    *,
    model: UNetDetector2D,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    best_val_f1: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_f1": float(best_val_f1),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def run_one_epoch(
    *,
    model: UNetDetector2D,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    threshold: float,
    pos_weight: float,
    log_every: int,
    logger,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sums = {"loss": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
    steps = 0

    pw = None
    if float(pos_weight) > 0:
        pw = torch.tensor([float(pos_weight)], device=device, dtype=torch.float32)

    for batch_idx, (x, y, _case_ids, _zs) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            m = _metrics_from_logits(logits, y, threshold=threshold)

        sums["loss"] += float(loss.item())
        for k in ("acc", "prec", "rec", "f1"):
            sums[k] += float(m[k])
        steps += 1

        if is_train and (batch_idx % int(log_every) == 0):
            logger.info(
                f"Batch {batch_idx}/{len(loader)} | loss {loss.item():.4f} | acc {m['acc']:.3f} | "
                f"prec {m['prec']:.3f} rec {m['rec']:.3f} f1 {m['f1']:.3f}"
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
    train_ds = BraTSFused2DDetectDataset(
        args.train_fused_dir,
        args.train_seg_dir,
        scale=int(args.scale),
    )
    val_ds = BraTSFused2DDetectDataset(
        args.val_fused_dir,
        args.val_seg_dir,
        scale=int(args.scale),
    )

    train_pos = float(np.mean(train_ds.labels)) if len(train_ds) else 0.0
    val_pos = float(np.mean(val_ds.labels)) if len(val_ds) else 0.0
    logger.info(f"Train slices: {len(train_ds)} (pos_rate={train_pos:.4f}) | Val slices: {len(val_ds)} (pos_rate={val_pos:.4f})")
    if float(args.pos_weight) <= 0 and train_pos > 0:
        suggested = (1.0 - train_pos) / (train_pos + 1e-12)
        logger.info(f"Tip: imbalance detected; try --pos_weight {suggested:.3f}")

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

    model = UNetDetector2D(in_channels=1, base_channels=int(args.base_channels), dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_csv = os.path.join(args.out_dir, "train_log.csv")
    val_csv = os.path.join(args.out_dir, "val_log.csv")
    with open(train_csv, "w", newline="", encoding="utf-8") as f_train, open(val_csv, "w", newline="", encoding="utf-8") as f_val:
        train_writer = csv.DictWriter(f_train, fieldnames=["epoch", "loss", "acc", "prec", "rec", "f1", "lr"])
        val_writer = csv.DictWriter(f_val, fieldnames=["epoch", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "lr"])
        train_writer.writeheader()
        val_writer.writeheader()

        best_val_f1 = -1.0
        started = time.time()
        for epoch in range(1, int(args.epochs) + 1):
            lr_now = float(optimizer.param_groups[0]["lr"])
            logger.info(f"Epoch {epoch}/{int(args.epochs)} | lr {lr_now:g}")

            train_logs = run_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                threshold=float(args.threshold),
                pos_weight=float(args.pos_weight),
                log_every=int(args.log_every),
                logger=logger,
            )
            val_logs = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                threshold=float(args.threshold),
                pos_weight=float(args.pos_weight),
                log_every=int(args.log_every),
                logger=logger,
            )

            train_writer.writerow(
                {
                    "epoch": epoch,
                    "loss": train_logs["loss"],
                    "acc": train_logs["acc"],
                    "prec": train_logs["prec"],
                    "rec": train_logs["rec"],
                    "f1": train_logs["f1"],
                    "lr": lr_now,
                }
            )
            val_writer.writerow(
                {
                    "epoch": epoch,
                    "val_loss": val_logs["loss"],
                    "val_acc": val_logs["acc"],
                    "val_prec": val_logs["prec"],
                    "val_rec": val_logs["rec"],
                    "val_f1": val_logs["f1"],
                    "lr": lr_now,
                }
            )
            f_train.flush()
            f_val.flush()

            logger.info(
                f"Train: loss {train_logs['loss']:.4f} f1 {train_logs['f1']:.3f} | "
                f"Val: loss {val_logs['loss']:.4f} f1 {val_logs['f1']:.3f}"
            )

            save_checkpoint(
                os.path.join(args.out_dir, "latest.pth"),
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                best_val_f1=best_val_f1,
            )
            if float(val_logs["f1"]) > best_val_f1:
                best_val_f1 = float(val_logs["f1"])
                save_checkpoint(
                    os.path.join(args.out_dir, "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    best_val_f1=best_val_f1,
                )
                logger.info(f"New best val f1: {best_val_f1:.3f}")

        logger.info(f"Done in {time.time() - started:.1f}s. Best val f1={best_val_f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


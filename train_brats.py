import argparse
import csv
import os
import time
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from util.brats_2d_dataset import BraTS2DPngDataset
from util.fusion_losses import fusion_loss_source_guided
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


DEFAULT_TRAIN_ROOT = os.path.join("Dataset-BraTS-2D", "Dataset_BraTS_2D", "BraTS_2D")
DEFAULT_VAL_ROOT = ""
DEFAULT_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
DEFAULT_EPOCHS = 100


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train BraTS fusion model on 2D PNG slices")
    parser.add_argument("--train_2d_root", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--val_2d_root", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--modalities", type=str, nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)

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

    parser.add_argument("--out_dir", type=str, default="./checkpoints_brats/")
    return parser.parse_args(argv)


def _collate(batch):
    xs, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)
    return x, list(case_ids), list(zs)


def _sample_train_subset(total_modalities: int, device: torch.device) -> torch.Tensor:
    k = int(torch.randint(low=2, high=total_modalities + 1, size=(1,)).item())
    return torch.randperm(total_modalities, device=device)[:k]


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


def run_validation(
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
        for batch_idx, (x4, _case_ids, _zs) in enumerate(loader, start=1):
            if int(args.val_batches) > 0 and batch_idx > int(args.val_batches):
                break
            x4 = x4.to(device)
            fused = model(x4, mod_ids=mod_ids_full)
            _, logs = fusion_loss_source_guided(
                fused,
                x4,
                lam_ssim=float(args.lam_ssim),
                lam_grad=float(args.lam_grad),
                lam_l1=float(args.lam_l1),
            )
            for key in sums:
                sums[key] += float(logs[key])
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Validation produced 0 batches. Check --val_2d_root and --val_batches.")
    return {key: value / num_batches for key, value in sums.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()

    if int(args.epochs) <= 0:
        raise SystemExit("--epochs must be > 0")
    if not str(args.train_2d_root).strip():
        raise SystemExit("--train_2d_root is required")
    if not str(args.val_2d_root).strip():
        raise SystemExit("--val_2d_root is required")

    args.modalities = [m.strip().lower() for m in args.modalities]
    if args.modalities != list(DEFAULT_MODALITIES):
        raise SystemExit(f"--modalities must be exactly: {' '.join(DEFAULT_MODALITIES)}")
    args.fuse_mode = "attn"
    args.no_modality_emb = False

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    train_dataset = BraTS2DPngDataset(args.train_2d_root, modalities=args.modalities, scale=int(args.scale))
    val_dataset = BraTS2DPngDataset(args.val_2d_root, modalities=args.modalities, scale=int(args.scale))

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

    model = FlexibleFusionNet(
        feat_channels=int(args.feat_channels),
        fuse_mode="attn",
        use_modality_emb=True,
        num_modalities=4,
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

    mod_ids_full = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
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

            for batch_idx, (x4, _case_ids, _zs) in enumerate(train_loader, start=1):
                global_step += 1
                x4 = x4.to(device)
                sel = _sample_train_subset(total_modalities=4, device=device)
                xk = x4.index_select(dim=1, index=sel)
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
                }
            )
            f_train.flush()

            val_logs = run_validation(
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

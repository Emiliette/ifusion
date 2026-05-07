import argparse
import csv
import os
import time
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from util.brats_bbox import BraTSFusedBBoxDataset, collate_bbox, read_bbox_manifest_csv, split_manifest_by_case, bbox_iou_xyxy_norm
from util.logger import get_logger
from util.unet_bbox import UNetBBox, bbox_loss


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Train U-Net bbox detector on fused BraTS slices (bbox-only, no segmentation)")
    p.add_argument("--manifest_csv", type=str, required=True, help="Training manifest CSV (e.g. bbox_manifest_train.csv).")
    p.add_argument("--fused_root", type=str, required=True, help="Training fused PNG root (folder containing <case>_z###.png).")
    p.add_argument("--val_manifest_csv", type=str, default="", help="If set, uses this CSV as the validation set (no internal split).")
    p.add_argument("--val_fused_root", type=str, default="", help="If set, uses this fused root for validation (defaults to --fused_root).")
    p.add_argument("--out_dir", type=str, default="./checkpoints_bbox_brats/")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--val_frac", type=float, default=0.2)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1, help="Save every N epochs")

    p.add_argument("--early_stop_patience", type=int, default=10, help="Patience in epochs. 0 disables.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--early_stop_warmup", type=int, default=5, help="Ignore early stop for first N epochs.")
    p.add_argument("--lr_reduce_patience", type=int, default=3, help="Patience in epochs. 0 disables.")
    p.add_argument("--lr_reduce_factor", type=float, default=0.5)
    p.add_argument("--lr_reduce_min_lr", type=float, default=1e-6)

    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--w_obj", type=float, default=1.0)
    p.add_argument("--w_box", type=float, default=5.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args(argv)


def save_checkpoint(path: str, *, model, optimizer, epoch: int, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate(model: UNetBBox, loader: DataLoader, device: torch.device):
    model.eval()
    tot = 0
    tot_pos = 0
    obj_correct = 0
    iou_sum = 0.0
    for x, has_t, bb, _case_ids, _zs in loader:
        x = x.to(device)
        has_t = has_t.to(device)
        bb = bb.to(device)
        obj_logit, bb_pred = model(x)
        obj_pred = (torch.sigmoid(obj_logit) > 0.5).float()
        obj_correct += int((obj_pred == (has_t > 0.5).float()).sum().item())
        tot += int(has_t.shape[0])

        pos = has_t > 0.5
        if pos.any():
            iou = bbox_iou_xyxy_norm(bb_pred[pos], bb[pos])
            iou_sum += float(iou.sum().item())
            tot_pos += int(pos.sum().item())

    acc = float(obj_correct) / float(max(1, tot))
    miou = float(iou_sum) / float(max(1, tot_pos))
    return {"acc": acc, "miou": miou, "n": tot, "n_pos": tot_pos}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    torch.manual_seed(int(args.seed))

    train_rows = read_bbox_manifest_csv(args.manifest_csv)
    if not train_rows:
        raise SystemExit(f"Empty train manifest: {args.manifest_csv}")

    use_external_val = bool(str(args.val_manifest_csv).strip())
    if use_external_val:
        val_rows = read_bbox_manifest_csv(args.val_manifest_csv)
        if not val_rows:
            raise SystemExit(f"Empty val manifest: {args.val_manifest_csv}")
        val_fused_root = args.val_fused_root.strip() or args.fused_root
        train_ds = BraTSFusedBBoxDataset(train_rows, fused_root=args.fused_root, indices=None, augment=bool(args.augment), seed=int(args.seed))
        val_ds = BraTSFusedBBoxDataset(val_rows, fused_root=val_fused_root, indices=None, augment=False, seed=int(args.seed))
        logger.info(f"Dataset: train={len(train_ds)} (external) val={len(val_ds)} (external)")
    else:
        train_idx, val_idx = split_manifest_by_case(train_rows, val_frac=float(args.val_frac), seed=int(args.seed))
        train_ds = BraTSFusedBBoxDataset(
            train_rows,
            fused_root=args.fused_root,
            indices=train_idx,
            augment=bool(args.augment),
            seed=int(args.seed),
        )
        val_ds = BraTSFusedBBoxDataset(train_rows, fused_root=args.fused_root, indices=val_idx, augment=False, seed=int(args.seed))
        logger.info(f"Dataset: train={len(train_ds)} val={len(val_ds)} (cases-split, val_frac={float(args.val_frac):.2f})")

    device = torch.device(args.device)
    model = UNetBBox(in_channels=1, base_channels=int(args.base_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    scheduler = None
    if int(args.lr_reduce_patience) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(args.lr_reduce_factor),
            patience=int(args.lr_reduce_patience),
            min_lr=float(args.lr_reduce_min_lr),
        )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_bbox,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_bbox,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "train_log.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["time", "epoch", "iter", "loss", "val_acc", "val_miou", "lr"],
        )
        if write_header:
            writer.writeheader()

        t0 = time.time()
        it_global = 0
        best_val_miou = float("-inf")
        best_epoch = start_epoch
        bad_epochs = 0
        for epoch in range(start_epoch, int(args.epochs)):
            model.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            for x, has_t, bb, _case_ids, _zs in train_loader:
                x = x.to(device)
                has_t = has_t.to(device)
                bb = bb.to(device)

                obj_logit, bb_pred = model(x)
                loss = bbox_loss(
                    obj_logit=obj_logit,
                    bbox_pred=bb_pred,
                    has_tumor=has_t,
                    bbox_true=bb,
                    w_obj=float(args.w_obj),
                    w_box=float(args.w_box),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.detach().cpu())
                epoch_steps += 1

                if it_global % int(args.log_every) == 0:
                    lr = float(optimizer.param_groups[0]["lr"])
                    logger.info(f"epoch {epoch} iter {it_global} | loss {float(loss.detach().cpu()):.4f} | lr {lr:.2e}")
                it_global += 1

            # Epoch-end validation + logging.
            lr = float(optimizer.param_groups[0]["lr"])
            val = evaluate(model, val_loader, device)
            avg_loss = float(epoch_loss_sum) / float(max(1, epoch_steps))
            row = {
                "time": float(time.time() - t0),
                "epoch": int(epoch),
                "iter": int(it_global),
                "loss": avg_loss,
                "val_acc": float(val["acc"]),
                "val_miou": float(val["miou"]),
                "lr": lr,
            }
            writer.writerow(row)
            f_csv.flush()
            logger.info(
                f"epoch {epoch} done | loss {row['loss']:.4f} | val acc {row['val_acc']:.3f} | "
                f"val mIoU {row['val_miou']:.3f} | lr {lr:.2e}"
            )

            if scheduler is not None:
                scheduler.step(float(val["miou"]))

            improved = float(val["miou"]) > float(best_val_miou) + float(args.early_stop_min_delta)
            if improved:
                best_val_miou = float(val["miou"])
                best_epoch = epoch
                bad_epochs = 0
                save_checkpoint(os.path.join(args.out_dir, "best.pth"), model=model, optimizer=optimizer, epoch=epoch + 1, args=args)
                logger.info(f"New best val mIoU: {best_val_miou:.3f} (saved best.pth)")
            else:
                bad_epochs += 1

            if (epoch + 1) % int(args.save_every) == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch+1}.pth")
                save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=epoch + 1, args=args)
                save_checkpoint(os.path.join(args.out_dir, "latest.pth"), model=model, optimizer=optimizer, epoch=epoch + 1, args=args)
                logger.info(f"Saved: {ckpt_path}")

            # Early stopping decision at epoch granularity.
            if int(args.early_stop_patience) > 0 and (epoch + 1) > int(args.early_stop_warmup):
                if bad_epochs >= int(args.early_stop_patience):
                    logger.info(
                        f"Early stop at epoch {epoch+1}: no val mIoU improvement for {bad_epochs} epochs "
                        f"(best={best_val_miou:.3f} at epoch {best_epoch+1})."
                    )
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

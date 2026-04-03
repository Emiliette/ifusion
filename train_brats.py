import argparse
import csv
import os
import time
from dataclasses import asdict
from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from util.brats_dataset import BraTSSliceDataset
from util.fusion_losses import FusionLossWeights, fusion_loss_unsupervised
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Unsupervised BraTS fusion training (flexible 2-4 modalities)")
    p.add_argument("--brats_root", type=str, default=os.path.join("Dataset-BraTS", "BraTS2024-BraTS-GLI-TrainingData"))
    p.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"])
    p.add_argument("--slice_mode", type=str, default="nonzero", choices=["all", "nonzero", "seg_nonzero"])
    p.add_argument("--min_nonzero_frac", type=float, default=0.01)
    p.add_argument("--p_low", type=float, default=0.5)
    p.add_argument("--p_high", type=float, default=99.5)
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--limit_cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--epochs", type=float, default=0.0, help="Optional. If set > 0, train for approx this many epochs (over the dataset length). Overrides --steps.")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--out_dir", type=str, default="./checkpoints_brats/")
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--feat_channels", type=int, default=32)
    p.add_argument("--min_inputs", type=int, default=2)
    p.add_argument("--max_inputs", type=int, default=4)

    # Metric objective weights
    p.add_argument("--w_en", type=float, default=1.0)
    p.add_argument("--w_mi", type=float, default=1.0)
    p.add_argument("--w_psnr", type=float, default=1.0)
    p.add_argument("--w_ssim", type=float, default=1.0)
    p.add_argument("--w_sd", type=float, default=1.0)
    p.add_argument("--w_ag", type=float, default=1.0)

    # MI/EN histogram params (tradeoff speed/quality)
    p.add_argument("--mi_bins", type=int, default=32)
    p.add_argument("--mi_sigma", type=float, default=0.04)
    p.add_argument("--en_bins", type=int, default=64)
    p.add_argument("--en_sigma", type=float, default=0.02)

    return p.parse_args(argv)


def _collate(batch):
    xs, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)  # (N,4,1,H,W)
    return x, list(case_ids), list(zs)


def _pick_subset_indices(min_k: int, max_k: int, device: torch.device) -> torch.Tensor:
    k = int(torch.randint(low=min_k, high=max_k + 1, size=(1,)).item())
    perm = torch.randperm(4)[:k]
    return perm.to(device)


def save_checkpoint(path: str, *, model, optimizer, step: int, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()

    if args.min_inputs < 2 or args.max_inputs > 4 or args.min_inputs > args.max_inputs:
        raise SystemExit("--min_inputs/--max_inputs must satisfy 2 <= min <= max <= 4")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    dataset = BraTSSliceDataset(
        args.brats_root,
        axis=args.axis,
        slice_mode=args.slice_mode,
        min_nonzero_frac=float(args.min_nonzero_frac),
        p_low=float(args.p_low),
        p_high=float(args.p_high),
        scale=int(args.scale),
        limit_cases=int(args.limit_cases),
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=_collate,
    )
    it = iter(loader)

    steps_per_epoch = max(1, len(loader))
    if args.epochs and float(args.epochs) > 0:
        args.steps = int(round(float(args.epochs) * steps_per_epoch))
    logger.info(f"Steps per epoch (approx): {steps_per_epoch} | Training steps: {int(args.steps)}")

    model = FlexibleFusionNet(feat_channels=int(args.feat_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        logger.info(f"Resumed from {args.resume} at step {start_step}")

    weights = FusionLossWeights(
        w_en=float(args.w_en),
        w_mi=float(args.w_mi),
        w_psnr=float(args.w_psnr),
        w_ssim=float(args.w_ssim),
        w_sd=float(args.w_sd),
        w_ag=float(args.w_ag),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "train_log.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "time",
                "step",
                "loss",
                "en",
                "mi",
                "ssim",
                "mse",
                "psnr",
                "sd",
                "ag",
                "k",
                "lr",
            ],
        )
        if write_header:
            writer.writeheader()

        t0 = time.time()
        for step in range(start_step, int(args.steps)):
            try:
                x4, _case_ids, _zs = next(it)
            except StopIteration:
                it = iter(loader)
                x4, _case_ids, _zs = next(it)

            x4 = x4.to(device)  # (N,4,1,H,W) in [0,1]
            sel = _pick_subset_indices(int(args.min_inputs), int(args.max_inputs), device=device)
            xk = x4.index_select(dim=1, index=sel)  # (N,K,1,H,W)

            model.train()
            fused = model(xk)  # (N,1,H,W) in [0,1]
            loss, logs = fusion_loss_unsupervised(
                fused,
                xk,
                weights=weights,
                mi_bins=int(args.mi_bins),
                mi_sigma=float(args.mi_sigma),
                en_bins=int(args.en_bins),
                en_sigma=float(args.en_sigma),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % int(args.log_every) == 0:
                lr = float(optimizer.param_groups[0]["lr"])
                logs_row = {
                    "time": float(time.time() - t0),
                    "step": int(step),
                    **{k: logs[k] for k in ["loss", "en", "mi", "ssim", "mse", "psnr", "sd", "ag", "k"]},
                    "lr": lr,
                }
                writer.writerow(logs_row)
                f_csv.flush()
                logger.info(
                    f"step {step} | loss {logs['loss']:.4f} | EN {logs['en']:.3f} | MI {logs['mi']:.3f} | "
                    f"SSIM {logs['ssim']:.4f} | PSNR {logs['psnr']:.2f} | SD {logs['sd']:.3f} | AG {logs['ag']:.3f} | K {int(logs['k'])}"
                )

            if (step + 1) % int(args.save_every) == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{step+1}.pth")
                save_checkpoint(ckpt_path, model=model, optimizer=optimizer, step=step + 1, args=args)
                save_checkpoint(os.path.join(args.out_dir, "latest.pth"), model=model, optimizer=optimizer, step=step + 1, args=args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

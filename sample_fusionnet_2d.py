import argparse
import os
import time
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.brats_2d_dataset import BraTS2DPngDataset
from util.fusion_model import FlexibleFusionNet
from util.logger import get_logger


def save_uint8_png(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Fuse BraTS 2D PNG slices using a trained FlexibleFusionNet")
    p.add_argument("--root_2d", type=str, required=True, help="Root with modality subfolders (t1n/t1c/t2w/t2f).")
    p.add_argument("--modalities", type=str, nargs="+", default=["t1n", "t1c", "t2w", "t2f"], help="2..4 modalities.")
    p.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints_brats", "latest.pth"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--limit_slices", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="./result_fusionnet_2d/")
    p.add_argument("--feat_channels", type=int, default=32, help="Must match training (or inferred from checkpoint args if present).")
    return p.parse_args(argv)


def _collate(batch):
    xs, case_ids, zs = zip(*batch)
    x = torch.stack(xs, dim=0)  # (N,M,1,H,W)
    return x, list(case_ids), list(zs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    modalities = [m.strip().lower() for m in args.modalities]
    if len(modalities) < 2 or len(modalities) > 4:
        raise SystemExit("--modalities must have 2..4 items.")
    mod_to_id = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
    try:
        mod_ids_full = torch.tensor([mod_to_id[m] for m in modalities], dtype=torch.long, device=device)
    except KeyError as e:
        raise SystemExit(f"Unknown modality in --modalities: {e}. Allowed: t1n t1c t2w t2f") from e

    ds = BraTS2DPngDataset(args.root_2d, modalities=modalities, scale=int(args.scale), limit_slices=int(args.limit_slices))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=_collate,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    feat_channels = int(ckpt_args.get("feat_channels", args.feat_channels))
    fuse_mode = str(ckpt_args.get("fuse_mode", "attn"))
    use_modality_emb = not bool(ckpt_args.get("no_modality_emb", False))

    model = FlexibleFusionNet(
        feat_channels=feat_channels,
        fuse_mode=fuse_mode,
        use_modality_emb=use_modality_emb,
        num_modalities=4,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    recon_dir = os.path.join(args.save_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)

    t0 = time.time()
    wrote = 0
    with torch.no_grad():
        for x, case_ids, zs in tqdm(loader, desc="Slices"):
            x = x.to(device)
            fused01 = model(x, mod_ids=mod_ids_full).detach().cpu().squeeze(1).numpy()  # (N,H,W)
            fused_u8 = np.clip(fused01 * 255.0, 0, 255).astype(np.uint8)
            for i in range(fused_u8.shape[0]):
                out_name = f"{case_ids[i]}_z{int(zs[i]):03d}.png"
                save_uint8_png(os.path.join(recon_dir, out_name), fused_u8[i])
                wrote += 1

    logger.info(f"Done. Wrote {wrote} fused PNGs to {recon_dir} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


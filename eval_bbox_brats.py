import argparse
import os
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from util.brats_bbox import (
    BraTSFusedBBoxDataset,
    collate_bbox,
    read_bbox_manifest_csv,
    split_manifest_by_case,
    xyxy_norm_to_px,
    bbox_iou_xyxy_norm,
)
from util.logger import get_logger
from util.unet_bbox import UNetBBox


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate U-Net bbox detector on fused BraTS slices")
    p.add_argument("--manifest_csv", type=str, default=os.path.join("results_brats", "bbox_manifest.csv"))
    p.add_argument("--fused_root", type=str, default=os.path.join("results_brats", "result", "recon"))
    p.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints_bbox_brats", "latest.pth"))
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--val_frac", type=float, default=0.2)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--base_channels", type=int, default=32)

    p.add_argument("--viz_dir", type=str, default="", help="If set, saves a small set of PNGs with GT/pred boxes.")
    p.add_argument("--viz_max", type=int, default=64)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()

    rows = read_bbox_manifest_csv(args.manifest_csv)
    train_idx, val_idx = split_manifest_by_case(rows, val_frac=float(args.val_frac), seed=int(args.seed))
    val_ds = BraTSFusedBBoxDataset(rows, fused_root=args.fused_root, indices=val_idx, augment=False, seed=int(args.seed))
    loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(args.device.startswith("cuda")),
        drop_last=False,
        collate_fn=collate_bbox,
    )

    device = torch.device(args.device)
    model = UNetBBox(in_channels=1, base_channels=int(args.base_channels)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tot = 0
    tot_pos = 0
    obj_correct = 0
    obj_tp = 0
    obj_fp = 0
    obj_fn = 0
    iou_sum = 0.0
    iou50 = 0

    do_viz = bool(args.viz_dir)
    viz_left = int(args.viz_max)
    if do_viz:
        os.makedirs(args.viz_dir, exist_ok=True)

    for x, has_t, bb, case_ids, zs in loader:
        x = x.to(device)
        has_t = has_t.to(device)
        bb = bb.to(device)
        obj_logit, bb_pred = model(x)
        prob = torch.sigmoid(obj_logit)
        obj_pred = (prob > 0.5).float()

        gt = (has_t > 0.5).float()
        obj_correct += int((obj_pred == gt).sum().item())
        obj_tp += int(((obj_pred > 0.5) & (gt > 0.5)).sum().item())
        obj_fp += int(((obj_pred > 0.5) & (gt <= 0.5)).sum().item())
        obj_fn += int(((obj_pred <= 0.5) & (gt > 0.5)).sum().item())

        pos = gt > 0.5
        if pos.any():
            iou = bbox_iou_xyxy_norm(bb_pred[pos], bb[pos])
            iou_sum += float(iou.sum().item())
            iou50 += int((iou >= 0.5).sum().item())
            tot_pos += int(pos.sum().item())
        tot += int(gt.shape[0])

        if do_viz and viz_left > 0:
            x_cpu = (x.detach().cpu().numpy() * 255.0).astype(np.uint8)  # (N,1,H,W)
            bb_pred_cpu = bb_pred.detach().cpu()
            bb_cpu = bb.detach().cpu()
            prob_cpu = prob.detach().cpu()
            gt_cpu = gt.detach().cpu()
            for i in range(x_cpu.shape[0]):
                if viz_left <= 0:
                    break
                img_u8 = x_cpu[i, 0]
                img = Image.fromarray(img_u8, mode="L").convert("RGB")
                draw = ImageDraw.Draw(img)
                w, h = img.size
                if gt_cpu[i].item() > 0.5:
                    x0, y0, x1, y1 = xyxy_norm_to_px(*bb_cpu[i].tolist(), w=w, h=h)
                    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)  # GT green
                if prob_cpu[i].item() > 0.5:
                    x0, y0, x1, y1 = xyxy_norm_to_px(*bb_pred_cpu[i].tolist(), w=w, h=h)
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)  # pred red
                out_name = f"{case_ids[i]}_z{int(zs[i]):03d}_p{prob_cpu[i].item():.2f}.png"
                img.save(os.path.join(args.viz_dir, out_name))
                viz_left -= 1

    acc = float(obj_correct) / float(max(1, tot))
    prec = float(obj_tp) / float(max(1, obj_tp + obj_fp))
    rec = float(obj_tp) / float(max(1, obj_tp + obj_fn))
    miou = float(iou_sum) / float(max(1, tot_pos))
    hit50 = float(iou50) / float(max(1, tot_pos))

    logger.info(f"Val set: n={tot} n_pos={tot_pos}")
    logger.info(f"Obj: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f}")
    logger.info(f"BBox: mIoU={miou:.3f} | IoU@0.5={hit50:.3f}")
    if do_viz:
        logger.info(f"Saved viz to: {args.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


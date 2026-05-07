import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from util.brats_bbox import (
    BraTSFusedBBoxDataset,
    collate_bbox,
    read_bbox_manifest_csv,
    xyxy_norm_to_px,
    bbox_iou_xyxy_norm,
)
from util.logger import get_logger
from util.unet_bbox import UNetBBox


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Case-level (patient-level) detection + bbox from a slice-level U-Net bbox detector")
    p.add_argument("--manifest_csv", type=str, required=True)
    p.add_argument("--fused_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5, help="Case is positive if max slice prob >= threshold.")
    p.add_argument(
        "--slice_threshold",
        type=float,
        default=0.5,
        help="Slices with prob >= this threshold contribute to the per-case (3D) union bbox.",
    )
    p.add_argument("--out_csv", type=str, default="./caselevel_bbox_preds.csv")
    p.add_argument("--viz_dir", type=str, default="", help="If set, saves per-case PNG with predicted bbox on best slice.")
    return p.parse_args(argv)


def _case_metrics(cases: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for v in cases.values():
        gt = int(v["gt"])
        pred = int(v["pred"])
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    acc = (tp + tn) / float(max(1, tp + tn + fp + fn))
    prec = tp / float(max(1, tp + fp))
    rec = tp / float(max(1, tp + fn))
    f1 = 2.0 * prec * rec / float(max(1e-12, prec + rec))
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()

    rows = read_bbox_manifest_csv(args.manifest_csv)
    if not rows:
        raise SystemExit(f"Empty manifest: {args.manifest_csv}")

    ds = BraTSFusedBBoxDataset(rows, fused_root=args.fused_root, indices=None, augment=False, seed=3407)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(str(args.device).startswith("cuda")),
        drop_last=False,
        collate_fn=collate_bbox,
    )

    device = torch.device(args.device)
    model = UNetBBox(in_channels=1, base_channels=int(args.base_channels)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Track per-case best slice prediction.
    cases: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "gt": 0.0,
            "prob": -1.0,
            "z": -1.0,
            "iou": float("nan"),
            "z_min": float("inf"),
            "z_max": float("-inf"),
            "n_sel": 0.0,
            "ux0": 1.0,
            "uy0": 1.0,
            "ux1": 0.0,
            "uy1": 0.0,
        }
    )
    best_bbox: Dict[str, Tuple[float, float, float, float]] = {}

    for x, has_t, bb, case_ids, zs in loader:
        x = x.to(device)
        has_t = has_t.to(device)
        bb = bb.to(device)
        obj_logit, bb_pred = model(x)
        prob = torch.sigmoid(obj_logit)  # (N,)

        for i in range(prob.shape[0]):
            cid = str(case_ids[i])
            z = int(zs[i])
            gt_slice = 1 if float(has_t[i].item()) > 0.5 else 0
            cases[cid]["gt"] = float(max(cases[cid]["gt"], float(gt_slice)))

            p = float(prob[i].item())
            if p >= float(args.slice_threshold):
                cases[cid]["z_min"] = float(min(float(cases[cid]["z_min"]), float(z)))
                cases[cid]["z_max"] = float(max(float(cases[cid]["z_max"]), float(z)))
                cases[cid]["n_sel"] = float(cases[cid]["n_sel"] + 1.0)
                x0p, y0p, x1p, y1p = (float(v) for v in bb_pred[i].detach().cpu().tolist())
                cases[cid]["ux0"] = float(min(float(cases[cid]["ux0"]), x0p))
                cases[cid]["uy0"] = float(min(float(cases[cid]["uy0"]), y0p))
                cases[cid]["ux1"] = float(max(float(cases[cid]["ux1"]), x1p))
                cases[cid]["uy1"] = float(max(float(cases[cid]["uy1"]), y1p))

            if p > float(cases[cid]["prob"]):
                cases[cid]["prob"] = p
                cases[cid]["z"] = float(z)
                best_bbox[cid] = tuple(float(v) for v in bb_pred[i].detach().cpu().tolist())
                # IoU only meaningful when GT slice is positive.
                if gt_slice == 1:
                    iou = bbox_iou_xyxy_norm(bb_pred[i].unsqueeze(0), bb[i].unsqueeze(0)).item()
                    cases[cid]["iou"] = float(iou)
                else:
                    cases[cid]["iou"] = float("nan")

    thr = float(args.threshold)
    for cid, v in cases.items():
        v["pred"] = 1.0 if float(v["prob"]) >= thr else 0.0
        if float(v["n_sel"]) <= 0:
            # Fallback: use best slice only.
            v["z_min"] = float(v["z"])
            v["z_max"] = float(v["z"])
            x0, y0, x1, y1 = best_bbox.get(cid, (0.0, 0.0, 0.0, 0.0))
            v["ux0"], v["uy0"], v["ux1"], v["uy1"] = float(x0), float(y0), float(x1), float(y1)

    metrics = _case_metrics(cases)
    logger.info(
        f"Cases: n={len(cases)} | acc={metrics['acc']:.3f} prec={metrics['prec']:.3f} "
        f"rec={metrics['rec']:.3f} f1={metrics['f1']:.3f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "gt_has_tumor",
                "pred_has_tumor",
                "pred_prob",
                "best_z",
                "x0",
                "y0",
                "x1",
                "y1",
                "z_min",
                "z_max",
                "ux0",
                "uy0",
                "ux1",
                "uy1",
                "n_slices_sel",
                "iou_best_slice",
            ],
        )
        w.writeheader()
        for cid in sorted(cases.keys()):
            v = cases[cid]
            x0, y0, x1, y1 = best_bbox.get(cid, (0.0, 0.0, 0.0, 0.0))
            w.writerow(
                {
                    "case_id": cid,
                    "gt_has_tumor": int(v["gt"]),
                    "pred_has_tumor": int(v["pred"]),
                    "pred_prob": float(v["prob"]),
                    "best_z": int(v["z"]),
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "z_min": int(v["z_min"]),
                    "z_max": int(v["z_max"]),
                    "ux0": float(v["ux0"]),
                    "uy0": float(v["uy0"]),
                    "ux1": float(v["ux1"]),
                    "uy1": float(v["uy1"]),
                    "n_slices_sel": int(v["n_sel"]),
                    "iou_best_slice": float(v["iou"]) if np.isfinite(float(v["iou"])) else "",
                }
            )

    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
        # Render only predicted-positive cases (or all if small).
        for cid in sorted(cases.keys()):
            v = cases[cid]
            if int(v["pred"]) == 0:
                continue
            z = int(v["z"])
            name = f"{cid}_z{z:03d}.png"
            img_path = os.path.join(args.fused_root, name)
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("L").convert("RGB")
            draw = ImageDraw.Draw(img)
            w_img, h_img = img.size
            x0, y0, x1, y1 = best_bbox.get(cid, (0.0, 0.0, 0.0, 0.0))
            px = xyxy_norm_to_px(x0, y0, x1, y1, w=w_img, h=h_img)
            draw.rectangle([px[0], px[1], px[2], px[3]], outline=(255, 0, 0), width=2)
            out_name = f"{cid}_z{z:03d}_p{float(v['prob']):.2f}.png"
            img.save(os.path.join(args.viz_dir, out_name))

        logger.info(f"Saved viz to: {args.viz_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

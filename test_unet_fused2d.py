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
from tqdm import tqdm

from util.logger import get_logger
from util.unet_detect import UNetDetector2D
from util.unet_seg import UNet2D


_SLICE_RE = re.compile(r"^(?P<case>.+)_z(?P<z>\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class SliceIndex:
    name: str
    case_id: str
    z: int


def _parse_slice_name(name: str) -> SliceIndex:
    m = _SLICE_RE.match(os.path.basename(name))
    if not m:
        stem = os.path.splitext(os.path.basename(name))[0]
        return SliceIndex(name=os.path.basename(name), case_id=stem, z=0)
    return SliceIndex(name=os.path.basename(name), case_id=m.group("case"), z=int(m.group("z")))


def _list_png_names(folder: str) -> List[str]:
    return sorted([n for n in os.listdir(folder) if n.lower().endswith(".png")])


def _remap_brats_labels(seg_u8: np.ndarray, *, remap_4_to_3: bool) -> np.ndarray:
    seg = seg_u8.astype(np.int64, copy=False)
    if remap_4_to_3:
        seg = np.where(seg == 4, 3, seg)
    return seg.astype(np.int64, copy=False)


class Fused2DInferDataset(Dataset):
    """
    Inference dataset for fused PNG slices.

    Expected:
      fused_dir/<case>_z###.png
      seg_dir/<case>_z###.png (optional)

    Returns:
      x: FloatTensor (1,H,W) in [0,1]
      y: LongTensor (H,W) if seg_dir is provided else None
      case_id: str
      z: int
      name: str
    """

    def __init__(
        self,
        fused_dir: str,
        *,
        seg_dir: str = "",
        scale: int = 1,
        limit_slices: int = 0,
        remap_4_to_3: bool = True,
    ) -> None:
        self.fused_dir = str(fused_dir)
        self.seg_dir = str(seg_dir)
        self.scale = int(scale)
        self.remap_4_to_3 = bool(remap_4_to_3)

        if not os.path.isdir(self.fused_dir):
            raise RuntimeError(f"Missing fused_dir: {self.fused_dir}")

        fused_names = _list_png_names(self.fused_dir)
        if not fused_names:
            raise RuntimeError(f"No PNGs found in fused_dir: {self.fused_dir}")

        if self.seg_dir:
            if not os.path.isdir(self.seg_dir):
                raise RuntimeError(f"Missing seg_dir: {self.seg_dir}")
            seg_names = set(_list_png_names(self.seg_dir))
            fused_names = [n for n in fused_names if n in seg_names]
            if not fused_names:
                raise RuntimeError(f"No aligned fused+seg PNGs found. fused_dir={self.fused_dir} seg_dir={self.seg_dir}")

        indices = [_parse_slice_name(n) for n in fused_names]
        indices.sort(key=lambda s: (s.case_id, s.z))
        if limit_slices and int(limit_slices) > 0:
            indices = indices[: int(limit_slices)]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], str, int, str]:
        it = self.indices[int(idx)]
        img_u8 = np.array(Image.open(os.path.join(self.fused_dir, it.name)).convert("L"), dtype=np.uint8)

        y = None
        if self.seg_dir:
            seg_u8 = np.array(Image.open(os.path.join(self.seg_dir, it.name)).convert("L"), dtype=np.uint8)
            h2 = min(img_u8.shape[0], seg_u8.shape[0])
            w2 = min(img_u8.shape[1], seg_u8.shape[1])
            if self.scale > 1:
                h2 = h2 - (h2 % self.scale)
                w2 = w2 - (w2 % self.scale)
            img_u8 = img_u8[:h2, :w2]
            seg_u8 = seg_u8[:h2, :w2]
            seg = _remap_brats_labels(seg_u8, remap_4_to_3=self.remap_4_to_3)
            y = torch.from_numpy(seg).long()
        else:
            h2, w2 = img_u8.shape
            if self.scale > 1:
                h2 = h2 - (h2 % self.scale)
                w2 = w2 - (w2 % self.scale)
            img_u8 = img_u8[:h2, :w2]

        img01 = (img_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)
        x = torch.from_numpy(img01).unsqueeze(0)  # (1,H,W)
        return x, y, it.case_id, int(it.z), it.name


def _collate(batch):
    xs, ys, case_ids, zs, names = zip(*batch)
    x = torch.stack(xs, dim=0)  # (N,1,H,W)
    has_y = all(t is not None for t in ys)
    y = torch.stack([t for t in ys], dim=0) if has_y else None  # (N,H,W) or None
    return x, y, list(case_ids), list(zs), list(names)


def _dice_accumulators(num_classes: int) -> Dict[str, np.ndarray]:
    c = int(num_classes)
    return {
        "inter": np.zeros((c,), dtype=np.float64),
        "pred_sum": np.zeros((c,), dtype=np.float64),
        "gt_sum": np.zeros((c,), dtype=np.float64),
    }


def _dice_update(acc: Dict[str, np.ndarray], pred: torch.Tensor, gt: torch.Tensor, *, num_classes: int) -> None:
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    for cls in range(int(num_classes)):
        p = (pred == cls)
        t = (gt == cls)
        acc["inter"][cls] += float(torch.sum(p & t).item())
        acc["pred_sum"][cls] += float(torch.sum(p).item())
        acc["gt_sum"][cls] += float(torch.sum(t).item())


def _dice_from_acc(acc: Dict[str, np.ndarray], *, eps: float = 1e-6) -> np.ndarray:
    inter = acc["inter"]
    denom = acc["pred_sum"] + acc["gt_sum"]
    return (2.0 * inter + eps) / (denom + eps)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Test (inference/eval) on fused BraTS 2D PNG slices after training")
    p.add_argument("--mode", type=str, default="seg", choices=["seg", "detect"], help="seg=mask prediction, detect=binary slice detect")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint from train_unet_fused2d.py or train_unet_detect_fused2d.py")
    p.add_argument("--fused_dir", type=str, required=True, help="Folder with fused PNGs: <case>_z###.png")
    p.add_argument("--seg_dir", type=str, default="", help="Optional GT seg PNGs (same names) to compute metrics")

    p.add_argument("--out_dir", type=str, default="./result_test_unet_fused2d/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--limit_slices", type=int, default=0)
    p.add_argument("--scale", type=int, default=30, help="Crop H/W to multiples of this value (defaults to ckpt args if present).")

    # seg model params
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--remap_4_to_3", action="store_true")
    p.add_argument("--no_remap_4_to_3", dest="remap_4_to_3", action="store_false")
    p.set_defaults(remap_4_to_3=True)

    # detect params
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args(argv)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_mask_png(path: str, mask_u8: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    Image.fromarray(mask_u8, mode="L").save(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    scale = int(ckpt_args.get("scale", args.scale))
    remap_4_to_3 = bool(ckpt_args.get("remap_4_to_3", args.remap_4_to_3))

    dataset = Fused2DInferDataset(
        args.fused_dir,
        seg_dir=str(args.seg_dir or ""),
        scale=scale,
        limit_slices=int(args.limit_slices),
        remap_4_to_3=remap_4_to_3,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=_collate,
    )
    logger.info(f"Slices: {len(dataset)} | scale={scale} | has_gt={bool(args.seg_dir)}")

    _ensure_dir(args.out_dir)
    pred_dir = os.path.join(args.out_dir, "pred")
    _ensure_dir(pred_dir)

    t0 = time.time()
    wrote = 0

    if args.mode == "seg":
        num_classes = int(ckpt_args.get("num_classes", args.num_classes))
        base_channels = int(ckpt_args.get("base_channels", args.base_channels))
        model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=base_channels).to(device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        model.eval()
        logger.info(f"Loaded seg checkpoint: {args.checkpoint} | num_classes={num_classes} base_channels={base_channels}")

        dice_acc = _dice_accumulators(num_classes) if args.seg_dir else None
        with torch.no_grad():
            for x, y, case_ids, zs, names in tqdm(loader, desc="Seg"):
                x = x.to(device)
                logits = model(x)  # (N,C,H,W)
                pred = torch.argmax(logits, dim=1).to(torch.uint8)  # (N,H,W)

                if y is not None and dice_acc is not None:
                    _dice_update(dice_acc, pred, y, num_classes=num_classes)

                pred_np = pred.detach().cpu().numpy()
                for i in range(pred_np.shape[0]):
                    out_name = names[i]
                    _save_mask_png(os.path.join(pred_dir, out_name), pred_np[i])
                    wrote += 1

        if dice_acc is not None:
            dice = _dice_from_acc(dice_acc)
            fg = dice[1:] if dice.shape[0] > 1 else dice
            fg_mean = float(np.mean(fg)) if fg.size else 0.0
            dice_str = " ".join([f"dice{c}={dice[c]:.4f}" for c in range(dice.shape[0])])
            logger.info(f"Dice: {dice_str} | dice_fg_mean={fg_mean:.4f}")

            out_csv = os.path.join(args.out_dir, "metrics_seg.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for c in range(dice.shape[0]):
                    w.writerow([f"dice_class_{c}", float(dice[c])])
                w.writerow(["dice_fg_mean", fg_mean])
            logger.info(f"Wrote: {out_csv}")

    else:
        base_channels = int(ckpt_args.get("base_channels", args.base_channels))
        dropout = float(ckpt_args.get("dropout", args.dropout))
        threshold = float(ckpt_args.get("threshold", args.threshold))
        model = UNetDetector2D(in_channels=1, base_channels=base_channels, dropout=dropout).to(device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        model.eval()
        logger.info(
            f"Loaded detect checkpoint: {args.checkpoint} | base_channels={base_channels} dropout={dropout:g} threshold={threshold:g}"
        )

        out_csv = os.path.join(args.out_dir, "pred_detect.csv")
        tp = tn = fp = fn = 0.0

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["name", "case_id", "z", "prob", "pred", "gt"],
            )
            w.writeheader()

            with torch.no_grad():
                for x, y, case_ids, zs, names in tqdm(loader, desc="Detect"):
                    x = x.to(device)
                    logits = model(x)  # (N,)
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    pred = (probs >= threshold).astype(np.int64)

                    gt = None
                    if y is not None:
                        # y: (N,H,W) class ids; slice is positive if any tumor pixel exists.
                        gt = (torch.sum(y > 0, dim=(1, 2)) > 0).to(torch.int64).numpy()
                        tp += float(np.sum((pred == 1) & (gt == 1)))
                        tn += float(np.sum((pred == 0) & (gt == 0)))
                        fp += float(np.sum((pred == 1) & (gt == 0)))
                        fn += float(np.sum((pred == 0) & (gt == 1)))

                    for i in range(len(names)):
                        w.writerow(
                            {
                                "name": names[i],
                                "case_id": case_ids[i],
                                "z": int(zs[i]),
                                "prob": float(probs[i]),
                                "pred": int(pred[i]),
                                "gt": int(gt[i]) if gt is not None else "",
                            }
                        )
                        wrote += 1

        logger.info(f"Wrote: {out_csv}")
        if args.seg_dir:
            total = tp + tn + fp + fn + 1e-12
            acc = (tp + tn) / total
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
            logger.info(f"Detect metrics: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
            out_m = os.path.join(args.out_dir, "metrics_detect.csv")
            with open(out_m, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                w.writerow(["acc", float(acc)])
                w.writerow(["prec", float(prec)])
                w.writerow(["rec", float(rec)])
                w.writerow(["f1", float(f1)])
            logger.info(f"Wrote: {out_m}")

    logger.info(f"Done. Wrote {wrote} outputs under {os.path.abspath(args.out_dir)} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


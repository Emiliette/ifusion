from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from util.brats_bbox import bbox_xyxy_from_mask


@dataclass(frozen=True)
class Sample:
    image_path: str
    mask_path: str
    rel_key: str
    label: int


def parse_args() -> argparse.Namespace:
    # Edit these five paths once, then run the script without passing CLI paths.
    default_train_fused_dir = "/storage/student5/ifusion/Intern_ImageFusion/result_fusionnet_2d_train/recon"
    default_train_mask_dir = "/storage/student5/ifusion/Intern_ImageFusion/scripts/Dataset/BraTS_2D_Training/seg"
    default_val_fused_dir = "/storage/student5/ifusion/Intern_ImageFusion/result_fusionnet_2d_val/recon"
    default_val_mask_dir = "/storage/student5/ifusion/Intern_ImageFusion/scripts/Dataset/BraTS_2D_Validation/seg"
    default_out_dir = "/storage/student5/ifusion/Intern_ImageFusion/checkpoints_densenet121_yolov8s"

    p = argparse.ArgumentParser(
        "Train DenseNet121 classifier and YOLOv8s detector from fused 2D images + segmentation masks."
    )
    p.add_argument("--train_fused_dir", type=str, default=default_train_fused_dir, help="Train fused PNG root.")
    p.add_argument("--train_mask_dir", type=str, default=default_train_mask_dir, help="Train segmentation-mask PNG root.")
    p.add_argument("--val_fused_dir", type=str, default=default_val_fused_dir, help="Validation fused PNG root.")
    p.add_argument("--val_mask_dir", type=str, default=default_val_mask_dir, help="Validation segmentation-mask PNG root.")
    p.add_argument("--out_dir", type=str, default=default_out_dir, help="Output root for checkpoints, labels, and logs.")

    p.add_argument("--img_size", type=int, default=224, help="DenseNet input size.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100, help="DenseNet training epochs.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--cls_early_stop_patience", type=int, default=10, help="DenseNet early-stop patience in epochs. 0 disables.")
    p.add_argument("--cls_early_stop_min_delta", type=float, default=0.0, help="Minimum val_f1 improvement to reset patience.")
    p.add_argument("--cls_lr_reduce_patience", type=int, default=3, help="DenseNet LR-reduce patience in epochs. 0 disables.")
    p.add_argument("--cls_lr_reduce_factor", type=float, default=0.5, help="Factor for DenseNet ReduceLROnPlateau.")
    p.add_argument("--cls_min_lr", type=float, default=1e-6, help="Minimum LR for DenseNet ReduceLROnPlateau.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet initialization for DenseNet121.")

    p.add_argument("--yolo_imgsz", type=int, default=640)
    p.add_argument("--yolo_epochs", type=int, default=100)
    p.add_argument("--yolo_batch", type=int, default=16)
    p.add_argument("--yolo_patience", type=int, default=20)
    p.add_argument("--yolo_device", type=str, default=None, help="YOLO device override, e.g. 0 or cpu.")

    p.add_argument("--prepare_only", action="store_true", help="Only build derived labels and dataset folders.")
    p.add_argument("--skip_cls", action="store_true", help="Skip DenseNet121 training.")
    p.add_argument("--skip_yolo", action="store_true", help="Skip YOLOv8x training.")
    p.add_argument("--resume", action="store_true", help="Resume DenseNet121 and YOLOv8s from latest checkpoints if available.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _norm_relpath(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def _scan_pngs(root: str) -> Dict[str, str]:
    root = os.path.abspath(root)
    out: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(".png"):
                continue
            full = os.path.join(dirpath, name)
            rel = _norm_relpath(full, root)
            out[rel] = full
    return out


def _scan_pngs_by_dir(root: str) -> Dict[str, List[str]]:
    root = os.path.abspath(root)
    out: Dict[str, List[str]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        pngs = [os.path.join(dirpath, name) for name in filenames if name.lower().endswith(".png")]
        if not pngs:
            continue
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        out[rel_dir] = sorted(pngs)
    return out


def _build_pair_index(fused_dir: str, mask_dir: str) -> List[Sample]:
    fused_files = _scan_pngs(fused_dir)
    mask_files = _scan_pngs(mask_dir)

    samples: List[Sample] = []
    common_rel = sorted(set(fused_files.keys()) & set(mask_files.keys()))
    for rel in common_rel:
        mask_path = mask_files[rel]
        label = 1 if _mask_has_tumor(mask_path) else 0
        samples.append(Sample(image_path=fused_files[rel], mask_path=mask_path, rel_key=rel, label=label))

    if samples:
        return samples

    # Fallback for flat folders with matching basenames only.
    fused_by_name = {os.path.basename(path): path for path in fused_files.values()}
    mask_by_name = {os.path.basename(path): path for path in mask_files.values()}
    common_name = sorted(set(fused_by_name.keys()) & set(mask_by_name.keys()))
    for name in common_name:
        label = 1 if _mask_has_tumor(mask_by_name[name]) else 0
        samples.append(
            Sample(
                image_path=fused_by_name[name],
                mask_path=mask_by_name[name],
                rel_key=name,
                label=label,
            )
        )

    if samples:
        return samples

    # Fallback for mirrored folder trees where each sample folder contains one fused PNG
    # and one mask PNG, but the filenames differ, e.g. fused.png vs seg.png.
    fused_dirs = _scan_pngs_by_dir(fused_dir)
    mask_dirs = _scan_pngs_by_dir(mask_dir)
    for rel_dir in sorted(set(fused_dirs.keys()) & set(mask_dirs.keys())):
        fused_group = fused_dirs[rel_dir]
        mask_group = mask_dirs[rel_dir]
        if len(fused_group) != 1 or len(mask_group) != 1:
            continue
        image_path = fused_group[0]
        mask_path = mask_group[0]
        label = 1 if _mask_has_tumor(mask_path) else 0
        rel_name = f"{rel_dir}/{Path(image_path).stem}.png" if rel_dir != "." else Path(image_path).name
        samples.append(Sample(image_path=image_path, mask_path=mask_path, rel_key=rel_name, label=label))

    return samples


def _mask_has_tumor(mask_path: str) -> bool:
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return bool(np.any(mask > 0))


def _ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _bbox_to_yolo(mask_path: str) -> Tuple[int, str]:
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    h, w = mask.shape
    bbox = bbox_xyxy_from_mask(mask)
    if bbox is None:
        return 0, ""
    x0, y0, x1, y1 = bbox
    xc = ((x0 + x1) * 0.5) / float(w)
    yc = ((y0 + y1) * 0.5) / float(h)
    bw = (x1 - x0) / float(w)
    bh = (y1 - y0) / float(h)
    return 1, f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def prepare_yolo_split(samples: Sequence[Sample], split: str, out_root: str) -> Dict[str, int]:
    images_dir = os.path.join(out_root, "images", split)
    labels_dir = os.path.join(out_root, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    pos = 0
    neg = 0
    for sample in samples:
        name = Path(sample.rel_key).stem + ".png"
        image_out = os.path.join(images_dir, name)
        label_out = os.path.join(labels_dir, Path(sample.rel_key).stem + ".txt")
        _link_or_copy(sample.image_path, image_out)
        has_tumor, line = _bbox_to_yolo(sample.mask_path)
        with open(label_out, "w", encoding="utf-8") as f:
            f.write(line)
        if has_tumor:
            pos += 1
        else:
            neg += 1
    return {"n": len(samples), "pos": pos, "neg": neg}


def write_yolo_yaml(out_root: str) -> str:
    yaml_path = os.path.join(out_root, "dataset.yaml")
    lines = [
        f"path: {Path(out_root).resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: tumor",
        "",
    ]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return yaml_path


def write_prepare_summary(path: str, train_stats: Dict[str, int], val_stats: Dict[str, int]) -> None:
    payload = {
        "train": train_stats,
        "val": val_stats,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class FusedSliceClassificationDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], img_size: int, train: bool) -> None:
        self.samples = list(samples)
        self.img_size = int(img_size)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.samples)

    def _transform(self, img: Image.Image) -> torch.Tensor:
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        x = (x - 0.5) / 0.5
        return x

    def __getitem__(self, idx: int):
        sample = self.samples[int(idx)]
        img = Image.open(sample.image_path).convert("L")
        x = self._transform(img)
        y = torch.tensor(float(sample.label), dtype=torch.float32)
        return x, y, sample.rel_key


def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    target = targets.float()

    tp = float(((pred == 1) & (target == 1)).sum().item())
    tn = float(((pred == 0) & (target == 0)).sum().item())
    fp = float(((pred == 1) & (target == 0)).sum().item())
    fn = float(((pred == 0) & (target == 1)).sum().item())
    total = max(1.0, tp + tn + fp + fn)

    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / total
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_densenet121(pretrained: bool) -> nn.Module:
    model = models.densenet121(pretrained=bool(pretrained))
    in_features = int(model.classifier.in_features)
    model.classifier = nn.Linear(in_features, 1)
    return model


@torch.no_grad()
def evaluate_cls(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    logits_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    loss_sum = 0.0
    n = 0
    for x, y, _keys in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        batch = int(y.shape[0])
        loss_sum += float(loss.item()) * batch
        n += batch
        logits_all.append(logits.detach().cpu())
        targets_all.append(y.detach().cpu())
    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    metrics = binary_metrics_from_logits(logits_cat, targets_cat)
    metrics["loss"] = loss_sum / max(1, n)
    return metrics


def train_densenet121(
    *,
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    out_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    lr_reduce_patience: int,
    lr_reduce_factor: float,
    min_lr: float,
    device: str,
    pretrained: bool,
    resume: bool,
) -> str:
    device_obj = torch.device(device)
    train_ds = FusedSliceClassificationDataset(train_samples, img_size=img_size, train=True)
    val_ds = FusedSliceClassificationDataset(val_samples, img_size=img_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=(device_obj.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device_obj.type == "cuda"),
        drop_last=False,
    )

    pos_count = sum(int(s.label) for s in train_samples)
    neg_count = max(0, len(train_samples) - pos_count)
    if pos_count > 0 and neg_count > 0:
        pos_weight = torch.tensor([float(neg_count) / float(pos_count)], device=device_obj)
    else:
        pos_weight = torch.tensor([1.0], device=device_obj)

    model = make_densenet121(pretrained=pretrained).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = None
    if int(lr_reduce_patience) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(lr_reduce_factor),
            patience=int(lr_reduce_patience),
            min_lr=float(min_lr),
        )

    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_densenet121.pth")
    latest_path = os.path.join(out_dir, "latest_densenet121.pth")
    csv_path = os.path.join(out_dir, "densenet121_log.csv")

    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 0

    if resume and os.path.isfile(latest_path):
        checkpoint = torch.load(latest_path, map_location=device_obj)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_f1 = float(checkpoint.get("best_f1", best_f1))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        bad_epochs = int(checkpoint.get("bad_epochs", bad_epochs))
        print(f"[DenseNet121] resume from {latest_path} at epoch {start_epoch}.")
    elif resume:
        print(f"[DenseNet121] resume requested but checkpoint not found: {latest_path}")

    if start_epoch >= int(epochs):
        print(f"[DenseNet121] skip training because latest epoch {start_epoch} >= target epochs {epochs}.")
        return best_path if os.path.isfile(best_path) else latest_path

    csv_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    csv_mode = "a" if resume and csv_exists else "w"
    with open(csv_path, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "lr"],
        )
        if csv_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch + 1, int(epochs) + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            for x, y, _keys in train_loader:
                x = x.to(device_obj)
                y = y.to(device_obj)
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch = int(y.shape[0])
                loss_sum += float(loss.item()) * batch
                seen += batch

            train_loss = loss_sum / max(1, seen)
            val_metrics = evaluate_cls(model, val_loader, device_obj, criterion)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            writer.writerow(row)
            f.flush()

            if scheduler is not None:
                scheduler.step(float(val_metrics["f1"]))

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "img_size": int(img_size),
                    "best_f1": best_f1,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                },
                latest_path,
            )
            improved = float(val_metrics["f1"]) > float(best_f1) + float(early_stop_min_delta)
            if improved:
                best_f1 = float(val_metrics["f1"])
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "img_size": int(img_size),
                        "best_f1": best_f1,
                        "best_epoch": best_epoch,
                        "bad_epochs": bad_epochs,
                    },
                    latest_path,
                )
                shutil.copy2(latest_path, best_path)
            else:
                bad_epochs += 1
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "img_size": int(img_size),
                        "best_f1": best_f1,
                        "best_epoch": best_epoch,
                        "bad_epochs": bad_epochs,
                    },
                    latest_path,
                )

            print(
                f"[DenseNet121] epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f}"
            )

            if int(early_stop_patience) > 0 and bad_epochs >= int(early_stop_patience):
                print(
                    f"[DenseNet121] early stop at epoch {epoch}: "
                    f"no val_f1 improvement for {bad_epochs} epochs "
                    f"(best_f1={best_f1:.4f} at epoch {best_epoch})."
                )
                break

    return best_path


def train_yolov8x(
    *,
    data_yaml: str,
    out_dir: str,
    epochs: int,
    batch: int,
    imgsz: int,
    patience: int,
    device: str | None,
    resume: bool,
) -> str:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Install it first, then rerun with --skip_cls or full training as needed."
        ) from exc

    weights_dir = os.path.join(out_dir, "yolov8x", "weights")
    last_path = os.path.join(weights_dir, "last.pt")
    if resume and os.path.isfile(last_path):
        model = YOLO(last_path)
        train_kwargs = {"resume": True}
        print(f"[YOLOv8x] resume from {last_path}")
    else:
        if resume:
            print(f"[YOLOv8x] resume requested but checkpoint not found: {last_path}")
        model = YOLO("yolov8x.pt")
        train_kwargs = {
            "data": data_yaml,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "patience": int(patience),
            "project": out_dir,
            "name": "yolov8x",
            "exist_ok": True,
        }
    if device is not None:
        train_kwargs["device"] = device
    model.train(**train_kwargs)
    return os.path.join(out_dir, "yolov8x", "weights", "best.pt")


def describe_required_layout() -> str:
    return (
        "Expected input: fused PNGs and mask PNGs aligned either by identical relative path under train/val roots, "
        "or by identical basename in flat folders. Mask pixels > 0 are treated as tumor."
    )


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))

    train_samples = _build_pair_index(args.train_fused_dir, args.train_mask_dir)
    val_samples = _build_pair_index(args.val_fused_dir, args.val_mask_dir)
    if not train_samples:
        raise SystemExit("No aligned train fused/mask PNG pairs found.")
    if not val_samples:
        raise SystemExit("No aligned val fused/mask PNG pairs found.")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(describe_required_layout())
    print(f"train pairs={len(train_samples)} val pairs={len(val_samples)}")

    prepared_root = os.path.join(out_dir, "prepared_yolo")
    _ensure_clean_dir(prepared_root)
    train_stats = prepare_yolo_split(train_samples, "train", prepared_root)
    val_stats = prepare_yolo_split(val_samples, "val", prepared_root)
    data_yaml = write_yolo_yaml(prepared_root)
    write_prepare_summary(os.path.join(out_dir, "prepare_summary.json"), train_stats, val_stats)

    print(
        f"[Prepare] train: n={train_stats['n']} pos={train_stats['pos']} neg={train_stats['neg']} | "
        f"val: n={val_stats['n']} pos={val_stats['pos']} neg={val_stats['neg']}"
    )
    print(f"[Prepare] YOLO dataset YAML: {data_yaml}")

    if args.prepare_only:
        return 0

    if not args.skip_cls:
        cls_out = os.path.join(out_dir, "classification")
        best_cls = train_densenet121(
            train_samples=train_samples,
            val_samples=val_samples,
            out_dir=cls_out,
            img_size=int(args.img_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            early_stop_patience=int(args.cls_early_stop_patience),
            early_stop_min_delta=float(args.cls_early_stop_min_delta),
            lr_reduce_patience=int(args.cls_lr_reduce_patience),
            lr_reduce_factor=float(args.cls_lr_reduce_factor),
            min_lr=float(args.cls_min_lr),
            device=str(args.device),
            pretrained=not bool(args.no_pretrained),
            resume=bool(args.resume),
        )
        print(f"[DenseNet121] best checkpoint: {best_cls}")

    if not args.skip_yolo:
        yolo_project = os.path.join(out_dir, "detection")
        best_yolo = train_yolov8x(
            data_yaml=data_yaml,
            out_dir=yolo_project,
            epochs=int(args.yolo_epochs),
            batch=int(args.yolo_batch),
            imgsz=int(args.yolo_imgsz),
            patience=int(args.yolo_patience),
            device=args.yolo_device,
            resume=bool(args.resume),
        )
        print(f"[YOLOv8x] best checkpoint: {best_yolo}")

    return 0


if __name__ == "__main__":
    start = time.time()
    code = main()
    print(f"Done in {(time.time() - start):.1f}s")
    raise SystemExit(code)

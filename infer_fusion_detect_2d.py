import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import models

from util.fusion_model import FlexibleFusionNet


DEFAULT_CONFIG = {
    "inputs": [
        r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\Dataset-BraTS-2D\BraTS-GLI-02856-101_z100\t1n_upscayl.jpg",
        r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\Dataset-BraTS-2D\BraTS-GLI-02856-101_z100\t1c_upscayl.jpg",
        r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\Dataset-BraTS-2D\BraTS-GLI-02856-101_z100\t2w_upscayl.jpg",
        r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\Dataset-BraTS-2D\BraTS-GLI-02856-101_z100\t2f_upscayl.jpg",
    ],
    "modalities": ["t1n", "t1c", "t2w", "t2f"],
    "fusion_checkpoint": r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\best_fusion.pt",
    "cls_checkpoint": r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\best_densenet121.pth",
    "det_checkpoint": r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\best.pt",
    "out_dir": r"D:\All\USTH\B3\Thesis\FlexiD-Fuse code\FlexiD-Fuse-Kaggle2\kaggle\working\FlexiD-Fuse\result_fusion_detect",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "scale": 30,
    "threshold": 0.5,
    "feat_channels": 32,
    "cls_img_size": 224,
    "yolo_imgsz": 640,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Fuse multi-modal 2D images, save an intermediate fused image, then run detection."
    )
    p.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=list(DEFAULT_CONFIG["inputs"]),
        help="Input grayscale image paths, one per modality.",
    )
    p.add_argument(
        "--modalities",
        type=str,
        nargs="*",
        default=list(DEFAULT_CONFIG["modalities"]),
        help="Modality names aligned with --inputs. Allowed: t1n t1c t2w t2f",
    )
    p.add_argument("--fusion_checkpoint", type=str, default=str(DEFAULT_CONFIG["fusion_checkpoint"]))
    p.add_argument("--det_checkpoint", type=str, default=str(DEFAULT_CONFIG["det_checkpoint"]))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_CONFIG["out_dir"]))
    p.add_argument("--device", type=str, default=str(DEFAULT_CONFIG["device"]))
    p.add_argument("--scale", type=int, default=int(DEFAULT_CONFIG["scale"]), help="Crop H/W down to common multiples of this value.")
    p.add_argument("--threshold", type=float, default=float(DEFAULT_CONFIG["threshold"]), help="Detection threshold.")
    p.add_argument("--cls_checkpoint", type=str, default=str(DEFAULT_CONFIG["cls_checkpoint"]), help="Required DenseNet121 checkpoint from train_densenet121_yolov8s.py")
    p.add_argument("--feat_channels", type=int, default=int(DEFAULT_CONFIG["feat_channels"]))
    p.add_argument("--cls_img_size", type=int, default=int(DEFAULT_CONFIG["cls_img_size"]))
    p.add_argument("--yolo_imgsz", type=int, default=int(DEFAULT_CONFIG["yolo_imgsz"]))
    return p.parse_args(argv)


def _load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _resolve_common_shape(images: List[np.ndarray], scale: int) -> Tuple[int, int]:
    h = min(img.shape[0] for img in images)
    w = min(img.shape[1] for img in images)
    if scale > 1:
        h = (h // scale) * scale
        w = (w // scale) * scale
    if h <= 0 or w <= 0:
        raise ValueError("Images are too small after cropping to the requested scale.")
    return h, w


def _center_crop(img: np.ndarray, h: int, w: int) -> np.ndarray:
    top = max(0, (img.shape[0] - h) // 2)
    left = max(0, (img.shape[1] - w) // 2)
    return np.ascontiguousarray(img[top : top + h, left : left + w])


def _save_gray(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(path)


def _load_fusion_model(args: argparse.Namespace, device: torch.device) -> FlexibleFusionNet:
    ckpt = torch.load(args.fusion_checkpoint, map_location="cpu")
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
    return model


def _make_densenet121() -> torch.nn.Module:
    model = models.densenet121(pretrained=False)
    in_features = int(model.classifier.in_features)
    model.classifier = torch.nn.Linear(in_features, 1)
    return model


def _load_cls_model(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, int]:
    ckpt = torch.load(args.cls_checkpoint, map_location="cpu")
    model = _make_densenet121().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    img_size = int(ckpt.get("img_size", args.cls_img_size))
    return model, img_size


def _build_fusion_input(images: List[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(2)  # (1,K,1,H,W)
    return ten.to(device)


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_densenet_cls(
    fused_u8: np.ndarray,
    model: torch.nn.Module,
    img_size: int,
    device: torch.device,
) -> float:
    img = Image.fromarray(fused_u8, mode="L").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob = float(torch.sigmoid(logit)[0].item())
    return prob


def _run_yolo(
    args: argparse.Namespace,
    fused_path: str,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency 'ultralytics', required for YOLO detection.") from exc

    model = YOLO(args.det_checkpoint)
    preds = model.predict(
        source=fused_path,
        imgsz=int(args.yolo_imgsz),
        conf=float(args.threshold),
        device=args.device,
        verbose=False,
        save=False,
    )
    pred = preds[0]
    boxes = []
    if pred.boxes is not None:
        xyxy = pred.boxes.xyxy.detach().cpu().numpy()
        confs = pred.boxes.conf.detach().cpu().numpy()
        clses = pred.boxes.cls.detach().cpu().numpy()
        for i in range(len(xyxy)):
            x0, y0, x1, y1 = xyxy[i].tolist()
            boxes.append(
                {
                    "bbox_px_xyxy": [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))],
                    "confidence": float(confs[i]),
                    "class_id": int(round(clses[i])),
                }
            )
    return {
        "detected": len(boxes) > 0,
        "num_boxes": len(boxes),
        "boxes": boxes,
    }


def _save_yolo_overlay(fused_u8: np.ndarray, yolo_result: dict, out_path: str) -> None:
    overlay = Image.fromarray(fused_u8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for item in yolo_result["boxes"]:
        x0, y0, x1, y1 = item["bbox_px_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    overlay.save(out_path)


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.inputs:
        raise SystemExit("No input images configured. Edit DEFAULT_CONFIG['inputs'] in this file.")
    for path in args.inputs:
        if "path\\to" in path or "path/to" in path:
            raise SystemExit(f"Placeholder path detected: {path}. Edit DEFAULT_CONFIG at the top of the file.")
        if not os.path.isfile(path):
            raise SystemExit(f"Input image not found: {path}")

    for name, path in (
        ("fusion_checkpoint", args.fusion_checkpoint),
        ("cls_checkpoint", args.cls_checkpoint),
        ("det_checkpoint", args.det_checkpoint),
    ):
        if not path:
            continue
        if "path\\to" in path or "path/to" in path:
            raise SystemExit(f"Placeholder {name} detected: {path}. Edit DEFAULT_CONFIG at the top of the file.")
        if not os.path.isfile(path):
            raise SystemExit(f"{name} not found: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if len(args.inputs) != len(args.modalities):
        raise SystemExit("--inputs and --modalities must have the same length.")
    if len(args.inputs) < 2 or len(args.inputs) > 4:
        raise SystemExit("This pipeline expects 2..4 modalities.")
    _validate_paths(args)

    mod_to_id = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
    try:
        mod_ids = torch.tensor([mod_to_id[m.strip().lower()] for m in args.modalities], dtype=torch.long)
    except KeyError as e:
        raise SystemExit(f"Unknown modality: {e}. Allowed: t1n t1c t2w t2f") from e

    device = torch.device(args.device)
    images = [_load_gray(path) for path in args.inputs]
    h, w = _resolve_common_shape(images, int(args.scale))
    cropped = [_center_crop(img, h, w) for img in images]

    x = _build_fusion_input(cropped, device=device)
    mod_ids = mod_ids.to(device)

    fusion_model = _load_fusion_model(args, device)
    with torch.no_grad():
        fused01 = fusion_model(x, mod_ids=mod_ids).detach().cpu().squeeze().numpy()
    fused_u8 = np.clip(fused01 * 255.0, 0, 255).astype(np.uint8)

    os.makedirs(args.out_dir, exist_ok=True)
    fused_path = os.path.join(args.out_dir, "fused.png")
    _save_gray(fused_path, fused_u8)

    result = {
        "inputs": [os.path.abspath(p) for p in args.inputs],
        "modalities": [m.strip().lower() for m in args.modalities],
        "fusion_checkpoint": os.path.abspath(args.fusion_checkpoint),
        "det_checkpoint": os.path.abspath(args.det_checkpoint),
        "cls_checkpoint": os.path.abspath(args.cls_checkpoint) if args.cls_checkpoint else "",
        "detector_type": "densenet_yolo",
        "fused_path": os.path.abspath(fused_path),
        "image_size": {"height": int(h), "width": int(w)},
        "threshold": float(args.threshold),
    }

    if not args.cls_checkpoint.strip():
        raise SystemExit("--cls_checkpoint is required.")

    cls_model, cls_img_size = _load_cls_model(args, device)
    cls_prob = _run_densenet_cls(fused_u8, cls_model, cls_img_size, device)
    result["classification_probability"] = cls_prob
    result["classification_detected"] = cls_prob >= float(args.threshold)
    result["classification_img_size"] = int(cls_img_size)
    if cls_prob < float(args.threshold):
        result["detected"] = False
        result["num_boxes"] = 0
        result["boxes"] = []
        result["overlay_path"] = ""
        result_path = os.path.join(args.out_dir, "result.json")
        _save_json(result_path, result)
        print(json.dumps(result, indent=2))
        return 0

    yolo_result = _run_yolo(args, fused_path)
    overlay_path = os.path.join(args.out_dir, "det_overlay.png")
    _save_yolo_overlay(fused_u8, yolo_result, overlay_path)
    result.update(yolo_result)
    result["overlay_path"] = os.path.abspath(overlay_path)

    result_path = os.path.join(args.out_dir, "result.json")
    _save_json(result_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

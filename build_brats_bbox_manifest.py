import argparse
import os
from typing import Optional, Sequence

from util.brats_bbox import build_bbox_manifest
from util.logger import get_logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Build bbox manifest CSV from fused BraTS slices + seg labels")
    p.add_argument(
        "--fused_dir",
        type=str,
        default=os.path.join("results_brats", "result", "recon"),
        help="Folder containing fused slice PNGs named like <case>_z###.png",
    )
    p.add_argument(
        "--brats_root",
        type=str,
        default=os.path.join("Dataset-BraTS", "BraTS2024-BraTS-GLI-TrainingData"),
        help="Root containing BraTS case folders with *-seg.nii.gz",
    )
    p.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"])
    p.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join("results_brats", "bbox_manifest.csv"),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    logger.info(f"fused_dir={args.fused_dir}")
    logger.info(f"brats_root={args.brats_root}")
    logger.info(f"axis={args.axis}")
    logger.info(f"out_csv={args.out_csv}")

    n = build_bbox_manifest(fused_dir=args.fused_dir, brats_root=args.brats_root, axis=args.axis, out_csv=args.out_csv)
    logger.info(f"Done. Wrote {n} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


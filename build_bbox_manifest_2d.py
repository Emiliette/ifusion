import argparse
import os
from typing import Optional, Sequence

from util.brats_bbox import build_bbox_manifest_from_seg_png
from util.logger import get_logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Build bbox manifest CSV from fused 2D PNG slices + seg PNG slices")
    p.add_argument("--fused_dir", type=str, required=True, help="Folder containing fused PNG slices: <case>_z###.png")
    p.add_argument("--seg_dir", type=str, required=True, help="Folder containing seg PNG slices: <case>_z###.png")
    p.add_argument("--out_csv", type=str, default=os.path.join("results_brats", "bbox_manifest_2d.csv"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    logger.info(f"fused_dir={args.fused_dir}")
    logger.info(f"seg_dir={args.seg_dir}")
    logger.info(f"out_csv={args.out_csv}")

    n = build_bbox_manifest_from_seg_png(fused_dir=args.fused_dir, seg_dir=args.seg_dir, out_csv=args.out_csv)
    logger.info(f"Done. Wrote {n} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


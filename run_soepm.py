# run_soepm.py
# Quick test runner for SOEPM

import argparse
import os
import cv2
from soepm import SOEPM, load_bgr, save_bgr

def main():
    parser = argparse.ArgumentParser(description="Run SOEPM on an image (inference-only).")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", required=True, help="Path to save output image")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    parser.add_argument("--no-sharpen", action="store_true", help="Disable sharpening")
    parser.add_argument("--x2", action="store_true", help="Enable 2x upscaling")
    parser.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit (default 2.0)")
    parser.add_argument("--grid", type=int, nargs=2, default=[8, 8], help="CLAHE tile grid size (e.g., 8 8)")
    parser.add_argument("--sharpen", type=float, default=1.0, help="Sharpen strength (0=off, 1=default)")
    args = parser.parse_args()

    img = load_bgr(args.input)

    so = SOEPM(
        use_clahe=not args.no_clahe,
        use_sharpen=not args.no_sharpen,
        upscale_x2=args.x2,
        clahe_clip_limit=args.clip,
        clahe_tile_grid_size=tuple(args.grid),
        sharpen_strength=args.sharpen,
    )

    out = so(img)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_bgr(args.output, out)

    h, w = out.shape[:2]
    print(f"[OK] Saved: {args.output} (size: {w}x{h})")

if __name__ == "__main__":
    main()

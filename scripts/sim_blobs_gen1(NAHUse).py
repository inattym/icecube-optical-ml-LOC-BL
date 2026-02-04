import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add repo root

import csv, random
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

def blob(h, w, cx, cy, sigma):
    y, x = np.ogrid[:h, :w]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=128, help="Output H=W in pixels")
    ap.add_argument("--n", type=int, default=800, help="Total samples to generate")
    ap.add_argument("--train_frac", type=float, default=0.75, help="Train split fraction")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    # --- Always save to top-level data/camsim ---
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "data" / "camsim"
    img_dir = root / "images"
    meta_dir = root / "meta"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)


    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    H = W = int(args.img_size)

    # --- Geometry scaling ---
    # Your original used: cx = W/2 + x*0.5 pixels at 64x64
    # Keep the same "physical" range (x,y in [-16,16]) but scale pixels/unit with resolution.
    # At 64x64, pixels_per_unit = 0.5 => generalize: pixels_per_unit = W / 128
    pixels_per_unit = W / 128.0

    rows = []
    for i in range(args.n):
        la = rng.uniform(20, 40)
        ls = rng.uniform(30, 60)
        x  = rng.uniform(-16, 16)   # "physical" units
        y  = rng.uniform(-16, 16)

        # sigma scaling:
        # Original: sigma = 4 + 0.08*(ls-30)  (pixels @ 64x64).
        # Preserve shape across resolutions by scaling with (W/64).
        sigma64 = 4.0 + 0.08 * (ls - 30.0)
        sigma   = sigma64 * (W / 64.0)

        cx = W / 2.0 + x * pixels_per_unit
        cy = H / 2.0 + y * pixels_per_unit

        im = blob(H, W, cx, cy, sigma)
        im = (im / (im.max() + 1e-12) * 255.0).astype(np.uint8)

        p = img_dir / f"fake_{i:04d}.png"
        Image.fromarray(im).save(p)

        # Write RELATIVE path for portability
        rows.append([str(p), f"{la:.3f}", f"{ls:.3f}", f"{x:.3f}", f"{y:.3f}"])

    # --- split ---
    n_train = int(round(args.train_frac * len(rows)))
    splits = [("train.csv", rows[:n_train]), ("val.csv", rows[n_train:])]

    for name, split in splits:
        with open(meta_dir / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "la", "ls", "x", "y"])
            w.writerows(split)

    print(f"Wrote {n_train} train and {len(rows) - n_train} val at {H}x{W}px to {root}")

if __name__ == "__main__":
    main()

    




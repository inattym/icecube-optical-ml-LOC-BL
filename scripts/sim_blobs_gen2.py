#!/usr/bin/env python3
"""
Physics-based fake blob generator.

Single underlying physical model:

  r(x, y) = sqrt((x - cx)^2 + (y - cy)^2)

  A(z)      = z^{-2} * exp(-z / L_att)
  sigma(z)  = sigma0_pix * sqrt(1 + z / z0)
  I(r, z)   = A(z) * exp(-r^2 / (2 sigma(z)^2))

Modes are different "views" of this PSF:

  - fixed:
      Top-hat disk where intensity exceeds a fixed fraction f_iso of the peak.
      radius R_pix:
          I(R_pix, z) = f_iso * I(0, z)
          R_pix = sigma(z) * sqrt(2 ln(1 / f_iso))

  - gauss2d:
      Gaussian PSF at a single fixed distance z_fixed.
      I_gauss2d(r) = I(r, z_fixed).

  - gauss_depth:
      PSF with distance z sampled in [z_min, z_max].

All coordinates (x, y, r, cx, cy, sigma, R_pix) are in PIXELS.
z, z0, L_att are in consistent distance units (arbitrary but the same).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------- shared geometry ----------

def make_radius_grid(img_size: int, cx: float, cy: float) -> np.ndarray:
    """
    Radial distance r(x, y) from center (cx, cy) on an img_size x img_size grid.
    All coordinates in pixels.
    """
    y, x = np.mgrid[0:img_size, 0:img_size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return r


# ---------- A(z), sigma(z), I(r,z) ----------

def sigma_of_z(z: float, sigma0_pix: float, z0: float) -> float:
    """
    sigma(z) = sigma0_pix * sqrt(1 + z / z0)
    Diffusion-like widening of the PSF with distance (sigma^2 ~ sigma0^2 + const * z).
    """
    return sigma0_pix * np.sqrt(1.0 + z / z0)


def amplitude_of_z(z: float, L_att: float) -> float:
    """
    A(z) = z^{-2} * exp(-z / L_att)

    Geometric inverse-square falloff (1 / z^2) times
    Beer–Lambert attenuation exp(-z / L_att).
    """
    return (1.0 / (z ** 2)) * np.exp(-z / L_att)


def physical_psf(
    img_size: int,
    cx: float,
    cy: float,
    z: float,
    sigma0_pix: float,
    z0: float,
    L_att: float,
) -> tuple[np.ndarray, float, float]:
    """
    Full physical PSF:

        I(r, z) = A(z) * exp(-r^2 / (2 sigma(z)^2))

    Returns:
      img     : 2D array of intensities
      sigma_z : sigma(z) in pixels
      A_z     : amplitude A(z)
    """
    r = make_radius_grid(img_size, cx, cy)
    sigma_z = sigma_of_z(z, sigma0_pix, z0)
    A_z = amplitude_of_z(z, L_att)

    img = np.exp(-r ** 2 / (2.0 * sigma_z ** 2))
    img /= img.max()  # normalize PSF peak to 1
    img *= A_z        # apply physical amplitude

    return img.astype(np.float32), sigma_z, A_z


# ---------- MODE 1: fixed (isophote top-hat from PSF) ----------

def blob_fixed_from_physics(
    img_size: int,
    cx: float,
    cy: float,
    z: float,
    sigma0_pix: float,
    z0: float,
    L_att: float,
    f_iso: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    fixed mode.

    Define radius R_pix as the contour where intensity drops
    to a fraction f_iso of the peak:

        I(R_pix, z) = f_iso * I(0, z)
        => R_pix = sigma(z) * sqrt(2 ln(1 / f_iso))

    Then make a top-hat disk:
        I_fixed(r) = 1 if r <= R_pix, 0 otherwise.

    Returns:
      img     : binary disk image (float32 in [0,1])
      R_pix   : disk radius in pixels
      sigma_z : sigma(z) in pixels
      A_z     : A(z) (for logging only)
    """
    sigma_z = sigma_of_z(z, sigma0_pix, z0)
    A_z = amplitude_of_z(z, L_att)

    # radius where Gaussian PSF falls to f_iso of its peak
    R_pix = sigma_z * np.sqrt(2.0 * np.log(1.0 / f_iso))

    r = make_radius_grid(img_size, cx, cy)
    img = (r <= R_pix).astype(np.float32)

    # normalize to [0, 1] (geometry only, not scaled by A(z))
    if img.max() > 0:
        img /= img.max()

    return img, R_pix, sigma_z, A_z


# ---------- MODE 2: gauss2d (PSF at fixed z_fixed) ----------

def blob_gauss2d_physical(
    img_size: int,
    cx: float,
    cy: float,
    z_fixed: float,
    sigma0_pix: float,
    z0: float,
    L_att: float,
) -> tuple[np.ndarray, float, float]:
    """
    'gauss2d' mode: physical PSF at a single fixed distance z_fixed.

        I_gauss2d(r) = I(r, z_fixed)

    Returns:
      img     : PSF image
      sigma_z : sigma(z_fixed)
      A_z     : A(z_fixed)
    """
    img, sigma_z, A_z = physical_psf(
        img_size=img_size,
        cx=cx,
        cy=cy,
        z=z_fixed,
        sigma0_pix=sigma0_pix,
        z0=z0,
        L_att=L_att,
    )
    return img, sigma_z, A_z


# ---------- MODE 3: gauss_depth (PSF with z sampled in [z_min, z_max]) ----------

def blob_gauss_depth_physical(
    img_size: int,
    cx: float,
    cy: float,
    z: float,
    sigma0_pix: float,
    z0: float,
    L_att: float,
) -> tuple[np.ndarray, float, float]:
    """
    'gauss_depth' mode: physical PSF with z drawn from [z_min, z_max].

        I_gauss_depth(r, z) = I(r, z)

    Returns:
      img     : PSF image
      sigma_z : sigma(z)
      A_z     : A(z)
    """
    img, sigma_z, A_z = physical_psf(
        img_size=img_size,
        cx=cx,
        cy=cy,
        z=z,
        sigma0_pix=sigma0_pix,
        z0=z0,
        L_att=L_att,
    )
    return img, sigma_z, A_z



# ---------- image saving  ----------

def save_raw_image(img: np.ndarray, out_path: Path):
    """
    Save a raw grayscale PNG (no axes, no title, no colorbar).
    Assumes img is float in [0,1].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    plt.imsave(out_path, arr, cmap="gray", vmin=0, vmax=255, origin="lower")



def save_image_with_title(
    img: np.ndarray,
    out_path: Path,
    mode: str,
    cx: float,
    cy: float,
    R_pix: float | None,
    sigma_pix: float | None,
    z: float | None,
):
    """
    Save grayscale image with a title showing mode, x, y, R, sigma, z.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(img, origin="lower", cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    parts = [
        f"mode={mode}",
        f"x={cx:.1f}",
        f"y={cy:.1f}",
    ]
    if R_pix is not None:
        parts.append(f"R_pix={R_pix:.2f}")
    if sigma_pix is not None:
        parts.append(f"sigma_pix={sigma_pix:.2f}")
    if z is not None:
        parts.append(f"z={z:.2f}")

    title = " | ".join(parts)
    ax.set_title(title, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- main script ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=128,
                    help="Height=Width of output images in pixels")
    ap.add_argument("--n_per_mode", type=int, default=100,
                    help="How many images per mode to generate")
    ap.add_argument("--raw_dir", type=str, default="data/camsim/images_raw",
                    help="Directory to save RAW PNGs (training)")
    ap.add_argument("--ann_dir", type=str, default="data/camsim/images_ann",
                    help="Directory to save annotated PNGs (debug)")
    ap.add_argument("--csv_path", type=str, default="data/camsim/meta/blob_metadata_phys.csv",
                    help="Path to CSV metadata file")
    ap.add_argument("--seed", type=int, default=123,
                    help="Random seed")

    # physics parameters
    ap.add_argument("--sigma0_pix", type=float, default=1.5,
                    help="Base PSF width at z=0 (in pixels)")
    ap.add_argument("--z0", type=float, default=20.0,
                    help="Distance scale for sigma(z)")
    ap.add_argument("--L_att", type=float, default=5.0,
                    help="Attenuation length (same units as z)")

    # distance range
    ap.add_argument("--z_min", type=float, default=1.0,
                    help="Minimum distance z for gauss_depth and fixed")
    ap.add_argument("--z_max", type=float, default=10.0,
                    help="Maximum distance z for gauss_depth and fixed")
    ap.add_argument("--z_fixed", type=float, default=3.0,
                    help="Fixed distance for gauss2d mode")

    # fixed-mode isophote fraction
    ap.add_argument("--f_iso", type=float, default=0.1,
                    help="Isophote fraction for fixed mode (0 < f_iso < 1)")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)



    raw_dir = Path(args.raw_dir)
    ann_dir = Path(args.ann_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] RAW images -> {raw_dir}")
    print(f"[info] ANN images -> {ann_dir}")
    print(f"[info] Writing CSV to: {csv_path}")




    # CSV
    with csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "filename",
            "mode",
            "x_pix",
            "y_pix",
            "R_pix",
            "sigma_pix",
            "z",
            "sigma_depth_pix",
            "A_z",
        ])

        #modes = ["fixed", "gauss2d", "gauss_depth"]
        modes = ["gauss2d"]


        for mode in modes:
            for i in range(args.n_per_mode):
                # keep away from borders
                margin = 0.15 * args.img_size
                cx = rng.uniform(margin, args.img_size - margin)
                cy = rng.uniform(margin, args.img_size - margin)

                # defaults for logging
                R_pix = None
                sigma_pix = None
                z = None
                sigma_depth_pix = None
                A_z = None

                if mode == "fixed":
                    # draw z in [z_min, z_max], compute isophote disk
                    z = rng.uniform(args.z_min, args.z_max)
                    img, R_pix, sigma_depth_pix, A_z = blob_fixed_from_physics(
                        img_size=args.img_size,
                        cx=cx,
                        cy=cy,
                        z=z,
                        sigma0_pix=args.sigma0_pix,
                        z0=args.z0,
                        L_att=args.L_att,
                        f_iso=args.f_iso,
                    )
                    sigma_pix = sigma_depth_pix

                elif mode == "gauss2d":
                    # use fixed z_fixed for all gauss2d samples
                    z = args.z_fixed
                    img, sigma_depth_pix, A_z = blob_gauss2d_physical(
                        img_size=args.img_size,
                        cx=cx,
                        cy=cy,
                        z_fixed=z,
                        sigma0_pix=args.sigma0_pix,
                        z0=args.z0,
                        L_att=args.L_att,
                    )
                    sigma_pix = sigma_depth_pix

                elif mode == "gauss_depth":
                    # sample z in [z_min, z_max]
                    z = rng.uniform(args.z_min, args.z_max)
                    img, sigma_depth_pix, A_z = blob_gauss_depth_physical(
                        img_size=args.img_size,
                        cx=cx,
                        cy=cy,
                        z=z,
                        sigma0_pix=args.sigma0_pix,
                        z0=args.z0,
                        L_att=args.L_att,
                    )
                    sigma_pix = sigma_depth_pix

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Normalize image to [0,1]
                img = img - img.min()
                if img.max() > 0:
                    img = img / img.max()


                # --- RAW for training ---
                fname_raw = f"phys_{mode}_{i:04d}.png"          # optional: drop _raw since folder is images_raw
                out_path_raw = raw_dir / fname_raw
                save_raw_image(img, out_path_raw)

                # --- Annotated ---
                fname_ann = f"phys_{mode}_{i:04d}_ann.png"
                out_path_ann = ann_dir / fname_ann
                save_image_with_title(
                    img=img,
                    out_path=out_path_ann,
                    mode=mode,
                    cx=cx,
                    cy=cy,
                    R_pix=R_pix,
                    sigma_pix=sigma_pix,
                    z=z,
                )

                # write CSV using RAW filename (training uses raw)
                writer.writerow([
                    fname_raw,
                    mode,
                    f"{cx:.6f}",
                    f"{cy:.6f}",
                    f"{R_pix:.6f}" if R_pix is not None else "",
                    f"{sigma_pix:.6f}" if sigma_pix is not None else "",
                    f"{z:.6f}" if z is not None else "",
                    f"{sigma_depth_pix:.6f}" if sigma_depth_pix is not None else "",
                    f"{A_z:.8e}" if A_z is not None else "",
                ])

                print(f"[info] Saved RAW {out_path_raw} | ANN {out_path_ann}")




    print(f"[info] Wrote metadata to {csv_path}")


if __name__ == "__main__":
    main()






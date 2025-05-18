# stellar_photometry_hr.py
"""
End-to-end photometry + H-R diagram pipeline.

* Uses `preprocess_methods.build_calibration()` to obtain:
    - k_B, k_V  (extinction coefficients)
    - STD_B, STD_V  (standard-star flux & airmass)
* Uses `fetch_standard_info()` / `fetch_cluster_distance()` so
  catalog magnitudes, target distance are **not hard-coded**.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os, math, glob, warnings
from typing import List, Dict, Sequence

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.stats import mad_std
from astropy.modeling import models, fitting
from matplotlib.patches import Rectangle

from preprocess_methods import (
    build_calibration,
    fetch_standard_info,
    fetch_cluster_distance,
)

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
DRIVE_ROOT   = "/content/drive/MyDrive/최원섭/2024KSHS/윤철 탐논"
CALIB_DIR    = os.path.join(DRIVE_ROOT, "Master Frame")
SCIENCE_DIR  = os.path.join(DRIVE_ROOT, "Targets")
OUTPUT_DIR   = os.path.join(DRIVE_ROOT, "output")
CSV_BASENAME = "m45.csv"

EXTINCTION_FITS = sorted(
    glob.glob(os.path.join(DRIVE_ROOT, "대기소광", "capella", "3", "*.fits"))
)

BIAS_FILES   = glob.glob(os.path.join(CALIB_DIR, "Master_Bias*.fit"))
DARK_FILES   = glob.glob(os.path.join(CALIB_DIR, "Master_Dark*.fit"))
FLAT_B_FILES = glob.glob(os.path.join(CALIB_DIR, "Master_Flat B*.fit"))
FLAT_V_FILES = glob.glob(os.path.join(CALIB_DIR, "Master_Flat V*.fit"))

STANDARD_STAR_NAME = "Capella"
TARGET_NAME        = "M45"

ROI_X = (0, 2262)
ROI_Y = (0, 1812)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_fits(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with fits.open(path) as h:
        return h[0].data.copy(), h[0].header.copy()

def create_master_frame(lst: Sequence[str]):
    return load_fits(lst[0])[0] if len(lst) == 1 else np.median(
        np.stack([load_fits(f)[0] for f in lst]), axis=0
    )

def moffat_fwhm(sub: np.ndarray):
    yy, xx = np.mgrid[: sub.shape[0], : sub.shape[1]]
    init = models.Moffat2D(
        amplitude=sub.max() - np.median(sub),
        x_0=sub.shape[1] / 2,
        y_0=sub.shape[0] / 2,
        gamma=1.5,
        alpha=2.5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = fitting.LevMarLSQFitter()(init, xx, yy, sub)
            return float(
                np.clip(
                    2 * fit.gamma.value * math.sqrt(2 ** (1 / fit.alpha.value) - 1),
                    1,
                    20,
                )
            )
        except Exception:
            return 3.0

def calc_flux(img, x, y, r):
    from photutils.aperture import (
        CircularAperture,
        CircularAnnulus,
        aperture_photometry,
    )

    ap = CircularAperture((x, y), r)
    ann = CircularAnnulus((x, y), r * 1.5, r * 2)
    phot = aperture_photometry(img, [ap, ann])
    ann_mask = ann.to_mask(method="center").multiply(img)
    bkg = np.mean(ann_mask[~np.isnan(ann_mask)])
    return float(phot["aperture_sum_0"][0] - bkg * math.pi * r**2)

def detect_stars(img, sigma=5):
    from photutils.detection import DAOStarFinder

    finder = DAOStarFinder(fwhm=3, threshold=sigma * mad_std(img))
    src = finder(img)
    if src is None:
        return []
    stars = []
    for row in src:
        flux = calc_flux(img, row["xcentroid"], row["ycentroid"], 4)
        stars.append(
            dict(x=row["xcentroid"], y=row["ycentroid"], flux=flux)
        )
    return stars

def match_catalog(a, b, r=5):
    tree = cKDTree([(s["x"], s["y"]) for s in b])
    return [
        (ia, int(tree.query([sa["x"], sa["y"]], 1)[1]))
        for ia, sa in enumerate(a)
        if tree.query([sa["x"], sa["y"]], 1)[0] <= r
    ]

def save_catalog(rows: List[Dict], csv_path: str):
    import csv

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["Index", "X_B", "Y_B", "Flux_B", "B_mag", "X_V", "Y_V",
             "Flux_V", "V_mag", "B-V"]
        )
        for r in rows:
            w.writerow(
                [
                    r[k] if not isinstance(r[k], float) else f"{r[k]:.3f}"
                    for k in [
                        "index",
                        "x_B",
                        "y_B",
                        "flux_B",
                        "B_mag",
                        "x_V",
                        "y_V",
                        "flux_V",
                        "V_mag",
                        "B-V",
                    ]
                ]
            )

def plot_field(img, stars, png):
    plt.figure(figsize=(9, 7))
    vmin, vmax = np.percentile(img, [1, 99])
    plt.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    for s in stars:
        plt.text(
            s["x_V"],
            s["y_V"],
            str(s["index"]),
            color="red",
            fontsize=6,
            ha="center",
            va="center",
        )
    plt.gca().add_patch(
        Rectangle(
            (ROI_X[0], ROI_Y[0]),
            ROI_X[1] - ROI_X[0],
            ROI_Y[1] - ROI_Y[0],
            lw=1,
            ec="yellow",
            ls=":",
            fc="none",
        )
    )
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_hr(csv_path: str, dist: float | None):
    df = pd.read_csv(csv_path)
    x = df["B-V"]
    y = (
        df["V_mag"] - 5 * (np.log10(dist) - 1)
        if dist
        else df["V_mag"]
    )
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=12)
    for idx, bv, mag in zip(df["Index"], x, y):
        plt.text(bv + 0.02, mag, str(idx), fontsize=4, color="blue")
    plt.gca().invert_yaxis()
    plt.xlabel("B–V")
    plt.ylabel("Abs Mag" if dist else "App Mag")
    plt.grid()
    plt.savefig(
        os.path.splitext(csv_path)[0] + "_H-R.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Lookup catalogue data & distance ------------------------
    std_info = fetch_standard_info(STANDARD_STAR_NAME)
    target_dist_pc = fetch_cluster_distance(TARGET_NAME)

    # -- Build nightly extinction & standard-star flux -----------
    k_B, k_V, STD_B, STD_V = build_calibration(
        EXTINCTION_FITS,
        BIAS_FILES,
        DARK_FILES,
        FLAT_B_FILES,
        FLAT_V_FILES,
        std_info["B"],
        std_info["V"],
        verbose=True,
    )

    # -- Master frames for science images -----------------------
    bias = create_master_frame(BIAS_FILES)
    dark = create_master_frame(DARK_FILES) - bias
    flat_b = (create_master_frame(FLAT_B_FILES) - bias) / np.median(
        create_master_frame(FLAT_B_FILES) - bias
    )
    flat_v = (create_master_frame(FLAT_V_FILES) - bias) / np.median(
        create_master_frame(FLAT_V_FILES) - bias
    )

    master_b = create_master_frame(
        sorted(glob.glob(os.path.join(SCIENCE_DIR, "*B*.fits")))
    )
    master_v = create_master_frame(
        sorted(glob.glob(os.path.join(SCIENCE_DIR, "*V*.fits")))
    )
    img_b = (master_b - bias - dark) / flat_b
    img_v = (master_v - bias - dark) / flat_v

    # -- Star detection / matching -------------------------------
    stars_b = detect_stars(img_b)
    stars_v = detect_stars(img_v)
    matches = match_catalog(stars_b, stars_v)

    hdr_b = load_fits(
        sorted(glob.glob(os.path.join(SCIENCE_DIR, "*B*.fits")))[0]
    )[1]
    hdr_v = load_fits(
        sorted(glob.glob(os.path.join(SCIENCE_DIR, "*V*.fits")))[0]
    )[1]
    airm_b = hdr_b.get("AIRMASS", 1.0)
    airm_v = hdr_v.get("AIRMASS", 1.0)

    def mag(flux, std, k, airm):
        return std["catalog_m"] - 2.5 * np.log10(flux / std["flux"]) - k * airm

    rows = []
    for idx, (ib, iv) in enumerate(matches, 1):
        sb, sv = stars_b[ib], stars_v[iv]
        m_b = mag(sb["flux"], STD_B, k_B, airm_b)
        m_v = mag(sv["flux"], STD_V, k_V, airm_v)
        rows.append(
            dict(
                index=idx,
                x_B=sb["x"],
                y_B=sb["y"],
                flux_B=sb["flux"],
                B_mag=m_b,
                x_V=sv["x"],
                y_V=sv["y"],
                flux_V=sv["flux"],
                V_mag=m_v,
                **{"B-V": m_b - m_v},
            )
        )

    rows.sort(key=lambda s: (-s["y_V"], s["x_V"]))

    csv_path = os.path.join(OUTPUT_DIR, CSV_BASENAME)
    save_catalog(rows, csv_path)
    plot_field(img_v, rows, os.path.splitext(csv_path)[0] + ".png")
    plot_hr(csv_path, target_dist_pc)
    print("Pipeline complete → outputs saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

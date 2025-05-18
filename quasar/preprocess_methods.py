# preprocess_methods.py
"""
Utility module for atmospheric-extinction calibration
and standard-star photometry.

Public API
----------
load_fits_file(path)                 → (data, header)
create_master_frame(file_list)       → ndarray
detect_stars(image)                  → List[dict]
find_brightest_star(stars)           → dict | None
process_single_fits(path, …)         → (instrumental_mag, airmass, flux)
estimate_extinction_coefficients(results)
select_standard_star(results)
build_calibration(…)                 → k_B, k_V, STD_B, STD_V

Lookup helpers (NEW)
--------------------
fetch_standard_info(star_name)       → {"B": m_B, "V": m_V}
fetch_cluster_distance(target_name)  → distance (pc)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import os, math, glob, warnings, json
from typing import List, Tuple, Dict, Sequence

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u
from scipy.ndimage import label, find_objects
from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus,
)
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Built-in reference data (간단 예시 — 필요 시 확장)
# ---------------------------------------------------------------------------
_STANDARD_CATALOG: Dict[str, Dict[str, float]] = {
    "CAPELLA": {"B": 0.84, "V": 0.05},
    # "VEGA":    {"B": 0.03, "V": 0.03},
}

_CLUSTER_DISTANCE_PC: Dict[str, float] = {
    "M45": 136.2,  # Pleiades
    # "HYADES": 46.3,
}

def fetch_standard_info(star_name: str) -> Dict[str, float]:
    """Johnson B/V catalogue magnitudes for *star_name* (case-insensitive)."""
    key = star_name.strip().upper()
    if key not in _STANDARD_CATALOG:
        raise KeyError(
            f"Standard star '{star_name}' not in catalogue; "
            "add it to _STANDARD_CATALOG."
        )
    return _STANDARD_CATALOG[key]

def fetch_cluster_distance(target_name: str) -> float:
    """Return distance (pc) for *target_name* cluster/target."""
    key = target_name.strip().upper()
    if key not in _CLUSTER_DISTANCE_PC:
        raise KeyError(
            f"Target '{target_name}' distance unknown; "
            "add to _CLUSTER_DISTANCE_PC."
        )
    return _CLUSTER_DISTANCE_PC[key]

# ---------------------------------------------------------------------------
# Low-level FITS helpers
# ---------------------------------------------------------------------------
def load_fits_file(path: str):
    """Return (data, header) with float32 data copy."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32), hdul[0].header.copy()

def create_master_frame(file_list: Sequence[str]) -> np.ndarray:
    """Median-stack frames into a single master frame."""
    if not file_list:
        raise ValueError("Empty file list for master frame.")
    if len(file_list) == 1:
        return load_fits_file(file_list[0])[0]
    stack = np.stack([load_fits_file(f)[0] for f in file_list])
    return np.median(stack, axis=0)

# ---------------------------------------------------------------------------
# Very simple star detection (threshold + blob labelling)
# ---------------------------------------------------------------------------
def detect_stars(image: np.ndarray, threshold_factor: float = 30.0):
    median = np.median(image)
    std = mad_std(image)
    thresh = median + threshold_factor * std
    mask = image > thresh
    labels, nsrc = label(mask)
    if nsrc == 0:
        return []
    slices = find_objects(labels)
    stars = []
    for i, sl in enumerate(slices):
        y_idx, x_idx = np.where(labels[sl] == (i + 1))
        y_idx += sl[0].start
        x_idx += sl[1].start
        flux = float(np.sum(image[y_idx, x_idx]))
        x_cen = float(np.mean(x_idx))
        y_cen = float(np.mean(y_idx))
        radius = math.sqrt(len(x_idx) / math.pi)
        stars.append(dict(flux=flux, x=x_cen, y=y_cen, radius=radius))
    return stars

def find_brightest_star(stars):
    return max(stars, key=lambda s: s["flux"]) if stars else None

# ---------------------------------------------------------------------------
# Airmass & flux helpers
# ---------------------------------------------------------------------------
def _calc_altitude(header):
    loc = EarthLocation(
        lat=float(header["OBS-LAT"]) * u.deg,
        lon=float(header["OBS-LONG"]) * u.deg,
    )
    t = Time(header["DATE-OBS"])
    sc = SkyCoord(
        ra=float(header["RA"]) * u.hourangle,
        dec=float(header["DEC"]) * u.deg,
    )
    altaz = sc.transform_to(AltAz(obstime=t, location=loc))
    return float(altaz.alt.deg)

def calc_airmass(header) -> float:
    """Gueymard 1993 polynomial if AIRMASS absent."""
    if "AIRMASS" in header:
        return float(header["AIRMASS"])
    required = ("RA", "DEC", "DATE-OBS", "OBS-LAT", "OBS-LONG")
    if not all(k in header for k in required):
        raise ValueError("Header lacks keys for airmass calculation")
    alt = _calc_altitude(header)
    secz = 1 / np.cos(np.deg2rad(90 - alt))
    return float(
        secz
        - 0.0018167 * (secz - 1)
        - 0.002875 * (secz - 1) ** 2
        - 0.0008083 * (secz - 1) ** 3
    )

def calc_flux(image: np.ndarray, x: float, y: float, r: float) -> float:
    ap = CircularAperture((x, y), r)
    ann = CircularAnnulus((x, y), r * 1.5, r * 2.0)
    phot = aperture_photometry(image, [ap, ann])
    ann_mask = ann.to_mask(method="center").multiply(image)
    bkg_mean = np.mean(ann_mask[~np.isnan(ann_mask)])
    return float(phot["aperture_sum_0"][0] - bkg_mean * math.pi * r**2)

# ---------------------------------------------------------------------------
# Frame-level processing
# ---------------------------------------------------------------------------
def process_single_fits(
    file_path: str,
    master_bias: np.ndarray,
    master_dark: np.ndarray,
    master_flat: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Returns instrumental magnitude, airmass, raw flux of brightest star.
    """
    data, hdr = load_fits_file(file_path)
    img = (data - master_bias - master_dark) / master_flat
    stars = detect_stars(img)
    br = find_brightest_star(stars)
    if br is None:
        raise RuntimeError(f"No star in {file_path}")
    flux = calc_flux(img, br["x"], br["y"], br["radius"])
    mag = -2.5 * np.log10(flux)
    return mag, calc_airmass(hdr), flux

# ---------------------------------------------------------------------------
# Extinction fit & standard-star selection
# ---------------------------------------------------------------------------
def estimate_extinction_coefficients(results: List[Tuple[float, float]]):
    """
    Linear fit  mag = k × airmass + c  → returns k, c.
    """
    if len(results) < 2:
        raise ValueError("Need at least two frames for extinction fit")
    x = np.array([r[1] for r in results]).reshape(-1, 1)
    y = np.array([r[0] for r in results])
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0]), float(model.intercept_)

def select_standard_star(results: List[Tuple[float, float, float]]):
    """
    Pick frame with lowest airmass as standard; returns dict{'flux','airmass'}.
    """
    best = min(results, key=lambda r: r[1])  # (mag, airmass, flux)
    return {"flux": best[2], "airmass": best[1]}

# ---------------------------------------------------------------------------
# One-stop public builder
# ---------------------------------------------------------------------------
def build_calibration(
    fits_files: Sequence[str],
    bias_files: Sequence[str],
    dark_files: Sequence[str],
    flat_b_files: Sequence[str],
    flat_v_files: Sequence[str],
    catalog_mag_b: float,
    catalog_mag_v: float,
    verbose: bool = False,
):
    """
    Returns k_B, k_V, STD_B, STD_V   (STD_* dict → {'flux','catalog_m','airmass'})
    """
    # --- Split extinction sequence by filter ---
    b_files, v_files = [], []
    for f in fits_files:
        _, h = load_fits_file(f)
        filt = h.get("FILTER", "UNK").strip().upper()
        (b_files if filt == "B" else v_files if filt == "V" else []).append(f)
    if verbose:
        print(f"[build_calibration] {len(b_files)} B frames, {len(v_files)} V frames")

    # --- Master frames (compute once) ---
    bias = create_master_frame(bias_files)
    dark = create_master_frame(dark_files) - bias
    flat_b_raw = create_master_frame(flat_b_files) - bias
    flat_v_raw = create_master_frame(flat_v_files) - bias
    flat_b = flat_b_raw / np.median(flat_b_raw)
    flat_v = flat_v_raw / np.median(flat_v_raw)

    def gather(flist, flat):
        res = []
        for f in flist:
            mag, am, flux = process_single_fits(f, bias, dark, flat)
            res.append((mag, am, flux))
        return res

    res_b = gather(b_files, flat_b)
    res_v = gather(v_files, flat_v)

    k_b, _ = estimate_extinction_coefficients([(m, a) for m, a, _ in res_b])
    k_v, _ = estimate_extinction_coefficients([(m, a) for m, a, _ in res_v])

    std_b_info = select_standard_star(res_b)
    std_b_info["catalog_m"] = catalog_mag_b
    std_v_info = select_standard_star(res_v)
    std_v_info["catalog_m"] = catalog_mag_v

    if verbose:
        print(f"[build_calibration] k_B={k_b:.4f}, k_V={k_v:.4f}")

    return k_b, k_v, std_b_info, std_v_info

# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "[preprocess_methods] Stand-alone test — "
        "provide your own extinction-sequence FITS paths."
    )

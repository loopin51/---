"""
main.py – Gradio GUI with ROI sliders & download buttons
=======================================================
* Extends previous interface:
  - Two **RangeSliders** for ROI (X‑range, Y‑range)
  - Live preview of the first uploaded science FITS with a green rectangle
    that moves as sliders change
  - Download buttons for the star‑field PNG, H‑R diagram PNG, and CSV
"""
from __future__ import annotations

import os, shutil, tempfile, io, csv, math, warnings
from typing import List, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from astroquery.simbad import Simbad
from astropy.wcs import WCS
import gradio as gr

from preprocess_methods import (
    build_calibration,
    fetch_standard_info,
    fetch_cluster_distance,
    load_fits_file,
    create_master_frame,
)
from astropy.stats import mad_std
from scipy.spatial import cKDTree
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
)
from astropy.modeling import models, fitting

# ------------------------ utility helpers ----------------------------

def moffat_fwhm(sub):
    yy, xx = np.mgrid[: sub.shape[0], : sub.shape[1]]
    init = models.Moffat2D(
        amplitude=sub.max() - np.median(sub),
        x_0=sub.shape[1] / 2,
        y_0=sub.shape[0] / 2,
        gamma=1.5,
        alpha=2.5,
    )
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
    ap = CircularAperture((x, y), r)
    ann = CircularAnnulus((x, y), r * 1.5, r * 2.0)
    phot = aperture_photometry(img, [ap, ann])
    ann_mask = ann.to_mask(method="center").multiply(img)
    bkg = np.mean(ann_mask[~np.isnan(ann_mask)])
    return float(phot["aperture_sum_0"][0] - bkg * math.pi * r**2)

def detect_stars(img, sigma=5):
    from photutils.detection import DAOStarFinder

    finder = DAOStarFinder(fwhm=3, threshold=sigma * mad_std(img))
    src = finder(img)
    return [] if src is None else [
        dict(x=row["xcentroid"], y=row["ycentroid"], flux=calc_flux(img, row["xcentroid"], row["ycentroid"], 4))
        for row in src
    ]

def match_catalog(a, b, r=5):
    tree = cKDTree([(s["x"], s["y"]) for s in b])
    return [
        (ia, int(tree.query([sa["x"], sa["y"]], 1)[1]))
        for ia, sa in enumerate(a)
        if tree.query([sa["x"], sa["y"]], 1)[0] <= r
    ]

# ------------------------ core functions -----------------------------

def make_preview(science_files: List[Tuple[str, bytes]], x_range, y_range):
    """Return path to preview PNG with ROI rectangle."""
    if not science_files:
        return None
    # save first file to tmp
    tmp = tempfile.mkdtemp()
    name, data = science_files[0]
    path = os.path.join(tmp, os.path.basename(name))
    with open(path, "wb") as f:
        f.write(data)
    img, _ = load_fits_file(path)
    plt.figure(figsize=(5, 4))
    vmin, vmax = np.percentile(img, [1, 99])
    plt.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    # green rectangle
    plt.gca().add_patch(
        plt.Rectangle(
            (x_range[0], y_range[0]),
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            lw=1.5,
            ec="lime",
            fc="none",
        )
    )
    plt.axis("off")
    out = os.path.join(tmp, "preview.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def run_pipeline(
    extinction_files,
    bias_files,
    dark_files,
    flatb_files,
    flatv_files,
    science_files,
    standard_star,
    target_name,
    x_range,
    y_range,
):
    # temp workspace
    workdir = tempfile.mkdtemp()
    def save(lst, sub):
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)
        out = []
        for name, data in lst:
            p = os.path.join(workdir, sub, os.path.basename(name))
            with open(p, "wb") as f:
                f.write(data)
            out.append(p)
        return out

    EXT = save(extinction_files, "ext")
    BIAS = save(bias_files, "bias")
    DARK = save(dark_files, "dark")
    FB   = save(flatb_files, "flatb")
    FV   = save(flatv_files, "flatv")
    SCI  = save(science_files, "sci")

    # split science by filter
    SCI_B, SCI_V = [], []
    for p in SCI:
        _, h = load_fits_file(p)
        (SCI_B if h.get("FILTER", "").strip().upper() == "B" else SCI_V).append(p)

    std_info = fetch_standard_info(standard_star)
    dist_pc  = fetch_cluster_distance(target_name)

    k_B, k_V, STD_B, STD_V = build_calibration(EXT, BIAS, DARK, FB, FV, std_info["B"], std_info["V"], verbose=False)

    bias = create_master_frame(BIAS)
    dark = create_master_frame(DARK) - bias
    flat_b = (create_master_frame(FB) - bias) / np.median(create_master_frame(FB) - bias)
    flat_v = (create_master_frame(FV) - bias) / np.median(create_master_frame(FV) - bias)

    master_b = create_master_frame(SCI_B)
    master_v = create_master_frame(SCI_V)
    img_b = (master_b - bias - dark) / flat_b
    img_v = (master_v - bias - dark) / flat_v

    # crop to ROI
    xs, xe = int(x_range[0]), int(x_range[1])
    ys, ye = int(y_range[0]), int(y_range[1])
    img_b_roi = img_b[ys:ye, xs:xe]
    img_v_roi = img_v[ys:ye, xs:xe]

    stars_b = detect_stars(img_b_roi)
    stars_v = detect_stars(img_v_roi)
    matches = match_catalog(stars_b, stars_v)

    def mag(flux, std, k):
        # use central airmass ~1 for ROI quick-look
        return std["catalog_m"] - 2.5 * np.log10(flux / std["flux"]) - k * 1.0

    rows=[]
    for idx,(ib,iv) in enumerate(matches,1):
        sb,sv=stars_b[ib],stars_v[iv]
        m_b=mag(sb['flux'],STD_B,k_B); m_v=mag(sv['flux'],STD_V,k_V)
        rows.append(dict(Index=idx,X_B=f"{sb['x']+xs:.1f}",Y_B=f"{sb['y']+ys:.1f}",Flux_B=f"{sb['flux']:.1f}",
                         B_mag=f"{m_b:.3f}",X_V=f"{sv['x']+xs:.1f}",Y_V=f"{sv['y']+ys:.1f}",Flux_V=f"{sv['flux']:.1f}",
                         V_mag=f"{m_v:.3f}",BV=f"{m_b-m_v:.3f}"))

    df=pd.DataFrame(rows)

    # Field PNG (within ROI)
    field_png=os.path.join(workdir,'field.png')
    plt.figure(figsize=(5,4)); vmin,vmax=np.percentile(img_v_roi,[1,99]);
    plt.imshow(img_v_roi,origin='lower',cmap='gray',vmin=vmin,vmax=vmax)
    for r in rows:
        plt.text(float(r['X_V'])-xs,float(r['Y_V'])-ys,str(r['Index']),color='red',fontsize=6)
    plt.axis('off'); plt.tight_layout(); plt.savefig(field_png,dpi=120); plt.close()

    # H-R diagram
    hr_png=os.path.join(workdir,'hr.png')
    plt.figure(figsize=(5,4))
    x=df['BV'].astype(float); y=df['V_mag'].astype(float)-5*(np.log10(dist_pc)-1)
    plt.scatter(x,y,s=12); [plt.text(bv+.02,mag,idx,fontsize=4) for bv,mag,idx in zip(x,y,df['Index'])]
    plt.gca().invert_yaxis(); plt.xlabel('B-V'); plt.ylabel('Absolute Mag'); plt.grid(); plt.tight_layout()
    plt.savefig(hr_png,dpi=120); plt.close()

    # CSV
    csv_path=os.path.join(workdir,'catalog.csv'); df.to_csv(csv_path,index=False)

    # File paths for download components
    return field_png, df.to_html(index=False), hr_png, csv_path, field_png, hr_png

# ------------------------ Gradio Blocks UI ----------------------------
with gr.Blocks(title="Photometry + H-R with ROI") as demo:
    gr.Markdown("# Stellar Photometry with ROI Selection")

    with gr.Row():
        std_box  = gr.Textbox(label="Standard Star", value="Capella")
        tgt_box  = gr.Textbox(label="Target / Cluster", value="M45")

    gr.Markdown("### 1. Upload FITS files")
    ext_up = gr.File(label="Extinction sequence (multi-select)", file_count="multiple")
    with gr.Row():
        bias_up = gr.File(label="Bias", file_count="multiple")
        dark_up = gr.File(label="Dark", file_count="multiple")
    with gr.Row():
        fb_up   = gr.File(label="Flat B", file_count="multiple")
        fv_up   = gr.File(label="Flat V", file_count="multiple")
    sci_up = gr.File(label="Science frames (B,V)", file_count="multiple")

    gr.Markdown("### 2. Select ROI (px)")
    roi_x = gr.Slider(label="X-range", minimum=0, maximum=2500, step=10, value=[0, 2262])
    roi_y = gr.Slider(label="Y-range", minimum=0, maximum=2000, step=10, value=[0, 1812])
    roi_preview = gr.Image(label="ROI Preview")

    def update_preview(files,xr,yr):
        return make_preview(files,xr,yr)

    sci_up.change(update_preview, [sci_up, roi_x, roi_y], roi_preview)
    roi_x.change(update_preview, [sci_up, roi_x, roi_y], roi_preview)
    roi_y.change(update_preview, [sci_up, roi_x, roi_y], roi_preview)

    run_btn = gr.Button("Run Pipeline ▶", variant="primary")

    gr.Markdown("### 3. Outputs")
    field_img = gr.Image(label="Star Field (ROI)")
    mag_html  = gr.HTML()
    hr_img    = gr.Image(label="H-R Diagram")

    with gr.Row():
        csv_dl   = gr.File(label="Download CSV")
        field_dl = gr.File(label="Download Field PNG")
        hr_dl    = gr.File(label="Download H-R PNG")

    run_btn.click(run_pipeline,
                  inputs=[ext_up,bias_up,dark_up,fb_up,fv_up,sci_up,std_box,tgt_box,roi_x,roi_y],
                  outputs=[field_img,mag_html,hr_img,csv_dl,field_dl,hr_dl])

if __name__ == '__main__':
    demo.launch()

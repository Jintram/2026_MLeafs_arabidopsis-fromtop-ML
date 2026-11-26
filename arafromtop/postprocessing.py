
"""Phase-4 postprocessing: plant ROIs, colour split and luminescence per plant.

Call `process_directory(config)` from a runner script; see
`pipelineclean_phase4_postprocessing.py`.
"""

################################################################################
# %% Libraries

import glob
import json
import os
import re
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from skimage.measure import regionprops

from arafromtop import calibration, luminescence, plotting
from arafromtop.color_classification import classify_rgb
from arafromtop.plant_rois import grid_mapping, segmentation_to_plant_rois

REPO_ROOT = Path(__file__).resolve().parents[1]
N_POSITIONS, N_COLS = 16, 4

################################################################################
# %% Configuration


@dataclass
class Phase4Config:
    # Input and output directories
    dir_seg: str
    dir_rgb: str
    dir_lum: str
    dir_output: str
    glob_pattern: str = "**/*_seg.npz"

    # Files in the repo (relative paths are resolved against the repo root)
    seed_mask_path: str = "plant_seed_mask_dilated.npz"
    transform_path: str = "calibration_data/tf_lum_to_rgb.npy"
    calib_lum_path: str = "calibration_data/Calibration16tray§004_TopESLive_2023-10-02T12-44-33.png"
    calib_rgb_path: str = "calibration_data/Calibration16tray§004_TopRGB_2023-10-02T12-44-33.png"

    # Plant ROIs (min_size_cluster is the undilated plant area of a cluster of
    # fragments that lie within dilation_radius of each other)
    label_class: int = 2
    dilation_radius: int = 10
    min_size_cluster: int = 600
    min_size_wholeplant: int = 400

    # Colour classification (swap the function to change the method)
    color_classifier: callable = classify_rgb
    color_params: dict = field(default_factory=dict)

    # Luminescence: cutoff = mode + max(n_sigma*sigma_below, n_sigma*min_sigma_frac*mode);
    # spot sizes in ES px; assign_radius in RGB px
    n_sigma: float = 5
    min_sigma_frac: float = 0.1
    min_spot_size: int = 1
    max_spot_size: int = 1000
    exclude_oversized: bool = False
    assign_radius: int = 15

    # Output
    gamma: float = 0.5
    save_npz: bool = True
    save_overview_pdf: bool = True
    save_plant_pdfs: bool = False
    dpi: int = 1200
    dpi_plant: int = 300

    def resolve(self, path):
        """Resolve a path relative to the repo root (absolute paths are kept)."""
        return str(path if os.path.isabs(path) else REPO_ROOT / path)

    def lum_params(self):
        return {k: getattr(self, k) for k in ["n_sigma", "min_sigma_frac", "min_spot_size",
                                              "max_spot_size", "exclude_oversized",
                                              "assign_radius"]}

################################################################################
# %% Files and names


def parse_sample_name(filename):
    """Extract dpi, batch, tray, replicate and timestamp from an image filename (NaN if absent)."""
    
    # TO DO: check this function
    
    patterns = {"dpi": r"(\d+)\s*_?dpi",
                "batch": r"(?:^|[_\s])(C\d+)(?=[_\s])",
                "tray": r"tray\s*(\d+)",
                "replicate": r"§(\d+)",
                "timestamp": r"_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})"}
    out = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, os.path.basename(filename), flags=re.IGNORECASE)
        out[key] = m.group(1) if m else np.nan
    for key in ["dpi", "tray", "replicate"]:
        out[key] = int(out[key]) if isinstance(out[key], str) else np.nan
    return out


def find_lum_file(rgb_filename, dir_lum):
    """Return the path of the ES image matching an RGB image (TopRGB -> TopES), or None."""

    # Search/replace to get corresponding ES file
    lum_name = os.path.basename(rgb_filename).replace("TopRGB", "TopES")

    # If direct corresponding file exists, use it
    direct = os.path.join(dir_lum, lum_name)
    if os.path.exists(direct):
        return direct
    
    # otherwise look for it    
    hits = glob.glob(os.path.join(glob.escape(dir_lum), "**", glob.escape(lum_name)), recursive=True)

    # if >1 hits, throw warning
    if len(hits) > 1:
        print(f"Warning: multiple ES files found (using 1st hit) for {rgb_filename}: {hits}")
    
    return hits[0] if hits else None


def get_transform(config):
    """Load the ES->RGB transform; fit and save it from the calibration images if missing."""
    
    path_tf = config.resolve(config.transform_path)
    
    if os.path.exists(path_tf):
        return calibration.load_transform(path_tf)

    # Fit, save, and save a visual check next to it
    img_lum = iio.imread(config.resolve(config.calib_lum_path))
    img_rgb = iio.imread(config.resolve(config.calib_rgb_path))
    tf, summary = calibration.fit_lum_to_rgb(img_lum, img_rgb)
    print(f"Calibration residuals (RGB px): {summary['residuals_rgb_px']}")
    calibration.save_transform(tf, summary, path_tf)
    calibration.plot_calibration_check(img_lum, img_rgb, tf, summary,
                                       path_tf.replace(".npy", "_check.pdf"))
    return calibration.load_transform(path_tf)

################################################################################
# %% Per-plant table


def plant_table(image_name, labelmask, color_res, lum_res):
    """One row per tray position (always 16), with area, colour and luminescence stats."""
    meta = parse_sample_name(image_name)
    props = {p.label: p for p in regionprops(labelmask)}
    n_lbl = N_POSITIONS + 1
    green_px = np.bincount(labelmask[color_res["green_mask"]], minlength=n_lbl)
    yellow_px = np.bincount(labelmask[color_res["yellow_mask"]], minlength=n_lbl)

    rows = []
    for lbl in range(1, N_POSITIONS + 1):
        # Identification and plant/colour areas (0 for empty positions)
        prop = props.get(lbl)
        area = int(prop.area) if prop else 0
        row = {"image": image_name, **meta, "position": grid_mapping(lbl, N_COLS),
               "row": (lbl - 1) // N_COLS, "col": (lbl - 1) % N_COLS, "label": lbl,
               "plant_present": prop is not None, "plant_area_px": area,
               "green_area_px": int(green_px[lbl]), "yellow_area_px": int(yellow_px[lbl]),
               "unclassified_area_px": area - int(green_px[lbl]) - int(yellow_px[lbl]),
               "yellow_pct": 100 * yellow_px[lbl] / area if area else np.nan,
               "centroid_x": prop.centroid[1] if prop else np.nan,
               "centroid_y": prop.centroid[0] if prop else np.nan}

        # Luminescence stats (NaN when the plant is absent or there is no ES image)
        if lum_res is not None and prop is not None:
            row.update(luminescence.plant_lum_stats(
                lum_res["lum"], lum_res["background"], lum_res["spot_lbl_es"],
                lum_res["spots"], lum_res["spot_to_plant"], lum_res["plant_lbl_es"],
                lbl, area))
        rows.append(row)
    return pd.DataFrame(rows)

################################################################################
# %% Single file


def process_single_file(segfile, config, seeds, tf):
    """Process one segmentation file; return (plants_df, image_row)."""

    # Paths: segfile is relative to dir_seg, e.g. "images/<name>.png_seg.npz"
    rel_rgb = segfile.replace("_seg.npz", ".png")
    image_name = os.path.basename(rel_rgb)
    path_rgb = os.path.join(config.dir_rgb, rel_rgb)
    path_lum = find_lum_file(image_name, config.dir_lum)
    out_base = os.path.join(config.dir_output, re.sub(r"\.png$", "", rel_rgb))
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    # Load segmentation (only the labels key; prepr_info is an object array) and image
    segmask = np.load(os.path.join(config.dir_seg, segfile))["img_pred_lbls"]
    img_rgb = iio.imread(path_rgb)[..., :3]
    img_bright = (np.power(img_rgb / 255.0, config.gamma) * 255).astype(np.uint8)

    # Plant ROIs and colour classification
    labelmask = segmentation_to_plant_rois(segmask, seeds, config.label_class,
                                           config.dilation_radius, config.min_size_cluster,
                                           config.min_size_wholeplant)
    color_res = config.color_classifier(img_rgb, labelmask > 0, config.color_params)

    # Luminescence (if an ES image is available)
    lum_res, lum_disp = None, None
    if path_lum is not None:
        lum_res = luminescence.analyse_luminescence(path_lum, labelmask, tf, config.lum_params())
        lum_sub = np.clip(lum_res["lum"] - lum_res["background"], 0, None)
        lum_disp = calibration.warp_intensity_to_rgb(lum_sub, tf, labelmask.shape)

    # Tables
    plants_df = plant_table(image_name, labelmask, color_res, lum_res)
    image_row = {"image": image_name, **parse_sample_name(image_name),
                 "path_rgb": path_rgb, "path_lum": path_lum,
                 "n_plants": int(plants_df["plant_present"].sum()),
                 **(lum_res["image_stats"] if lum_res is not None else {"lum_available": False}),
                 "status": "ok", "error": ""}

    # Save masks
    if config.save_npz:
        extra = {} if lum_res is None else {"labelmask_expanded": lum_res["labelmask_expanded"],
                                            "spot_labels_rgb": lum_res["spot_lbl_rgb"]}
        np.savez_compressed(out_base + "_proc.npz", labelmask_cleaned=labelmask,
                            yellow_mask=color_res["yellow_mask"], **extra)

    # Plots
    if config.save_overview_pdf:
        plotting.plot_image_overview(image_name, img_bright, labelmask, color_res["yellow_mask"],
                                     lum_res, lum_disp, out_base + "_overview.pdf", config.dpi)
    if config.save_plant_pdfs:
        dir_plants = os.path.join(os.path.dirname(out_base), "per_plant", os.path.basename(out_base))
        os.makedirs(dir_plants, exist_ok=True)
        for _, plant_row in plants_df[plants_df["plant_present"]].iterrows():
            plotting.plot_plant_detail(image_name, plant_row["label"], img_bright, img_rgb,
                                       labelmask, color_res, lum_res, lum_disp, plant_row,
                                       os.path.join(dir_plants, f"{plant_row['position']}.pdf"),
                                       dpi=config.dpi_plant)

    return plants_df, image_row

################################################################################
# %% Batch


def list_segfiles(config):
    """All segmentation files under dir_seg, relative to it, sorted."""
    return sorted(glob.glob(config.glob_pattern, root_dir=config.dir_seg, recursive=True))


def save_config(config, path_json):
    """Save the config used (function names instead of functions) as JSON."""
    cfg = asdict(config)
    cfg["color_classifier"] = config.color_classifier.__name__
    with open(path_json, "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def process_directory(config, max_files=None):
    """Process all segmentation files; write results_plants.csv and results_images.csv.

    Errors in a file are recorded in results_images.csv (status/error) and the
    batch continues. Returns (plants_df, images_df).
    """
    os.makedirs(config.dir_output, exist_ok=True)
    seeds = np.load(config.resolve(config.seed_mask_path))["mask_seeds_dilated"]
    tf = get_transform(config)
    segfiles = list_segfiles(config)[:max_files]

    # Process file by file
    all_plants, all_images = [], []
    for idx, segfile in enumerate(segfiles):
        print(f"Processing file {idx + 1}/{len(segfiles)}: {segfile}")
        try:
            plants_df, image_row = process_single_file(segfile, config, seeds, tf)
            all_plants.append(plants_df)
        except Exception as e:
            traceback.print_exc()
            image_row = {"image": os.path.basename(segfile), "status": "error", "error": repr(e)}
        all_images.append(image_row)

    # Combined tables and the config used
    plants_df = pd.concat(all_plants, ignore_index=True) if all_plants else pd.DataFrame()
    images_df = pd.DataFrame(all_images)
    plants_df.to_csv(os.path.join(config.dir_output, "results_plants.csv"), index=False)
    images_df.to_csv(os.path.join(config.dir_output, "results_images.csv"), index=False)
    save_config(config, os.path.join(config.dir_output, "config_used.json"))
    return plants_df, images_df

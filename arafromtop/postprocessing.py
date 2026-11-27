
"""Phase-4 postprocessing: plant ROIs, colour split and luminescence per plant.

Call `process_directory(config4)` from a runner script; see
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

import matplotlib.pyplot as plt

import imageio.v3 as iio
import numpy as np
import pandas as pd

import skimage as sk

import arafromtop.calibration as ara_cal
import arafromtop.color_classification as ara_col
import arafromtop.luminescence as ara_lum
import arafromtop.plant_rois as ara_plant
import arafromtop.plotting as ara_plot

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
    seed_mask_path: str = "calibration_data/seedmask_positions.png"
    transform_path: str = "calibration_data/tf_lum_to_rgb.npy"
    calib_lum_path: str = "calibration_data/Calibration16tray§004_TopESLive_2023-10-02T12-44-33.png"
    calib_rgb_path: str = "calibration_data/Calibration16tray§004_TopRGB_2023-10-02T12-44-33.png"

    # Plant ROIs 
    plant_classlabel: int = 2 # segmentation label that is plant
    dilation_radius: int = 10 # radius used to connect plants
    min_size_cluster: int = 600 # XXX
    min_size_wholeplant: int = 400 # XXX

    # Colour classification (swap the function to change the method)
    color_classifier: callable = ara_col.classify_rgb
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
        path_resolved = str(path if os.path.isabs(path) else REPO_ROOT / path)
        return path_resolved

    def lum_params(self):
        keys = ["n_sigma", "min_sigma_frac", "min_spot_size", "max_spot_size",
                "exclude_oversized", "assign_radius"]
        params = {k: getattr(self, k) for k in keys}
        return params

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


def get_transform(config4):
    """Load the ES->RGB transform; fit and save it from the calibration images if missing."""

    # The result (matrix_lum_to_rgb elsewhere) is a skimage ProjectiveTransform: the
    # 3x3 matrix (.params) plus methods such as .inverse
    
    path_matrix_tf = config4.resolve(config4.transform_path)
    
    # Fit, save, and save a visual check next to it (only if not saved before)
    if not os.path.exists(path_matrix_tf):
        img_lum = iio.imread(config4.resolve(config4.calib_lum_path))
        img_rgb = iio.imread(config4.resolve(config4.calib_rgb_path))
        matrix_lum_to_rgb, summary = ara_cal.fit_lum_to_rgb(img_lum, img_rgb)
        print(f"Calibration residuals (RGB px): {summary['residuals_rgb_px']}")
        ara_cal.save_transform(matrix_lum_to_rgb, summary, path_matrix_tf)
        ara_cal.plot_calibration_check(img_lum, img_rgb, matrix_lum_to_rgb, summary,
                                       path_matrix_tf.replace(".npy", "_check.pdf"))

    # Load from file (also after fitting, so both cases give the same object)
    matrix_lum_to_rgb = ara_cal.load_transform(path_matrix_tf)
    return matrix_lum_to_rgb

################################################################################
# %% Per-plant table


def plant_table(image_name, labelmask, color_res, lum_res):
    """One row per tray position (always 16), with area, colour and luminescence stats."""
    meta = parse_sample_name(image_name)
    props = {p.label: p for p in sk.measure.regionprops(labelmask)}
    n_lbl = N_POSITIONS + 1
    green_px = np.bincount(labelmask[color_res["green_mask"]], minlength=n_lbl)
    yellow_px = np.bincount(labelmask[color_res["yellow_mask"]], minlength=n_lbl)

    rows = []
    for lbl in range(1, N_POSITIONS + 1):
        # Identification and plant/colour areas (0 for empty positions)
        prop = props.get(lbl)
        area = int(prop.area) if prop else 0
        row = {"image": image_name, **meta, "position": ara_plant.grid_mapping(lbl, N_COLS),
               "row": (lbl - 1) // N_COLS, "col": (lbl - 1) % N_COLS, "label": lbl,
               "plant_present": prop is not None, "plant_area_px": area,
               "green_area_px": int(green_px[lbl]), "yellow_area_px": int(yellow_px[lbl]),
               "unclassified_area_px": area - int(green_px[lbl]) - int(yellow_px[lbl]),
               "yellow_pct": 100 * yellow_px[lbl] / area if area else np.nan,
               "centroid_x": prop.centroid[1] if prop else np.nan,
               "centroid_y": prop.centroid[0] if prop else np.nan}

        # Luminescence stats (NaN when the plant is absent or there is no ES image)
        if lum_res is not None and prop is not None:
            row.update(ara_lum.plant_lum_stats(
                lum_res["lum"], lum_res["background"], lum_res["spot_lbl_es"],
                lum_res["spots"], lum_res["spot_to_plant"], lum_res["plant_lbl_es"],
                lbl, area))
        rows.append(row)

    plants_df = pd.DataFrame(rows)
    return plants_df

################################################################################
# %% Single file


def process_single_file(relpath_segfile, config4, seedmask_positions, matrix_lum_to_rgb):
    """Process one segmentation file; return (plants_df, image_row)."""

    # Get relevant relative and absolute paths
    rel_rgb = relpath_segfile.replace("_seg.npz", ".png")
    image_name = os.path.basename(rel_rgb)
    path_rgb = os.path.join(config4.dir_rgb, rel_rgb)
    path_lum = find_lum_file(image_name, config4.dir_lum)
    out_base = os.path.join(config4.dir_output, re.sub(r"\.png$", "", rel_rgb))

    # Make output dir
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    # Load segmentation mask and rgb image
    segmask = np.load(os.path.join(config4.dir_seg, relpath_segfile))["img_pred_lbls"]
    img_rgb = iio.imread(path_rgb)[..., :3]
    img_bright = (np.power(img_rgb / 255.0, config4.gamma) * 255).astype(np.uint8)
    # plt.imshow(img_bright)

    # Plant ROIs and colour classification
    labelmask = ara_plant.segmentation_to_plant_rois(
        segmask, seedmask_positions, 
        plant_classlabel = config4.plant_classlabel, 
        dilation_radius=config4.dilation_radius,
        min_size_cluster=config4.min_size_cluster, 
        min_size_wholeplant=config4.min_size_wholeplant)
        # plt.imshow(labelmask)
        
    # XXXXXX CONTINUE GOING THROUGH CODE HERE <<<<<<<<<<<<<<<
        
    color_res = config4.color_classifier(img_rgb, labelmask > 0, config4.color_params)

    # Luminescence (if an ES image is available)
    lum_res, lum_disp = None, None
    if path_lum is not None:
        lum_res = ara_lum.analyse_luminescence(path_lum, labelmask, matrix_lum_to_rgb,
                                               config4.lum_params())
        lum_sub = np.clip(lum_res["lum"] - lum_res["background"], 0, None)
        lum_disp = ara_cal.warp_intensity_to_rgb(lum_sub, matrix_lum_to_rgb, labelmask.shape)

    # Tables
    plants_df = plant_table(image_name, labelmask, color_res, lum_res)
    image_row = {"image": image_name, **parse_sample_name(image_name),
                 "path_rgb": path_rgb, "path_lum": path_lum,
                 "n_plants": int(plants_df["plant_present"].sum()),
                 **(lum_res["image_stats"] if lum_res is not None else {"lum_available": False}),
                 "status": "ok", "error": ""}

    # Save masks
    if config4.save_npz:
        extra = {} if lum_res is None else {"labelmask_expanded": lum_res["labelmask_expanded"],
                                            "spot_labels_rgb": lum_res["spot_lbl_rgb"]}
        np.savez_compressed(out_base + "_proc.npz", labelmask_cleaned=labelmask,
                            yellow_mask=color_res["yellow_mask"], **extra)

    # Plots
    if config4.save_overview_pdf:
        ara_plot.plot_image_overview(image_name, img_bright, labelmask, color_res["yellow_mask"],
                                     lum_res, lum_disp, out_base + "_overview.pdf", config4.dpi)
    if config4.save_plant_pdfs:
        dir_plants = os.path.join(os.path.dirname(out_base), "per_plant", os.path.basename(out_base))
        os.makedirs(dir_plants, exist_ok=True)
        for _, plant_row in plants_df[plants_df["plant_present"]].iterrows():
            ara_plot.plot_plant_detail(image_name, plant_row["label"], img_bright, img_rgb,
                                       labelmask, color_res, lum_res, lum_disp, plant_row,
                                       os.path.join(dir_plants, f"{plant_row['position']}.pdf"),
                                       dpi=config4.dpi_plant)

    return plants_df, image_row

################################################################################
# %% Batch


def get_list_segfiles(config4):
    """All segmentation files under dir_seg, relative to it, sorted."""

    # get all sample seg files
    list_segfiles = sorted(glob.glob(config4.glob_pattern, root_dir=config4.dir_seg, recursive=True))
    
    return list_segfiles


def save_config(config4, path_json):
    """Save the config used (function names instead of functions) as JSON."""

    # convert to dict
    cfg_dict = asdict(config4)

    # replace function by its name
    cfg_dict["color_classifier"] = config4.color_classifier.__name__

    # save it as JSON
    with open(path_json, "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)


# %%

def process_directory(config4, max_files=None):
    """Process all segmentation files; write results_plants.csv and results_images.csv.

    Errors in a file are recorded in results_images.csv (status/error) and the
    batch continues. Returns (plants_df, images_df).
    """

    # Admin: make dirs and load seed/image transformation file
    os.makedirs(config4.dir_output, exist_ok=True)
    seedmask_positions = ara_plant.load_seedmask_positions(
        config4.resolve(config4.seed_mask_path))
    matrix_lum_to_rgb = get_transform(config4)

    # Get list of samples that were segmented
    list_segfiles = get_list_segfiles(config4)[:max_files]

    # Process file by file
    all_plants, all_images = [], []
    for idx, relpath_segfile in enumerate(list_segfiles):
        # idx, relpath_segfile = 0, list_segfiles[0]  # debug
        
        print(f"Processing file {idx + 1}/{len(list_segfiles)}: {relpath_segfile}")

        # process sample, try/except to keep loop going in case failure
        try:
                        
            # process sample, collect per-plant and per-image data
            plants_df, image_row = process_single_file(relpath_segfile, config4, seedmask_positions,
                                                       matrix_lum_to_rgb)

            # track results
            all_plants.append(plants_df)            
            all_images.append(image_row)
            
        except Exception as e:
            
            # put full traceback on screen
            traceback.print_exc()
            
            # set the result to error summary
            image_row = {"image": os.path.basename(relpath_segfile), "status": "error",
                         "error": repr(e)}
            all_images.append(image_row)        

    # Create and save combined tables 
    plants_df = pd.concat(all_plants, ignore_index=True) if all_plants else pd.DataFrame()
    images_df = pd.DataFrame(all_images)
    plants_df.to_csv(os.path.join(config4.dir_output, "results_plants.csv"), index=False)
    images_df.to_csv(os.path.join(config4.dir_output, "results_images.csv"), index=False)
    
    # Save the config used
    save_config(config4, os.path.join(config4.dir_output, "config_used.json"))
    
    return plants_df, images_df

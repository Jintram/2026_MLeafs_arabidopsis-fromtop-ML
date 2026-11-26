
"""Luminescence (ES camera) spot detection and per-plant statistics.

Spots are detected and intensities are measured in the native ES frame (raw data
are never resampled); only label images are warped between frames (nearest
neighbour). Spot areas are reported in RGB pixels, comparable with plant areas.
"""

################################################################################
# %% Libraries

import warnings

import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndi

from skimage.measure import label
from skimage.segmentation import expand_labels, relabel_sequential

from arafromtop import calibration

################################################################################
# %% Loading and background


def load_lum(path):
    """Load an ES image as a 2-D array; return (lum, channels_differ).

    The ES PNGs store grey copied over R=G=B (plus alpha), so channel 0 is used.
    If the channels ever differ, their mean is used instead and a warning raised.
    """
    img = iio.imread(path)
    if img.ndim == 2:
        return img.astype(np.int32), False

    # Check whether R, G, B are identical
    rgb = img[..., :3]
    channels_differ = not (np.array_equal(rgb[..., 0], rgb[..., 1])
                           and np.array_equal(rgb[..., 0], rgb[..., 2]))
    if channels_differ:
        warnings.warn(f"R, G, B differ in ES image {path}; using their mean.")
        return np.rint(rgb.mean(axis=2)).astype(np.int32), True
    return rgb[..., 0].astype(np.int32), False


def estimate_background(lum):
    """Background = the mode of the image (most pixels are background)."""
    return int(np.bincount(lum.ravel()).argmax())


def estimate_cutoff(lum, mode, n_sigma=5, min_sigma_frac=0.1):
    """Return (cutoff, sigma_below); signal is lum > cutoff.

    sigma_below is estimated from the pixels below the mode, assuming the
    background distribution is symmetric around the mode. Its minimum is
    `min_sigma_frac` * mode (relevant when the background is clipped at a floor).
    """

    # Spread of the lower half, mirrored
    below = lum[lum < mode].astype(float)
    sigma_below = float(np.sqrt(np.mean((below - mode) ** 2))) if below.size else 0.0

    # Cutoff, with the minimum sigma as fallback
    cutoff = mode + max(n_sigma * sigma_below, n_sigma * min_sigma_frac * mode)
    return float(cutoff), sigma_below

################################################################################
# %% Spot detection and assignment


def detect_spots(lum, cutoff, min_spot_size=2, max_spot_size=500, exclude_oversized=False):
    """Label the spots (connected pixels > cutoff) in the ES frame; return (spot_lbl, oversized).

    Spots smaller than `min_spot_size` ES px are removed. Spots larger than
    `max_spot_size` ES px set the `oversized` flag, and are removed if
    `exclude_oversized`.
    """

    # Threshold and label connected pixels
    spot_lbl = label(lum > cutoff)
    sizes = np.bincount(spot_lbl.ravel())
    sizes[0] = 0

    # Remove small spots, flag (and optionally remove) large ones
    to_remove = sizes < min_spot_size
    oversized = bool(np.any(sizes > max_spot_size))
    if exclude_oversized:
        to_remove |= sizes > max_spot_size
    to_remove[0] = False
    spot_lbl[to_remove[spot_lbl]] = 0

    return relabel_sequential(spot_lbl)[0], oversized


def expand_plant_rois(labelmask, assign_radius=15):
    """Grow each plant ROI into background by `assign_radius` RGB px.

    This absorbs the slight misalignment between ES and RGB. Original plant
    pixels are kept, and gaps between neighbouring plants are split by the
    nearest plant edge, so neighbours don't overlap.
    """
    return expand_labels(labelmask, distance=assign_radius)


def assign_spots_to_plants(spot_lbl_es, plant_lbl_es):
    """Return an array mapping spot id -> plant label (0 = unassigned).

    Each spot goes to the (expanded) plant ROI it overlaps most.
    """
    n_spots = int(spot_lbl_es.max())
    spot_to_plant = np.zeros(n_spots + 1, dtype=int)

    # Count overlapping pixels per (spot, plant) pair
    overlap = (spot_lbl_es > 0) & (plant_lbl_es > 0)
    pairs, counts = np.unique(np.stack([spot_lbl_es[overlap], plant_lbl_es[overlap]]),
                              axis=1, return_counts=True)

    # For each spot, keep the plant with the largest overlap (sort by count, so
    # the last assignment per spot is the largest)
    order = np.argsort(counts)
    spot_to_plant[pairs[0, order]] = pairs[1, order]
    return spot_to_plant

################################################################################
# %% Statistics


def spot_table(lum, background, spot_lbl_es, spot_lbl_rgb):
    """Per-spot area (RGB px) and median/mean of background-subtracted intensity."""
    n_spots = int(spot_lbl_es.max())
    idx = np.arange(1, n_spots + 1)
    lum_sub = np.clip(lum - background, 0, None)

    area_rgb = np.bincount(spot_lbl_rgb.ravel(), minlength=n_spots + 1)[1:]
    if n_spots == 0:
        return {"area_px": area_rgb, "median": np.array([]), "mean": np.array([])}
    return {"area_px": area_rgb,
            "median": np.asarray(ndi.median(lum_sub, spot_lbl_es, idx)),
            "mean": np.asarray(ndi.mean(lum_sub, spot_lbl_es, idx))}


def plant_lum_stats(lum, background, spot_lbl_es, spots, spot_to_plant, plant_lbl_es,
                    plant_label, plant_area_px):
    """Luminescence statistics for one plant (see the plan / readme for definitions)."""

    # Spots of this plant (spot ids start at 1, arrays in `spots` at 0)
    spot_ids = np.flatnonzero(spot_to_plant == plant_label)
    spot_ids = spot_ids[spot_ids > 0]
    areas = spots["area_px"][spot_ids - 1]
    medians = spots["median"][spot_ids - 1]
    means = spots["mean"][spot_ids - 1]

    # Pixel values within the plant's spots, and within its expanded ROI
    lum_sub = np.clip(lum - background, 0, None)
    spot_pixels = lum_sub[np.isin(spot_lbl_es, spot_ids)] if spot_ids.size else np.array([])
    roi = plant_lbl_es == plant_label
    has_spots = spot_ids.size > 0

    spot_area = int(areas.sum())
    return {
        "nr_spots": int(spot_ids.size),
        "spot_area_px": spot_area,
        "spot_area_pct": 100 * spot_area / plant_area_px if plant_area_px > 0 else np.nan,
        "spot_median_brightest": float(medians.max()) if has_spots else np.nan,
        "spot_median_dimmest": float(medians.min()) if has_spots else np.nan,
        "spot_mean_brightest": float(means.max()) if has_spots else np.nan,
        "spot_mean_dimmest": float(means.min()) if has_spots else np.nan,
        "spot_median_all": float(np.median(spot_pixels)) if has_spots else np.nan,
        "spot_mean_all": float(np.mean(spot_pixels)) if has_spots else np.nan,
        "largest_spot_area_px": int(areas.max()) if has_spots else 0,
        "smallest_spot_area_px": int(areas.min()) if has_spots else 0,
        "lum_max": int(lum[roi].max()) if roi.any() else np.nan,
        "lum_integrated": float(lum_sub[roi].sum()) if roi.any() else np.nan,
    }

################################################################################
# %% Whole-image pipeline


def analyse_luminescence(path_lum, labelmask, tf, params):
    """Detect spots and assign them to plants; return a dict with all intermediate results.

    `labelmask` is the plant ROI mask in the RGB frame; `params` holds n_sigma,
    min_sigma_frac, min_spot_size, max_spot_size, exclude_oversized, assign_radius.
    """

    # Load, background and cutoff
    lum, channels_differ = load_lum(path_lum)
    background = estimate_background(lum)
    cutoff, sigma_below = estimate_cutoff(lum, background, params["n_sigma"],
                                          params["min_sigma_frac"])

    # Spots in the ES frame, and their labels in the RGB frame (for areas, plots)
    spot_lbl_es, oversized = detect_spots(lum, cutoff, params["min_spot_size"],
                                          params["max_spot_size"], params["exclude_oversized"])
    spot_lbl_rgb = calibration.warp_labels_to_rgb(spot_lbl_es, tf, labelmask.shape)

    # Expanded plant ROIs (RGB frame), warped to the ES frame, and spot assignment
    labelmask_expanded = expand_plant_rois(labelmask, params["assign_radius"])
    plant_lbl_es = calibration.warp_labels_to_es(labelmask_expanded, tf, lum.shape)
    spot_to_plant = assign_spots_to_plants(spot_lbl_es, plant_lbl_es)

    # Image-level values
    image_stats = {
        "lum_available": True,
        "lum_mode": background,
        "lum_sigma_below": sigma_below,
        "lum_cutoff": cutoff,
        "lum_global_mean": float(lum[lum > lum.min()].mean()) if (lum > lum.min()).any() else 0.0,
        "lum_n_saturated_px": int((lum >= 255).sum()),
        "n_spots_total": int(spot_lbl_es.max()),
        "n_spots_unassigned": int((spot_to_plant[1:] == 0).sum()),
        "warning_oversized_spot": oversized,
        "warning_channels_differ": channels_differ,
    }
    return {"lum": lum, "background": background, "cutoff": cutoff,
            "spot_lbl_es": spot_lbl_es, "spot_lbl_rgb": spot_lbl_rgb,
            "labelmask_expanded": labelmask_expanded, "plant_lbl_es": plant_lbl_es,
            "spot_to_plant": spot_to_plant,
            "spots": spot_table(lum, background, spot_lbl_es, spot_lbl_rgb),
            "image_stats": image_stats}

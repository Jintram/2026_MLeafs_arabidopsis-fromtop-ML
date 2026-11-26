
"""Classification of plant pixels into green (healthy) and yellow (chlorotic).

A classifier is a function `classifier(img_rgb, plant_mask, params)` that returns
a dict with `green_mask`, `yellow_mask` and `debug`; `debug` holds `features`
(name -> full-image float array, NaN outside the plant mask) and `cutoffs`
(name -> list of threshold values), used for the histograms in the plots.
Swap the classifier via the config without changing anything else.
"""

################################################################################
# %% Libraries

import numpy as np

from scipy.ndimage import binary_closing, binary_opening

################################################################################
# %% Default parameters

# Converted from the old Digital_phenotyper HSV settings (OpenCV units):
# YELLOW H 26-35, S >= 85, V >= 25; GREEN H 36-70, S >= 50, V >= 25.
# For G >= R, hue = 120 deg - 60 deg * rg_ratio, so the yellow/green border at
# H ~71 deg is rg_ratio ~0.82, and the yellow/orange border at 52 deg is ~1.15.
PARAMS_RGB_DEFAULT = {
    "yellow_rg_ratio": (0.82, 1.15),
    "yellow_min_chroma": 0.33,
    "green_max_rg_ratio": 0.82,
    "green_min_chroma": 0.20,
    "min_brightness": 25,
    # Side of the square used to open/close the yellow mask (0 = off). The old
    # tool used 10, but on a blurred image; on unblurred pixels 10 removes ~all
    # yellow, 3 only removes isolated pixels.
    "morph_size": 3,
}

################################################################################
# %% RGB-based classifier


def rgb_features(img_rgb, plant_mask):
    """Compute rg_ratio, chroma and brightness for the plant pixels (NaN elsewhere)."""

    # Channels as float, restricted to plant pixels
    R, G, B = [img_rgb[..., i].astype(np.float32) for i in range(3)]
    cmax = np.maximum(np.maximum(R, G), B)
    cmin = np.minimum(np.minimum(R, G), B)

    # rg_ratio = (R-B)/(G-B): ~0 pure green, ~1 pure yellow; undefined (inf)
    # when G <= B (blue-ish or grey pixels, neither green nor yellow)
    with np.errstate(divide="ignore", invalid="ignore"):
        rg_ratio = np.where(G > B, (R - B) / (G - B), np.inf)
        chroma = np.where(cmax > 0, (cmax - cmin) / cmax, 0)

    # Blank out non-plant pixels
    features = {"rg_ratio": rg_ratio, "chroma": chroma, "brightness": cmax}
    for name in features:
        features[name] = np.where(plant_mask, features[name], np.nan).astype(np.float32)
    return features


def classify_rgb(img_rgb, plant_mask, params=None):
    """Split plant pixels into green / yellow using RGB-derived cutoffs."""
    p = {**PARAMS_RGB_DEFAULT, **(params or {})}
    f = rgb_features(img_rgb, plant_mask)

    # Raw rules (comparisons with NaN are False, so non-plant pixels drop out)
    bright_enough = f["brightness"] >= p["min_brightness"]
    yellow = (bright_enough & (f["chroma"] >= p["yellow_min_chroma"])
              & (f["rg_ratio"] >= p["yellow_rg_ratio"][0])
              & (f["rg_ratio"] <= p["yellow_rg_ratio"][1]))
    green = (bright_enough & (f["chroma"] >= p["green_min_chroma"])
             & (f["rg_ratio"] < p["green_max_rg_ratio"]))

    # Smooth the yellow mask (open, then close) as in the old tool, stay within
    # the plant, and keep green and yellow disjoint
    if p["morph_size"] > 0:
        footprint = np.ones((p["morph_size"], p["morph_size"]), dtype=bool)
        yellow = binary_closing(binary_opening(yellow, footprint), footprint) & plant_mask
    green = green & ~yellow

    cutoffs = {"rg_ratio": [p["green_max_rg_ratio"], *p["yellow_rg_ratio"]],
               "chroma": [p["green_min_chroma"], p["yellow_min_chroma"]],
               "brightness": [p["min_brightness"]]}
    return {"green_mask": green, "yellow_mask": yellow,
            "debug": {"features": f, "cutoffs": cutoffs}}

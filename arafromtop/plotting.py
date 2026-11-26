
"""Plots for phase-4 postprocessing: per-image overview and per-plant detail."""

################################################################################
# %% Libraries

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from skimage.measure import regionprops

from arafromtop.plant_rois import grid_mapping

################################################################################
# %% Style

COLOR_PLANT_CONTOUR = "white"
COLOR_EXPANDED_ROI = "deepskyblue"   # expanded ROI used for luminescence assignment
COLOR_SPOTS = "red"
COLOR_YELLOW = (1.0, 0.85, 0.0)
FIG_RC = {"font.size": 6, "font.family": "Arial"}
TEXT_BBOX = dict(facecolor="black", edgecolor="none", boxstyle="square,pad=0.2", alpha=.5)


def cmap_positions(n_positions=16, min_brightness=0.6, seed=0):
    """Colormap with one fixed colour per tray position (index 0 = black)."""
    rng = np.random.default_rng(seed)
    colors = np.zeros((n_positions + 1, 3))
    colors[1:] = rng.random((n_positions, 3)) * (1 - min_brightness) + min_brightness
    return ListedColormap(colors)

################################################################################
# %% Drawing helpers


def contour_labels(ax, labelmask, color, linewidth):
    """Draw the outline of each label; contours are computed per bounding box (fast)."""
    for prop in regionprops(labelmask):
        r0, c0, r1, c1 = prop.bbox
        r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
        r1, c1 = min(r1 + 1, labelmask.shape[0]), min(c1 + 1, labelmask.shape[1])
        sub = labelmask[r0:r1, c0:c1] == prop.label
        ax.contour(np.arange(c0, c1), np.arange(r0, r1), sub, levels=[0.5],
                   colors=color, linewidths=linewidth)


def overlay_mask(ax, mask, color, alpha=0.8):
    """Show a boolean mask as a coloured, semi-transparent layer."""
    rgba = np.zeros(mask.shape + (4,))
    rgba[mask] = (*color, alpha)
    ax.imshow(rgba, interpolation="nearest")


def show_lum(ax, lum_disp, vmax=None):
    """Show background-subtracted luminescence (RGB frame); background is black."""
    vmax = vmax or max(np.percentile(lum_disp[lum_disp > 0], 99.9) if (lum_disp > 0).any() else 1, 1)
    ax.imshow(lum_disp, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")


def label_positions(ax, labelmask, cmap):
    """Write the tray position (A1..D4) at the top-left of each plant."""
    for prop in regionprops(labelmask):
        ytop, xleft = prop.bbox[:2]
        ax.text(xleft, ytop, grid_mapping(prop.label), color=cmap(prop.label), fontsize=4,
                ha="center", va="center", bbox=TEXT_BBOX)

################################################################################
# %% Per-image overview


def plot_image_overview(title, img_bright, labelmask, yellow_mask, lum_res, lum_disp,
                        path_pdf, dpi=1200):
    """Save a 3-panel overview: RGB + ROIs, label mask, luminescence on grey RGB."""
    cmap = cmap_positions()

    with plt.rc_context(FIG_RC):
        fig, axs = plt.subplots(1, 3, figsize=(18 / 2.54, 8.5 / 2.54))
        fig.suptitle(title)

        # Panel 1: RGB with plant contours, yellow tissue, expanded ROIs and labels
        axs[0].imshow(img_bright)
        overlay_mask(axs[0], yellow_mask, COLOR_YELLOW)
        contour_labels(axs[0], labelmask, COLOR_PLANT_CONTOUR, 0.05)
        if lum_res is not None:
            contour_labels(axs[0], lum_res["labelmask_expanded"], COLOR_EXPANDED_ROI, 0.05)
        label_positions(axs[0], labelmask, cmap)

        # Panel 2: label mask, colours fixed per tray position
        axs[1].imshow(labelmask, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
        label_positions(axs[1], labelmask, cmap)

        # Panel 3: luminescence (warped to the RGB frame) with the ROI contours
        if lum_res is not None:
            show_lum(axs[2], lum_disp)
            contour_labels(axs[2], labelmask, COLOR_PLANT_CONTOUR, 0.05)
            contour_labels(axs[2], lum_res["labelmask_expanded"], COLOR_EXPANDED_ROI, 0.05)
            axs[2].contour(lum_res["spot_lbl_rgb"] > 0, levels=[0.5], colors=COLOR_SPOTS,
                           linewidths=0.05)
            fig.suptitle(f"{title}\nluminescence cutoff > {lum_res['cutoff']:.1f}, "
                         f"unassigned spots: {lum_res['image_stats']['n_spots_unassigned']}")
        else:
            axs[2].set_title("no ES image")

        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path_pdf, dpi=dpi)
        plt.close(fig)

################################################################################
# %% Per-plant detail


def crop_slices(mask, margin, shape):
    """Bounding-box slices around a boolean mask, with a margin."""
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin + 1, shape[1])
    return slice(r0, r1), slice(c0, c1)


def hist_with_cutoffs(ax, values, cutoffs, xlabel, bins=100, xlim=None):
    """Histogram of finite values with vertical lines at the cutoffs."""
    values = values[np.isfinite(values)]
    ax.hist(values, bins=bins, range=xlim, color="0.4")
    for c in cutoffs:
        ax.axvline(c, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_yticks([])


def plot_plant_detail(title, plant_label, img_bright, img_rgb, labelmask, color_res,
                      lum_res, lum_disp, plant_row, path_pdf, margin=30, dpi=300):
    """Save a 4-panel per-plant figure: RGB / luminescence crops (top), statistics (bottom)."""

    # Crop around the expanded ROI (or the plant if there's no luminescence)
    roi_mask = (lum_res["labelmask_expanded"] if lum_res is not None else labelmask) == plant_label
    sl = crop_slices(roi_mask, margin, labelmask.shape)
    plant_mask = labelmask == plant_label
    plant_crop = plant_mask[sl].astype(int)

    with plt.rc_context(FIG_RC):
        fig = plt.figure(figsize=(16 / 2.54, 16 / 2.54))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        fig.suptitle(f"{title} — {grid_mapping(plant_label)}")

        # Top left: RGB crop with plant contour, yellow tissue, expanded ROI
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img_bright[sl])
        overlay_mask(ax, color_res["yellow_mask"][sl] & plant_mask[sl], COLOR_YELLOW)
        ax.contour(plant_crop, levels=[0.5], colors=COLOR_PLANT_CONTOUR, linewidths=0.3)
        if lum_res is not None:
            ax.contour(roi_mask[sl], levels=[0.5], colors=COLOR_EXPANDED_ROI, linewidths=0.3)
        ax.set_title("RGB, yellow tissue")
        ax.set_xticks([]); ax.set_yticks([])

        # Top right: luminescence crop with spots (ids) of this plant
        ax = fig.add_subplot(gs[0, 1])
        if lum_res is not None:
            spot_ids = np.flatnonzero(lum_res["spot_to_plant"] == plant_label)
            spot_crop = lum_res["spot_lbl_rgb"][sl]
            show_lum(ax, lum_disp[sl])
            ax.contour(plant_crop, levels=[0.5], colors=COLOR_PLANT_CONTOUR, linewidths=0.3)
            ax.contour(roi_mask[sl], levels=[0.5], colors=COLOR_EXPANDED_ROI, linewidths=0.3)
            ax.contour(np.isin(spot_crop, spot_ids), levels=[0.5], colors=COLOR_SPOTS,
                       linewidths=0.3)
            for prop in regionprops(np.where(np.isin(spot_crop, spot_ids), spot_crop, 0)):
                ax.text(prop.centroid[1], prop.centroid[0], str(prop.label), color="cyan",
                        fontsize=3, ha="left", va="bottom")
            ax.set_title("luminescence (background-subtracted)")
        else:
            ax.set_title("no ES image")
        ax.set_xticks([]); ax.set_yticks([])

        # Bottom left: colour features of the plant pixels, with the cutoffs
        gs_bl = gs[1, 0].subgridspec(3, 1, hspace=0.9)
        ax = fig.add_subplot(gs_bl[0])
        for i, c in enumerate(["red", "green", "blue"]):
            ax.hist(img_rgb[..., i][sl][plant_mask[sl]], bins=64, range=(0, 255),
                    histtype="step", color=c, linewidth=0.6)
        ax.set_xlabel("R, G, B", labelpad=1); ax.set_yticks([])
        ax.set_title(f"green {plant_row['green_area_px']} px, yellow {plant_row['yellow_area_px']} px, "
                     f"unclassified {plant_row['unclassified_area_px']} px")
        features, cutoffs = color_res["debug"]["features"], color_res["debug"]["cutoffs"]
        for ax_idx, name, xlim in [(1, "rg_ratio", (0, 2)), (2, "chroma", (0, 1))]:
            ax = fig.add_subplot(gs_bl[ax_idx])
            hist_with_cutoffs(ax, features[name][sl][plant_mask[sl]], cutoffs[name], name,
                              xlim=xlim)

        # Bottom right: ES intensity in the expanded ROI, cutoff, spot medians, stats
        gs_br = gs[1, 1].subgridspec(2, 1, hspace=0.5)
        ax = fig.add_subplot(gs_br[0])
        if lum_res is not None:
            roi_es = lum_res["plant_lbl_es"] == plant_label
            vals = lum_res["lum"][roi_es]
            ax.hist(vals, bins=np.arange(vals.min(), vals.max() + 2) - 0.5, color="0.4", log=True)
            ax.axvline(lum_res["cutoff"], color="k", linestyle="--", linewidth=0.5)
            medians = lum_res["spots"]["median"][spot_ids - 1] + lum_res["background"] if spot_ids.size else []
            ax.plot(medians, np.ones(len(medians)), "|", color=COLOR_SPOTS, markersize=6)
            ax.set_xlabel("raw ES value in expanded ROI (| = spot medians)", labelpad=1)
        ax_txt = fig.add_subplot(gs_br[1]); ax_txt.axis("off")
        if lum_res is not None:
            keys = ["nr_spots", "spot_area_px", "spot_area_pct", "spot_median_brightest",
                    "spot_median_dimmest", "spot_mean_brightest", "spot_mean_dimmest",
                    "spot_median_all", "spot_mean_all", "largest_spot_area_px",
                    "smallest_spot_area_px", "lum_max", "lum_integrated"]
            txt = "\n".join(f"{k}: {plant_row[k]:.4g}" if isinstance(plant_row[k], float)
                            else f"{k}: {plant_row[k]}" for k in keys)
            ax_txt.text(0, 1, txt, va="top", family="monospace", fontsize=4.5)

        fig.savefig(path_pdf, dpi=dpi)
        plt.close(fig)


"""From a raw U-Net segmentation to (up to) 16 plant ROIs, one per tray position."""

################################################################################
# %% Libraries

import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, remove_small_objects
from skimage.segmentation import watershed

################################################################################
# %% Tray positions


def grid_mapping(lbl_idx, n_cols=4):
    """Map label 1..16 to its tray position A1..D4 (row letter, column number)."""
    row = (lbl_idx - 1) // n_cols
    col = (lbl_idx - 1) % n_cols
    return chr(65 + row) + str(col + 1)

################################################################################
# %% ROI steps


def watershed_by_seeds(segmask, seeds, label_class=2):
    """Split the plant class into regions grown from the fixed tray-position seeds."""
    plant_mask = segmask == label_class
    return watershed(-1 * plant_mask, markers=seeds, mask=plant_mask)


def clean_labelmask(labelmask_shed, dilation_radius=10, min_size_cluster=600,
                    min_size_wholeplant=400):
    """Remove small fragments that are not close to a larger plant part.

    Fragments within `dilation_radius` of each other are grouped into one cluster;
    clusters with fewer than `min_size_cluster` plant pixels (undilated area) are
    removed. Then, plants with an area below `min_size_wholeplant` are removed.
    """

    # Group plant pixels that are close together: label the dilated mask, and
    # give each (undilated) plant pixel the id of its cluster
    clusters_dilated = label(dilation(labelmask_shed > 0, disk(dilation_radius)))
    labelmask_clusters = np.zeros_like(labelmask_shed)
    labelmask_clusters[labelmask_shed > 0] = clusters_dilated[labelmask_shed > 0]

    # Remove small clusters (area counted on undilated pixels), keep the
    # watershed labels for the rest
    mask_tokeep = remove_small_objects(labelmask_clusters, min_size=min_size_cluster) > 0
    labelmask_cleaned = np.zeros_like(labelmask_shed)
    labelmask_cleaned[mask_tokeep] = labelmask_shed[mask_tokeep]

    # Second pass: remove whole plants that are too small
    for prop in regionprops(labelmask_cleaned):
        if prop.area < min_size_wholeplant:
            labelmask_cleaned[labelmask_cleaned == prop.label] = 0

    return labelmask_cleaned


def segmentation_to_plant_rois(segmask, seeds, label_class=2, dilation_radius=10,
                               min_size_cluster=600, min_size_wholeplant=400):
    """Return a label mask with label = tray position (1..16) and 0 = not a plant."""
    labelmask_shed = watershed_by_seeds(segmask, seeds, label_class)
    return clean_labelmask(labelmask_shed, dilation_radius, min_size_cluster,
                           min_size_wholeplant)


"""From a raw U-Net segmentation to (up to) 16 plant ROIs, one per tray position.

Labels corresponding to positions are assigned based on watershed with seeds
corresponding to pot positions, and mask is cleaned up by connecting 
neighboring regions and removing small parts.

Logic designed by human author (legacy/pipelineclean_phase4_postprocessing_exploratory.py);
code written by Claude Opus 5.5 (Anthropic), reviewed and edited by human author.
"""

################################################################################
# %% Libraries

import imageio.v3 as iio
import numpy as np

import skimage as sk

################################################################################
# %% Tray positions


def grid_mapping(lbl_idx, n_cols=4):
    """Map label 1..16 to its tray position A1..D4 (row letter, column number)."""

    # get row and col from lbl
    row = (lbl_idx - 1) // n_cols
    col = (lbl_idx - 1) % n_cols

    # give name (e.g. B5)
    position = chr(65 + row) + str(col + 1)
    
    return position


def load_seedmask_positions(path, n_positions=16):
    """Load the seed mask (8-bit PNG, editable in e.g. Fiji): 0 = none, 1..16 = tray positions."""

    # Load seedmask
    seedmask = iio.imread(path)

    # Check correct format (2-D, integer value <= n_positions)
    if seedmask.ndim != 2 or not np.all(np.isin(np.unique(seedmask), np.arange(n_positions + 1))):
        raise ValueError(f"Seed mask {path} must be a 2-D label image with values 0..{n_positions}")
      
    return seedmask.astype(np.int32)

################################################################################
# %% ROI steps

def watershed_by_seeds(segmask, seedmask_positions, plant_classlabel=2):
    """Assign regions to plant positions, regions not touching seeds get removed"""

    # Get boolean plant mask
    plant_mask = (segmask == plant_classlabel)

    # Apply the watershed
    labelmask_shed = sk.segmentation.watershed(
                        -1 * plant_mask, 
                        markers=seedmask_positions, 
                        mask=plant_mask)
    
    return labelmask_shed


def clean_labelmask(labelmask_shed, dilation_radius=10, min_size_cluster=600,
                    min_size_wholeplant=400):
    """Remove small fragments that are not close to a larger plant part.

    Fragments within `dilation_radius` of each other are grouped into one cluster;
    clusters with fewer than `min_size_cluster` plant pixels (undilated area) are
    removed. 
    
    Regions of different plants could be joined, to safeguard against
    plants keeping neighboring artifacts from other pots alive, a second
    pass size filter is applied.
    """

    # Connect close plant parts -- dilate and relabel
    mask_dilated = sk.morphology.dilation(labelmask_shed > 0, sk.morphology.disk(dilation_radius))
    clusters_dilated = sk.measure.label(mask_dilated)
    # Then create a mask identical to labelmask_shed, but renumbered 
    # according to joined regions ("clusters")
    labelmask_clusters = np.zeros_like(labelmask_shed)
    labelmask_clusters[labelmask_shed > 0] = clusters_dilated[labelmask_shed > 0]

    # Remove "clusters" with little area
    clusters_kept = sk.morphology.remove_small_objects(labelmask_clusters,
                                                       min_size=min_size_cluster)
    mask_tokeep = clusters_kept > 0

    # Then recreate original mask but only with kept regions
    labelmask_cleaned = np.zeros_like(labelmask_shed)
    labelmask_cleaned[mask_tokeep] = labelmask_shed[mask_tokeep]

    # Second pass: remove small labels (per tray position) within kept clusters, e.g. an
    # artifact in an empty pot that survived because it's close to a neighbouring plant
    labelmask_plants = sk.morphology.remove_small_objects(labelmask_cleaned,
                                                          min_size=min_size_wholeplant)

    return labelmask_plants


def segmentation_to_plant_rois(
        segmask, seedmask_positions, 
        plant_classlabel=2, dilation_radius=10, 
        min_size_cluster=600, min_size_wholeplant=400
        ):    
    """Return a label mask with label = tray position (1..16) and 0 = not a plant."""

    # Watershed according to seeds that can be related to tray positions (A1..D4)
    # Removes regions that don't touch the seed
    labelmask_shed = watershed_by_seeds(
        segmask=segmask, 
        seedmask_positions=seedmask_positions, 
        plant_classlabel=plant_classlabel)

    # Remove isolated small parts
    labelmask_plants = clean_labelmask(
        labelmask_shed=labelmask_shed, 
        dilation_radius=dilation_radius, 
        min_size_cluster=min_size_cluster, 
        min_size_wholeplant=min_size_wholeplant
        )
    
    return labelmask_plants

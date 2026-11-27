
# %% ###########################################################################

# Phase 4: postprocessing of the segmentations.
# Unlike phases 1–3 (cheeky_cells), this uses the project-specific package arafromtop/.
# Per plant (tray position A1..D4): area, green/yellow tissue, luminescence spots.
# (The original exploratory version is `pipelineclean_phase4_postprocessing_exploratory.py`.)

# For debugging
#
# import os; os.chdir("/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2026_MLeafs_arabidopsis-fromtop-ML/")



# %% ###########################################################################
# Libraries

import dataclasses

import numpy as np

import arafromtop.plant_rois as ara_plant
import arafromtop.postprocessing as p4
    # import importlib; importlib.reload(p4)

# %% ###########################################################################
# Configuration

DATA_ROOT = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/'

config4 = p4.Phase4Config(
    dir_seg = DATA_ROOT + 'SEG/seg_ORIGINALS_NOT-annotated_2/segfiles/',
    dir_rgb = DATA_ROOT + 'TRAINING/ORIGINALS_NOT-annotated/',
    dir_lum = DATA_ROOT + 'TRAINING/ORIGINALS-all-202604/SP_CR_Collaborator_ArabidopsisMutants_RawFiles/',
    dir_output = DATA_ROOT + 'SEG/seg_ORIGINALS_NOT-annotated_2/postprocessed_2/'
    # see Phase4Config for all other parameters (thresholds, plot options)
)

# %% ###########################################################################
# Calibration (run once; creates calibration_data/tf_lum_to_rgb.npy if missing)

matrix_lum_to_rgb = p4.get_transform(config4)
print(np.round(matrix_lum_to_rgb.params, 5))

# %% ###########################################################################
# Test on one file, including the per-plant plots

seedmask_positions = ara_plant.load_seedmask_positions(config4.resolve(config4.seed_mask_path))
config4_test = dataclasses.replace(config4, save_plant_pdfs=True)

plants_df, image_row = p4.process_single_file(
    "images/07_DPI_C01 tray1 7dpi§001_TopRGB_2026-01-15T14-57-00.png_seg.npz",
    config4_test, seedmask_positions, matrix_lum_to_rgb)
print(image_row)
plants_df

# %% ###########################################################################
# Run all files

# extra output per plant;
# config4.save_plant_pdfs=True
# max_files=3 for a test run
# plants_df, images_df = p4.process_directory(config4, max_files=1)

# Process a directory
plants_df, images_df = p4.process_directory(config4)
    
# Print some results
print(images_df[["image", "status", "n_plants", "lum_available", "n_spots_total",
                 "n_spots_unassigned", "warning_oversized_spot"]])

################################################################################
# %% Some debugging

# Testing on /Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS-all-202604/Images_copied_for_test_202611/test1_20261126/
#
# Phase 4 starts from the segmentation (phase 3 output); the test folder holds the
# RGB + ES images, the segfile of this image is taken from the phase-3 run.
# dir_seg points directly at the folder with the segfile (no "images/" subfolder),
# so the RGB/ES images are looked up directly in the test folder.

DIR_TEST = DATA_ROOT + 'TRAINING/ORIGINALS-all-202604/Images_copied_for_test_202611/test1_20261126/'

config4_dbg = dataclasses.replace(
    config4,
    dir_seg = DATA_ROOT + 'SEG/seg_ORIGINALS_NOT-annotated_2/segfiles/images/',
    dir_rgb = DIR_TEST,
    dir_lum = DIR_TEST,
    dir_output = DATA_ROOT + 'POSTPROC_tests/test1_20261126/',
    glob_pattern = '21_DPI_C01 tray6*_seg.npz',  # only this image
    save_npz = False,
    save_overview_pdf = True,                   # the 1200-dpi overview is slow
    save_plant_pdfs = True,                      # set False for the fastest run
)

# %% Option 1: whole pipeline on this one file (writes CSVs + per-plant PDFs)

# reload the submodules first (reloading p4 alone keeps the old cached submodules),
# and p4 last, so it picks up the reloaded ones
import importlib
import arafromtop.calibration as ara_cal
import arafromtop.color_classification as ara_col
import arafromtop.luminescence as ara_lum
import arafromtop.plotting as ara_plot
for module in [ara_cal, ara_plant, ara_col, ara_lum, ara_plot, p4]:
    importlib.reload(module)

plants_df_dbg, images_df_dbg = p4.process_directory(config4_dbg)
plants_df_dbg

# %%

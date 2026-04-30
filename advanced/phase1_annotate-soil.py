"""
I used this script to annotate soil.
"""

# %% ###########################################################################
# Libraries

import sys

import cheeky_cells.orchestrators.orchestrate_phase1_clean as o1
    # import importlib; importlib.reload(o1)
import cheeky_cells.annotating_data.dedicated_segmentation as cds
    # import importlib; importlib.reload(cds)


# %% ###########################################################################
# Configuration

# Colormap for display of annotation
custom_colormap = {
    0: 'transparent',
    1: 'white',
    2: 'red',
    3: 'green'
}

# Configuration for this dataset
config1 = o1.Phase1Config(
    inputdirectory = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/ORIGINALS_sel_202604/',
    outputdirectory = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/Training_set_Ara_Xantho_202604_soil_MW/',
    tile_size = 5000,
    bg_percentile = 10,
    file_formats = ('.png', ),
        # Note: "file_formats" needs to be a "tuple", either ('entry1',) or 
        # ('entry1', 'entry2') etc. Trailing comma required for length 1.
    segfn = None,
    rescalegreyforseg=False,
    mylabelcolormap=custom_colormap
)

o1.phase1_setup(config1)

# %% ###########################################################################
# Running it

# First, adjust the metadata_imagefiles_autogen.xlsx as required
# (and rename it to avoid overwriting behavior).

config1.metadatafiles_path = \
    config1.outputdirectory + "metadata_imagefiles_autogen.xlsx"

# now create annotated pictures
o1.phase1_annotate(config1)
# %%

""" 
This script takes separate plant leaf and pot annotation files respectively, and 
puts it in two layers.
"""

################################################################################
# %%

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# %%

# get list of soil annot files
soil_dir = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/Training_set_Ara_Xantho_202604_soil_MW/humanseg/"
soil_annot_files = glob.glob(soil_dir + "*_tile_seg.npy")

# check for matching files in other dir
leaf_dir = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/TRAINING-singlelayer/humanseg/"

# Check if everything in order
for filepath_soil in soil_annot_files:
    
    filename = filepath_soil.split("/")[-1]
    matching_filepath_leaf = leaf_dir + filename
    
    # check if file exists
    if not os.path.exists(matching_filepath_leaf):
        print(f"Not found: {filename}")
    else:
        print(f"FOUND: {filename}")


################################################################################
# %%

output_directory = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/TRAINING_2layers/humanseg/"

os.makedirs(output_directory, exist_ok=True)

# Now merge them, soil = 1, leaf = 2
for filepath_soil in soil_annot_files:
    # filepath_soil = soil_annot_files[4]
    
    filename = filepath_soil.split("/")[-1]
    filepath_leaf = leaf_dir + filename
    
    # load both files
    mask_soil = np.load(filepath_soil)
    mask_leaf = np.load(filepath_leaf)
    
    # now merge
    mask_both = mask_soil.copy()
    mask_both[mask_leaf==1] = 2
    
    # show
    # plt.imshow(mask_both)
    
    # save result in output directory
    output_path = output_directory + filename
    np.save(output_path, mask_both)
# %%

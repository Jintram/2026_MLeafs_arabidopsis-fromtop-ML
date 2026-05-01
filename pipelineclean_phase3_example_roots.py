
# %% ###########################################################################

# Segments arabidopsis root/shoot dataset


# %% ###########################################################################
# Libraries

from matplotlib.colors import ListedColormap

import cheeky_cells.orchestrators.orchestrate_phase3_clean as o3
    # import importlib; importlib.reload(o3)
    # import importlib; importlib.reload(crw)

import cheeky_cells.plotting.plotting as pp
    # import importlib; importlib.reload(pp)

# %% ###########################################################################
# Configuration

# dataset spcecific config
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/SEG/test-output/'
CURRENT_MODEL = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/models/modelUNet20260501_1518__trained00h21m.pth'
DATA_DIR = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/SEG/test-input/'

# Colormap for display of annotation
custom_colormap_mpl = ListedColormap([
        '#000000',
        '#E9BF94',
        'darkgreen',
    ])
# set global font-size to 8
pp.set_global_fontsize(8)

# Now initialize a configuration
config3 = o3.Phase3Config(
    outputdirectory = OUTPUT_DIR,
    nr_classes = 3,
    nr_channels_input = 3, # (input is rgb, so 3 channels)
    model_checkpoint_to_load = CURRENT_MODEL,
    bg_percentile = 10,
    data_path_input = DATA_DIR,
    cmap_custom = custom_colormap_mpl,
    fn_plotting = pp.plot_overlay_n_pred,
)    

# Collect all files that are to be segmented, store data in metadata
config3 = o3.collect_filelist(config3)
    # config = config3_ara_root
    
    # hacky option (do not use)
    # file_formats =["_img.npy"]
    
    
# now segment them
o3.segment_all_files(config3,
                     # max_files_to_process=200, # for test-run purposes
                     overwrite_files=True
                     )


################################################################################
# %% 

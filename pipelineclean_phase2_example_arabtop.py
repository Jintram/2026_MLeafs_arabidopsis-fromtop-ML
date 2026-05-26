"""
In phase 2, the annotated tiles from phase 1 are used to train
a U-Net segmentation model.
"""

# %% ###########################################################################
# Libraries

import cheeky_cells.orchestrators.orchestrate_phase2_clean as o2
    # import importlib; importlib.reload(o2)


# %% ###########################################################################
# Configuration

# Colormap for display of annotation
custom_colormap = {
    0: 'transparent',
    1: '#E9BF94',
    2: 'darkgreen',
    3: 'red'
}

config2 = o2.Phase2Config(
    outputdirectory = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/',
    metadata_customized_filename = 'metadata_imagefiles_manual.xlsx',
    target_device = 'mps',
    nr_classes = 3,
    img_suffix = '_tile_img_enhanced',
    lbl_suffix = '_tile_seg',
    learning_rate = 1e-3,
    batch_size = 8,
    # for test run (n=100 / e=30 --> 21 mins; .7 mins/epoch)
    # artificial_n=100, 
    # epochs = 30,
    # for actual run
    artificial_n = 1000, # 1000, # ±7min/epoch
    epochs = 120, # 120, # 12 training hours (720 mins) --> ±100 epochs
    lr_schedule_step_len = 40, # should be epochs/3
    cmap_custom = custom_colormap
)

dataset_train, dataset_test, model_unet = o2.phase2_setup(config2)

# %% ###########################################################################
# Running it

saved_model_path = o2.phase2_train(config2, dataset_train, dataset_test, model_unet)
print(f'Model saved to: {saved_model_path}')


# %% ###########################################################################
# Plotting performance

import numpy as np

import cheeky_cells.plotting.plot_train_stats as ptt
    # import importlib; importlib.reload(ptt)

list_confusion_matrices = np.load(
        '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/models/modelUNet20260524_2309_stats.npz'
    )['list_confusion_matrices']

class_names = ['background','soil','plant']
custom_colormap2 = {
    class_names[0]: 'black',
    class_names[1]: '#E9BF94',
    class_names[2]: 'darkgreen'
}

_, _ = ptt.plot_metrics(list_confusion_matrices, 
                 cmap_custom = custom_colormap2_named,
                 class_names=class_names,
                 save_path = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/performance/'+
                            'performance_during_training.pdf')

_, _ = ptt.plot_final_confusion_matrix(list_confusion_matrices, 
                                class_names = class_names,
                                save_path = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/performance/'+
                            'confusion_matrix.pdf')
_, _ = ptt.plot_final_confusion_matrix(list_confusion_matrices, 
                                class_names = class_names,
                                normalize_by = "predicted",
                                save_path = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/performance/'+
                            'confusion_matrix.pdf')



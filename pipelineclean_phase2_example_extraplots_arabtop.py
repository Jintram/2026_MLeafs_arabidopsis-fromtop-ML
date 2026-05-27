"""
Re-generate performance plots in a custom way. 
(These plots are already made during training as well.)
"""

# %% ###########################################################################
# Libraries

import numpy as np
import cheeky_cells.plotting.plot_train_stats as ptt


# %% ###########################################################################
# These plots are already made automatically, but can be further 
# customized here if desired.

    # import importlib; importlib.reload(ptt)

list_confusion_matrices = np.load(
        '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/models/modelUNet20260524_2309_stats.npz'
    )['list_confusion_matrices']
list_lr = np.load('/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/models/modelUNet20260524_2309_stats.npz'
    )['list_lr']

class_names = ['background','soil','plant']
custom_colormap2 = {
    class_names[0]: 'black',
    class_names[1]: '#E9BF94',
    class_names[2]: 'darkgreen'
}

_, _ = ptt.plot_metrics(list_confusion_matrices, 
                 cmap_custom = custom_colormap2,
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

_, _ = ptt.plot_learning_rate(list_lr,
                            save_path = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/performance/'+
                            'learning_rate.pdf')
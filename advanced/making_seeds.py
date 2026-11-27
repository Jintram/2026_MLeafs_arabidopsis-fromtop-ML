
"""
Initial mask was made using `advanced/comparing_images_napari.py`.
"""




# %%

import imageio
import numpy as np

import matplotlib.pyplot as plt

from skimage.morphology import dilation, disk

# %%

mask_seeds = np.load("legacy/plant_seed_mask.npz")["lbls"]
plt.imshow(mask_seeds)

# %%

mask_seeds_dilated = dilation(mask_seeds, disk(10))
for i in range(9):
    print(".")
    mask_seeds_dilated = dilation(mask_seeds_dilated, disk(10))
plt.imshow(mask_seeds_dilated)

# now save
np.savez("plant_seed_mask_dilated.npz", mask_seeds_dilated=mask_seeds_dilated)

# %% Save to png (8-bit grey, lossless, editable in e.g. Fiji)

imageio.imwrite("calibration_data/seedmask_positions.png", mask_seeds_dilated.astype(np.uint8))
"""This file was used to generate plant seeds for the post-processing.

"""

# %%

import napari
from pathlib import Path
import imageio

import numpy as np

import matplotlib.pyplot as plt

# %%

def show_images_in_napari(path_img1, path_img2):
    """
    Display two images in napari.
    
    Args:
        path_img1: Path to the first image
        path_img2: Path to the second image
    """
    
    # Load images
    img1 = np.array(imageio.imread(path_img1))
    img2 = np.array(imageio.imread(path_img2))
    
    # Display them    
    viewer = napari.Viewer()
    viewer.add_image(img1, name='Image 1')
    viewer.add_image(img2, name='Image 2')
    # Add annotation layer
    lbl = np.zeros((img1.shape[0], img1.shape[1]), dtype=int)
    viewer.add_labels(lbl, name='Annotation')
        
    napari.run()
    
    # now retrieve and return the labels
    lbls = viewer.layers['Annotation'].data
    
    return lbls

# %%

if __name__ == "__main__":
    
    path_img1 = Path("/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS_NOT-annotated/images/07_DPI_C01 tray1 7dpi§001_TopRGB_2026-01-15T14-57-00.png.png")
    path_img2 = Path("/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS_NOT-annotated/images/21_DPI_C02 tray5 21dpi§001_TopRGB_2026-02-05T11-57-05.png.png")    
    
    lbls = show_images_in_napari(path_img1, path_img2)
    
    # now save the lbs to a file
    np.savez_compressed("plant_seed_mask.npz")
    
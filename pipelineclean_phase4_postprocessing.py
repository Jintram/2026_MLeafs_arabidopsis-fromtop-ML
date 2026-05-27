


# %% ###########################################################################

import os
import imageio
import numpy as np

import matplotlib.pyplot as plt

from skimage.segmentation import watershed
from skimage.morphology import dilation, disk, remove_small_objects
from skimage.measure import regionprops

from matplotlib.colors import ListedColormap



# %% ###########################################################################


def cmap_random(N, min_brightness=.5):
    """ generate a cmap with random colors (first one black) """
    cmap = np.zeros((N+1, 3))
    for i in range(1, N+1):
        cmap[i] = np.random.rand(3)*(1-min_brightness) + min_brightness
    return ListedColormap(cmap)

def cmap_random_dark(N, maxbrightness=.4):
    """ generate a cmap with random colors (first one black) """
    cmap = np.zeros((N+1, 3))
    for i in range(1, N+1):
        cmap[i] = maxbrightness - np.random.rand(3)*maxbrightness
    return ListedColormap(cmap)

# set op grid mapping 1 .. 16 to A1 .. D4
def grid_mapping(lbl_idx):
    row = (lbl_idx-1) // 4
    col = (lbl_idx-1) % 4
    return chr(65+row) + str(col+1)


# %% ###########################################################################

CURRENT_SEGFILE = "07_DPI_C01 tray1 7dpi§001_TopRGB_2026-01-15T14-57-00.png_seg.npz"
SEG_MASK_DIR = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/SEG/seg_ORIGINALS_NOT-annotated_2/segfiles/images/"
PLANT_IMAGE_DIR = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS_NOT-annotated/images/"

current_file_plant = CURRENT_SEGFILE.replace("_seg.npz",".png")

path_segmask  = SEG_MASK_DIR + CURRENT_SEGFILE
path_plantimg = PLANT_IMAGE_DIR + current_file_plant

# load the file
segmask = np.load(path_segmask)['img_pred_lbls']
# 
plt.imshow(segmask==2)

# Load the image
img_plant = imageio.imread(path_plantimg)
plt.imshow(img_plant)

# apply strong gamme transform
img_plant_bright = (np.power(img_plant/255, .5)*255).astype(int)
plt.imshow(img_plant_bright)


# load seed mask
mask_seeds = np.load("plant_seed_mask_dilated.npz")["mask_seeds_dilated"]
plt.imshow(mask_seeds)

# now apply watershedding based one seeds
labelmask_shed = watershed((segmask==2)*-1, 
                   markers = mask_seeds,
                   mask = (segmask==2))

plt.imshow(labelmask_shed, cmap=cmap_random(16))



mask_shed_dilated = dilation(labelmask_shed>0, disk(10))
plt.imshow(mask_shed_dilated)

# now remove objects that are smaller than 600 px^2 
# removes stuff with diameter of 7;
# ((7+10+10)/2)**2*math.pi 
# = 572
mask_cleaned = remove_small_objects(mask_shed_dilated, min_size=600)
labelmask_cleaned = np.zeros_like(labelmask_shed)
labelmask_cleaned[mask_cleaned] = labelmask_shed[mask_cleaned]

plt.imshow(mask_cleaned)
plt.imshow(labelmask_cleaned, cmap=cmap_random(16))

# get some stats
plant_regionprops = regionprops(labelmask_cleaned)

# now plot 
OUTDIR = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS_NOT-annotated/processing/"
os.makedirs(OUTDIR, exist_ok=True)
# mycmap = cmap_random_dark(50, .6)
mycmap = cmap_random(1+np.max(labelmask_cleaned), min_brightness=.6)

plt.rcParams.update({'font.size': 6, 'font.family': 'Arial'})
fig, axs = plt.subplots(1,2, figsize=(16.8/2.54, 8/2.54))
fig.suptitle(f"{CURRENT_SEGFILE}")
axs[0].imshow(img_plant_bright)
axs[1].imshow(labelmask_cleaned, cmap=mycmap)
for lbl_idx in range(1,1+np.max(labelmask_cleaned)):
    axs[0].contour(labelmask_cleaned==lbl_idx, colors='white', linewidths=.05)
    # write plant label
    
    y, x = plant_regionprops[lbl_idx-1].centroid
    ytop, xleft = plant_regionprops[lbl_idx-1].bbox[:2]
    #axs[0].plot(x,y,'xk',markersize=1)
    #axs[0].plot(xleft,ytop,'s',markersize=5, color=mycmap(lbl_idx), markeredgewidth=0)
    axs[0].plot(xleft,ytop,'s',markersize=5, color='black', markeredgewidth=0)
    # text label next to the point
    #axs[0].text(xleft, ytop, grid_mapping(lbl_idx), color='white', fontsize=4, ha='center', va='center')
    axs[0].text(xleft, ytop, grid_mapping(lbl_idx), color=mycmap(lbl_idx), fontsize=4, ha='center', va='center')

    # also add to 2nd panel    
    axs[1].plot(xleft,ytop,'s',markersize=5, color='black', markeredgewidth=0, alpha=.5)
    axs[1].text(xleft, ytop, grid_mapping(lbl_idx), color=mycmap(lbl_idx), fontsize=4, ha='center', va='center')

plt.tight_layout()
# save
plt.savefig(OUTDIR + f"SEG-post_{CURRENT_SEGFILE}.pdf", dpi=1200)
plt.close()

# %% ###########################################################################

"""Batch postprocessing for Arabidopsis top-view segmentation masks."""

################################################################################
# %% 

import glob
import os
from dataclasses import dataclass

import imageio
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from skimage.measure import regionprops
from skimage.morphology import dilation, disk, remove_small_objects
from skimage.segmentation import watershed

################################################################################
# %% Configuration


@dataclass
class phase4_config:
	seg_mask_dir: str
	dir_img_files: str
	dir_output: str
	seed_mask_path: str = "plant_seed_mask_dilated.npz"
	glob_pattern: str = "**/*_seg.npz"
	label_class: int = 2
	dilation_radius: int = 10
	min_size: int = 600
	gamma: float = 0.5
	save_pdf: bool = True
	save_npz: bool = False
	dpi: int = 1200

################################################################################
# %% Utilities

def cmap_random(n_colors, min_brightness=0.5):
	"""Generate a colormap with random colors (first one black)."""
	cmap = np.zeros((n_colors + 1, 3))
	for idx in range(1, n_colors + 1):
		cmap[idx] = np.random.rand(3) * (1 - min_brightness) + min_brightness
	return ListedColormap(cmap)


def grid_mapping(lbl_idx):
	"""Map 1..16 to a 4x4 grid label such as A1 or D4."""
	row = (lbl_idx - 1) // 4
	col = (lbl_idx - 1) % 4
	return chr(65 + row) + str(col + 1)

################################################################################
# %% 


def process_single_file(
	current_segfile,
	config4,
	mask_seeds,
):
    # construct paths
	current_file_plant = current_segfile.replace("_seg.npz", ".png")
	path_segfile  = os.path.join(config4.seg_mask_dir , current_segfile)
	path_plantimg = os.path.join(config4.dir_img_files, current_file_plant)

    # load segmask & plant image
	segmask = np.load(path_segfile)["img_pred_lbls"]
        # plt.imshow(segmask)
	img_plant = imageio.imread(path_plantimg)
	img_plant_bright = (np.power(img_plant / 255.0, config4.gamma) * 255).astype(int)
        # plt.imshow(img_plant_bright)

	# Perform watershed
	labelmask_shed = watershed(
		-1 * (segmask == config4.label_class), 
		markers=mask_seeds, 
		mask=(segmask == config4.label_class))
	# Dilate binary mask to make plant-parts touching, then remove small parts
	mask_shed_dilated = dilation(labelmask_shed > 0, disk(config4.dilation_radius))
	mask_cleaned = remove_small_objects(mask_shed_dilated, min_size=config4.min_size)
	# Now retain the large objects from the watershed
	labelmask_cleaned = np.zeros_like(labelmask_shed)
	labelmask_cleaned[mask_cleaned] = labelmask_shed[mask_cleaned]
        # plt.imshow(labelmask_cleaned, cmap=cmap_random(50))

    # SHOULD BE A FUNCTION
	# The plotting step is kept separate because it is the only part that needs a
	# visual representation; the processing itself returns label masks.
	n_labels = int(np.max(labelmask_cleaned))
	mycmap = cmap_random(max(n_labels, 1), min_brightness=0.6)
	regionprops_map = {prop.label: prop for prop in regionprops(labelmask_cleaned)}

	with plt.rc_context({"font.size": 6, "font.family": "Arial"}):
		fig, axs = plt.subplots(1, 2, figsize=(16.8 / 2.54, 8 / 2.54))
		fig.suptitle(f"{current_segfile}")
		axs[0].imshow(img_plant_bright)
		axs[1].imshow(labelmask_cleaned, cmap=mycmap)

		for lbl_idx, prop in regionprops_map.items():
			axs[0].contour(labelmask_cleaned == lbl_idx, colors="white", linewidths=0.05)
			ytop, xleft = prop.bbox[:2]
			axs[0].plot(xleft, ytop, "s", markersize=5, color="black", markeredgewidth=0)
			axs[0].text(xleft, ytop, grid_mapping(lbl_idx), color=mycmap(lbl_idx), fontsize=4, ha="center", va="center")

			axs[1].plot(xleft, ytop, "s", markersize=5, color="black", markeredgewidth=0, alpha=0.5)
			axs[1].text(xleft, ytop, grid_mapping(lbl_idx), color=mycmap(lbl_idx), fontsize=4, ha="center", va="center")

		plt.tight_layout()
    # END SHOULD BE A FUNCTION
	
    
    # Prepare output file paths    
    os.makedirs(config4.dir_output, exist_ok=True)
	output_stem = f"SEG-post_{os.path.splitext(current_segfile)[0]}"
	output_pdf = os.path.join(config4.dir_output, f"{output_stem}.pdf")
	output_npz = os.path.join(config4.dir_output, f"{output_stem}.npz")
	
    # save  
    if config4.save_pdf:
        plt.savefig(output_pdf, dpi=config4.dpi)
    plt.close(fig)

    # save post-processed segmask
	if config4.save_npz:
		np.savez(output_npz, labelmask_cleaned=labelmask_cleaned, segmask=segmask)

	return {
		"path_segfile": path_segfile,
		"image_path": path_plantimg,
		"output_pdf": output_pdf if config4.save_pdf else None,
		"output_npz": output_npz if config4.save_npz else None,
		"n_labels": n_labels,
		"current_file_plant": current_file_plant,
	}

# %%

# --- Batch driver -------------------------------------------------------------
# This is the function you call from another Python script. It discovers all
# matching segmasks, loads the shared seed mask once, and then processes files
# one by one.
def process_segmask_directory(
	config4,
):
    
	config4 = phase4_config(
        seg_mask_dir = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/SEG/seg_ORIGINALS_NOT-annotated/segfiles/" ,
        dir_img_files = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS_NOT-annotated/",
        dir_output = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/SEG/seg_ORIGINALS_NOT-annotated_2/postprocessed/"
    )
    
    
	mask_seeds = np.load(config4.seed_mask_path)["mask_seeds_dilated"]
	
	segmask_files = sorted(
		glob.glob(
			config4.glob_pattern, 
            root_dir = config4.seg_mask_dir,
			recursive=True
        )
    )

	results = []
	for current_segfile in segmask_files:
		# current_segfile = segmask_files[0]
		
		results.append(
			process_single_file(
				segmask_path=current_segfile,
				config4=config4,
				mask_seeds=mask_seeds,
			)
		)

	return results








# %% ###########################################################################
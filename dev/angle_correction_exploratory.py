


# see claude discussion + paper notes.
# https://claude.ai/share/2e4d5dec-2cca-4789-aa9b-7d6eca486a6e

################################################################################
# %% Libs

import imageio.v3 as iio

import numpy as np

import matplotlib.pyplot as plt

import skimage as sk

################################################################################
# %% Load test images

path_test_images = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/test-images-angle-correction-mw/"
image1 = iio.imread(path_test_images + "image1_portrait.jpeg")
image2 = iio.imread(path_test_images + "image2.jpeg")

plt.imshow(image1)
plt.imshow(image2)

# convert to grey scale
image1_gray = sk.color.rgb2gray(image1)
image2_gray = sk.color.rgb2gray(image2)

################################################################################
# %% Collect reference point locations

plt.figure(); plt.imshow(image2_gray, cmap='gray'); plt.title('image2: click landmarks')
pts2 = np.array(plt.ginput(n=-1, timeout=0))   # (N, 2) as (x, y)

plt.figure(); plt.imshow(image1_gray, cmap='gray'); plt.title('image1: click SAME landmarks, SAME order')
pts1 = np.array(plt.ginput(n=-1, timeout=0))

# save both landmarks
np.save(path_test_images + "landmark_pts1.npy", pts1)
np.save(path_test_images + "landmark_pts2.npy", pts2)

# estimate transform
tf = sk.transform.estimate_transform("projective", pts2, pts1)



################################################################################
# %% Now try to get a transformation matrix to map image2 --> image1

image2_warped = sk.transform.warp(image2, tf.inverse)

# plot image 1 and 2 originals next to each other
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image1)
axs[0].set_title('Image 1')
axs[1].imshow(image2)
axs[1].set_title('Image 2')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image1)
axs[0].set_title('Image 1')
axs[1].imshow(image2_warped)
axs[1].set_title('Image 2 warped')


# binarized grey scale plotted on top of each other in different colors
# warp the grey version into image1's frame (same shape as image1)
image2_gray_warped = sk.transform.warp(image2_gray, tf.inverse, output_shape=image1_gray.shape, order=1)
# pixels that fall outside image2 after warping (to exclude from the overlay)
valid2 = sk.transform.warp(np.ones_like(image2_gray), tf.inverse, output_shape=image1_gray.shape, order=0) > 0.5

# binarize each image with its own Otsu threshold
# (foreground = darker than threshold; set to False if objects of interest are bright)
DARK_FOREGROUND = True
thr1 = sk.filters.threshold_otsu(image1_gray)
thr2 = sk.filters.threshold_otsu(image2_gray_warped[valid2])
bin1 = image1_gray < thr1 if DARK_FOREGROUND else image1_gray > thr1
bin2 = (image2_gray_warped < thr2 if DARK_FOREGROUND else image2_gray_warped > thr2) & valid2


plt.imshow(bin1, cmap="Reds", alpha=0.5)
plt.imshow(bin2, cmap="Blues", alpha=0.5)


# %% PART II actual images

# Use transformation matrix from previous script
# (no calibration pictures are available)
# Source: https://github.com/MolPlantPathology/Digital_phenotyper/blob/main/example.env
tf_ara = np.array([[1.38108321e+00, 3.13548708e-02, 1.16323458e+02],
                   [1.98864470e-02, 1.38255468e+00, 1.13998417e+03],
                    [1.13790555e-05, 1.43271706e-05, 1.00000000e+00]])


path_img_ara_lum = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/luminescence/21_DPI_C01 tray3 21dpi§001_TopES_2026-01-29T15-48-05.png.png"
path_img_ara_rgb = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/rgb/21_DPI_C01 tray3 21dpi§001_TopRGB_2026-01-29T15-48-05.png.png"

img_ara_rgb = iio.imread(path_img_ara_rgb)
img_ara_lum = iio.imread(path_img_ara_lum)



img_ara_rgb_gray = sk.color.rgb2gray(img_ara_rgb[:,:,:3])
img_ara_lum_maxp = np.max(img_ara_lum[:,:,:3], axis=2) 
lum_minval = np.min(img_ara_lum_maxp)
plt.imshow(img_ara_rgb_gray)
plt.imshow(img_ara_lum_maxp>lum_minval)

# now apply
img_ara_lum_maxp_warped = sk.transform.warp(img_ara_lum_maxp, tf_ara, output_shape=img_ara_rgb_gray.shape)

# show next to each other
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_ara_rgb_gray, cmap="viridis")
axs[1].imshow(img_ara_lum_maxp_warped, cmap="viridis")

# show on top of each other, without correction
plt.imshow(img_ara_rgb_gray, cmap="Greys")
img_ara_lum_maxp_warped_binary = img_ara_lum_maxp_warped>img_ara_lum_maxp_warped.min()
plt.imshow(img_ara_lum_maxp_warped_binary, alpha=img_ara_lum_maxp_warped_binary*1.0, cmap="Reds")
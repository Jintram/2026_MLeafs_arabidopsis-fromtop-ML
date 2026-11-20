


# see claude discussion + paper notes.
# https://claude.ai/share/2e4d5dec-2cca-4789-aa9b-7d6eca486a6e

################################################################################
# %% Libs

import imageio.v3 as iio

import numpy as np

import matplotlib.pyplot as plt

import skimage as sk

import cv2
import scipy.ndimage as ndi

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

plt.figure()
plt.imshow(bin1, cmap="Reds", alpha=0.5)
plt.imshow(bin2, cmap="Blues", alpha=0.5)


# %% PART II actual images

# Use transformation matrix from previous script
# (no calibration pictures are available)
# Source: https://github.com/MolPlantPathology/Digital_phenotyper/blob/main/example.env
tf_ara = np.array([[1.38108321e+00, 3.13548708e-02, 1.16323458e+02],
                   [1.98864470e-02, 1.38255468e+00, 1.13998417e+03],
                    [1.13790555e-05, 1.43271706e-05, 1.00000000e+00]])

# Example 1
path_img_ara_lum = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/luminescence/21_DPI_C01 tray3 21dpi§001_TopES_2026-01-29T15-48-05.png.png"
path_img_ara_rgb = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/rgb/21_DPI_C01 tray3 21dpi§001_TopRGB_2026-01-29T15-48-05.png.png"
# Example 2
path_img_ara_lum = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/luminescence/21_DPI_C01 tray3 21dpi§001_TopES_2026-01-29T15-48-05.png.png"
path_img_ara_rgb = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/rgb/21_DPI_C01 tray3 21dpi§001_TopRGB_2026-01-29T15-48-05.png.png"
# Example 3 (strong lum signal, best example)
path_img_ara_lum = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS-all-202604/Images_rosette_detection/21dpi Tray12§001_TopES_2025-06-10T16-23-28.png"
path_img_ara_rgb = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/ORIGINALS-all-202604/Images_rosette_detection/21dpi Tray12§001_TopRGB_2025-06-10T16-23-28.png"



img_ara_rgb = iio.imread(path_img_ara_rgb)
img_ara_lum = iio.imread(path_img_ara_lum)



img_ara_rgb_gray = sk.color.rgb2gray(img_ara_rgb[:,:,:3])
img_ara_lum_maxp = np.max(img_ara_lum[:,:,:3], axis=2) 
lum_minval = np.min(img_ara_lum_maxp)
plt.imshow(img_ara_rgb_gray)
plt.imshow(img_ara_lum_maxp>lum_minval)

# convert tf_ara to matrix format 
tf_ara_m = sk.transform.ProjectiveTransform(matrix=tf_ara)

# now apply
img_ara_lum_maxp_warped = sk.transform.warp(img_ara_lum_maxp, tf_ara_m.inverse, output_shape=img_ara_rgb_gray.shape)
valid_lum = img_ara_lum_maxp_warped > 0
plt.imshow(img_ara_lum_maxp_warped)

# show next to each other
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_ara_rgb_gray, cmap="viridis")
axs[1].imshow(img_ara_lum_maxp_warped, cmap="viridis")

# %% Warp binary version for display as well

# binarize BEFORE warping (background is exactly lum_minval here)
lum_mask = img_ara_lum_maxp > lum_minval

# warp the mask with nearest neighbour -> stays strictly 0/1, no edge blending
lum_mask_warped = sk.transform.warp(lum_mask.astype(float), tf_ara_m.inverse,
                                    output_shape=img_ara_rgb_gray.shape, order=0) > 0.5


# overlay on a fresh figure
plt.figure(figsize=(8, 8))
plt.imshow(img_ara_rgb_gray, cmap="Greys")
#plt.imshow(lum_mask_warped * 1.0, alpha=lum_mask_warped * 1.0, cmap="Reds", vmin=0, vmax=1)
plt.contour(lum_mask_warped, colors="red", linewidths=1, alpha=0.5)

# # show on top of each other, without correction
# plt.imshow(img_ara_rgb_gray, cmap="Greys")
# img_ara_lum_maxp_warped_binary = img_ara_lum_maxp_warped > np.min(img_ara_lum_maxp_warped[valid_lum])
#     # plt.imshow(img_ara_lum_maxp_warped_binary)
# plt.imshow(img_ara_lum_maxp_warped_binary*1.0, alpha=img_ara_lum_maxp_warped_binary*1.0, cmap="Reds")


################################################################################
# %% Re-do the calibration using the original calibration data

path_calib = "/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2026_MLeafs_arabidopsis-fromtop-ML/calibration_data/"
img_calib_lum = iio.imread(path_calib + "Calibration16tray§004_TopESLive_2023-10-02T12-44-33.png")  # 2048x2048 RGBA
img_calib_rgb = iio.imread(path_calib + "Calibration16tray§004_TopRGB_2023-10-02T12-44-33.png")     # 4112x3008 RGBA

def prep_calib_image(img):
    """Grey uint8 image with only the (contrast-stretched) paper sheet; everything else white.
    Masking is needed because the dark holes of the tray around the sheet are detected as dots too."""
    g = sk.color.rgb2gray(img[:, :, :3])
    paper = g > sk.filters.threshold_otsu(g)
    lab = sk.measure.label(paper)
    paper = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1   # largest bright region = sheet
    paper = ndi.binary_fill_holes(paper)                          # include the dots
    lo, hi = np.percentile(g[paper], [1, 99])
    g = np.clip((g - lo) / (hi - lo), 0, 1)
    g[~paper] = 1
    return (g * 255).astype(np.uint8)

# blob detector with a larger max area than the default (5000 px^2);
# dots in the RGB image are ~86 px across (~5800 px^2) and would otherwise be missed
blob_params = cv2.SimpleBlobDetector_Params()
blob_params.minArea, blob_params.maxArea = 200, 50000
blob_params.filterByCircularity = blob_params.filterByConvexity = blob_params.filterByInertia = False
blob_detector = cv2.SimpleBlobDetector_create(blob_params)

def find_grid(img, pattern=(4, 11)):
    """Returns (44, 2) dot centres as (x, y), sub-pixel, in a fixed grid order."""
    ok, centers = cv2.findCirclesGrid(prep_calib_image(img), pattern,
                                      flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blob_detector)
    assert ok, "grid not found"
    return centers.reshape(-1, 2)

pts_calib_lum = find_grid(img_calib_lum)
pts_calib_rgb = find_grid(img_calib_rgb)

# fit affine and projective with all 44 points, compare residuals
for model in ["affine", "projective"]:
    tf_tmp = sk.transform.estimate_transform(model, pts_calib_lum, pts_calib_rgb)
    resid = np.linalg.norm(tf_tmp(pts_calib_lum) - pts_calib_rgb, axis=1)
    print(f"{model:10s}: median {np.median(resid):.2f} px, max {resid.max():.2f} px")

tf_calib = sk.transform.estimate_transform("projective", pts_calib_lum, pts_calib_rgb)  # lum (x,y) -> rgb (x,y)
print(np.array2string(tf_calib.params / tf_calib.params[2, 2], precision=5))

# compare with the old 4-point matrix
resid_old = np.linalg.norm(tf_ara_m(pts_calib_lum) - pts_calib_rgb, axis=1)
print(f"old tf_ara : median {np.median(resid_old):.2f} px, max {resid_old.max():.2f} px")

# visual check: detected points (order shown by colour; first point = big dot) in both images
fig, axs = plt.subplots(1, 2, figsize=(12, 7))
for ax, img, pts, title in zip(axs, [img_calib_lum, img_calib_rgb], [pts_calib_lum, pts_calib_rgb], ["lum", "rgb"]):
    ax.imshow(img[:, :, :3])
    ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap="viridis", s=10)
    ax.plot(*pts[0], "co", ms=8)
    ax.set_title(title)

# overlay: warped lum (green) on rgb (magenta)
calib_rgb_gray = sk.exposure.rescale_intensity(sk.color.rgb2gray(img_calib_rgb[:, :, :3]))
calib_lum_warped = sk.exposure.rescale_intensity(
    sk.transform.warp(sk.color.rgb2gray(img_calib_lum[:, :, :3]), tf_calib.inverse,
                      output_shape=calib_rgb_gray.shape))
plt.figure(figsize=(8, 10))
plt.imshow(np.dstack([calib_rgb_gray, calib_lum_warped, calib_rgb_gray]))
plt.plot(pts_calib_rgb[:, 0], pts_calib_rgb[:, 1], "y.", ms=3)
plt.title("magenta = rgb, green = warped lum")

# residual map across the sheet (systematic pattern -> lens distortion / model too simple)
resid_vec = tf_calib(pts_calib_lum) - pts_calib_rgb
plt.figure(figsize=(6, 8))
plt.imshow(calib_rgb_gray, cmap="gray")
plt.quiver(pts_calib_rgb[:, 0], pts_calib_rgb[:, 1], resid_vec[:, 0], -resid_vec[:, 1],
           color="red", angles="xy", scale_units="xy", scale=0.01)   # residuals exaggerated 100x
plt.title("projective residuals (100x exaggerated)")

# %% Now apply it to the same examples as above

# warp the mask with nearest neighbour -> stays strictly 0/1, no edge blending
lum_mask_warped_new = sk.transform.warp(lum_mask.astype(float), tf_calib.inverse,
                                    output_shape=img_ara_rgb_gray.shape, order=0) > 0.5


# overlay on a fresh figure
plt.figure(figsize=(8, 8))
plt.imshow(img_ara_rgb_gray, cmap="Greys")
#plt.imshow(lum_mask_warped * 1.0, alpha=lum_mask_warped * 1.0, cmap="Reds", vmin=0, vmax=1)
plt.contour(lum_mask_warped, colors="red", linewidths=1, alpha=0.5)
plt.contour(lum_mask_warped_new, colors="blue", linewidths=1, alpha=0.5)


# %% Conclusion
"""

CONCLUSIONS

- all three cases work (my own test image, rgb/lum based on original matrix, rgb/lum on new matrix from calib images)
- new matrix shows improvement, due to better point extraction: all 44 dots (least-squares fit)
  with sub-pixel centres, instead of 4 integer-rounded corner dots
- a checkerboard pattern would in theory be better (circle centres are slightly biased under
  perspective/distortion), but here that bias is negligible (~0.01-0.02 px)
- more importantly, the field of view isn't entirely covered, leading to large displacement outside where the paper was

Details:
- residuals on the 44 calibration dots (rgb px):
    affine            median 1.25, max 3.29
    projective (new)  median 0.30, max 1.18   -> perspective matters, use projective
    old tf_ara        median 1.69, max 3.20
- old vs new matrix disagree by ~0-2 px where the sheet was (lum x 771-1345, y 732-1543),
  but by up to ~10-17 px at the image edges -> both are extrapolating there
- the extra "padding" in the RGB image (large y-offset) does NOT degrade the fit; translation is
  part of the transform (pre-shifting points gives an identical mapping, diff ~1e-9 px)
- RGB calib image needed preprocessing for cv2.findCirclesGrid: mask to the paper sheet
  (tray holes are detected as dots otherwise), stretch contrast, and raise the blob detector maxArea

Next steps:
- calibrate with the sheet at multiple positions covering the whole field (at plant height),
  pool all point pairs into one fit
- with full coverage, check edge residuals for projective vs polynomial/thin-plate-spline
  (systematic pattern -> lens distortion)


"""




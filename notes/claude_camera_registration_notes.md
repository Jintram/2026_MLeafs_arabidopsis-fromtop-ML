# Two-camera registration — context & theory notes

Context notes from a discussion with Claude (claude.ai), for use as project context.

## Project context

- Two cameras image small *Arabidopsis* plants; each camera acquires a different kind of data.
- The cameras view the scene from slightly different angles.
- Goal: transform images from the **secondary** camera so they align with the **primary** camera, then combine the two data layers.
- Plan: calibrate once with a reference image taken by both cameras, fit a transform, then apply that fixed transform to all later secondary images.
- Decision: the scene is 3D, but we **accept some error and treat it as 2D (planar)**. This is reasonable given the camera distance; most signal comes from roughly one plane.

## Design choices / recommendations

- **Model**: fit both an **affine** (6 DOF) and a **homography / projective** (8 DOF) transform, and compare residuals. A homography is the exact model for a plane seen from two viewpoints by ideal pinhole cameras.
- **Lens distortion**: if residuals show a systematic pattern toward the image edges, correct distortion per camera first (checkerboard calibration, OpenCV), or use a polynomial or thin-plate-spline model.
- **Calibration target**: features visible to *both* cameras/modalities (dots, grid, fiducial markers), spread over the whole field of view. Put the target at the height where most plant tissue is, not at pot level.
- **Fitting**: point correspondences → least squares; use RANSAC if some matches may be wrong. Point-based fitting is more robust than intensity-based fitting across modalities; if intensity-based fitting is needed, use mutual information (SimpleITK / elastix).
- **Validation**: hold out points, report the residual error in pixels/mm, and plot residuals across the field of view (random scatter = good; a systematic pattern = model too simple).
- **Practice**: recalibrate periodically (drift, bumps); resample only once (bilinear, `order=1`) and keep the raw data; use nearest neighbour (`order=0`) for label masks; define the overlapping field of view and the target grid.

### Parallax error estimate (cost of the 2D assumption)

For camera separation B, working distance Z, focal length f (in pixels), an object at height ΔZ away from the calibration plane is misaligned by roughly:

```
error_px ≈ f · B · ΔZ / Z²
```

## Theory summary

### Homogeneous coordinates

A 2×2 matrix always maps the origin to the origin, so it cannot translate. Writing points as `[x, y, 1]` and using 3×3 matrices puts the translation inside the matrix:

```
[x']   [a  b  tx] [x]
[y'] = [c  d  ty] [y]
[1 ]   [0  0  1 ] [1]
```

- Chaining transforms = matrix product (e.g. `T @ S @ R`; order matters).
- Undoing a transform = matrix inverse.
- Homography: the bottom row is a free parameter → the output is `[x', y', w]`, then divide: `x = x'/w`, `y = y'/w` (this gives perspective). The matrix is defined up to scale → 8 DOF.

### Transform hierarchy

| Transform | Adds | Preserves | DOF | Min. point pairs |
|---|---|---|---|---|
| Translation | shift | everything but position | 2 | 1 |
| Euclidean (rigid) | rotation | distances, angles | 3 | 2 |
| Similarity | uniform scale | angles, shapes | 4 | 2 |
| Affine | non-uniform scale, shear | parallel lines | 6 | 3 |
| Projective (homography) | perspective | straight lines | 8 | 4 |

Nonlinear models beyond these: lens distortion (radial polynomial), polynomial transforms, thin-plate splines.

### Rotation matrix

The columns of a matrix are the images of the basis vectors. Rotating by θ (counter-clockwise) sends (1,0) → (cos θ, sin θ) and (0,1) → (−sin θ, cos θ):

```
R(θ) = [cos θ  -sin θ]
       [sin θ   cos θ]
```

- Inverse = transpose (an orthogonal matrix); det = 1 (area preserved).
- The determinant of the 2×2 part of an affine transform = the area scale factor (useful as a check on the relative magnification).
- Rotation about a centre c: `T(c) @ R(θ) @ T(-c)`.
- In image arrays the y-axis points down, so a "counter-clockwise" rotation appears clockwise on screen.

### How an image is warped (inverse mapping)

The matrix acts on **pixel coordinates**, not on the image array. Libraries loop over the **output** grid: map each output pixel back into the source with M⁻¹ (a non-integer position), then interpolate the source intensity there. This leaves no holes and no collisions. This is why `skimage.transform.warp` takes `tform.inverse`.

```python
import numpy as np
from scipy.ndimage import map_coordinates

M = ...                                   # 3x3: secondary -> primary coords (x, y)
Minv = np.linalg.inv(M)
rows, cols = np.indices(out_shape)
out = np.stack([cols.ravel(), rows.ravel(), np.ones(rows.size)])   # (x, y, 1)
src = Minv @ out
src_x, src_y = src[0] / src[2], src[1] / src[2]
warped = map_coordinates(img, [src_y, src_x], order=1, cval=0).reshape(out_shape)
```

**Pitfall:** arrays are indexed (row, col) = (y, x); transform matrices are usually written for (x, y). Mixing these up gives mirrored or wrongly rotated results.

## Tools

- `skimage.transform`: `estimate_transform` / `AffineTransform`, `ProjectiveTransform`, `PolynomialTransform`, `warp`; `skimage.measure.ransac`
- OpenCV: `cv2.findHomography(..., cv2.RANSAC)`, `cv2.estimateAffine2D`, `cv2.warpPerspective`, camera calibration for lens distortion
- SimpleITK / `itk-elastix`: intensity-based or non-rigid registration (mutual information)
- pystackreg: quick rigid/affine alignment

## References

- scikit-image, Types of homographies: https://scikit-image.org/docs/stable/auto_examples/transform/plot_transform_types.html
- scikit-image, Using geometric transformations: https://scikit-image.org/docs/stable/auto_examples/transform/plot_geometric.html
- OpenCV, Basic concepts of the homography: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- OpenCV, Camera calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. (free draft): https://szeliski.org/Book — §2.1 and the image alignment chapter
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, ch. 2
- Wikipedia: Transformation matrix, Homogeneous coordinates, Affine transformation, Homography (computer vision), Distortion (optics), Thin plate spline, Image registration

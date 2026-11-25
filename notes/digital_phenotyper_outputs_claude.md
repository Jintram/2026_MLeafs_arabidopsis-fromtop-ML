# Outputs of Digital_phenotyper_arabidopsis

*Author: Claude (Anthropic, model Claude Opus 5.5), on request of M. Wehrens, 2026-11-25. Written by reading the source code; not checked by running it.*

This file summarises the outputs of the previous phenotyper, one section per output-file type.
It is a reference for building a new tool with the same or improved outputs.
It comes from reading the source code, not from running it.

Input is image pairs: an RGB photo (`*RGB*`) and a CCD luminescence image (`*ES*`) of a tray of rosettes (`ROWS × COLUMNS`).
For each pair, `analyze.py` writes one CSV and one PNG into `CSV_DIR`.
`calibrator.py` only updates the `.env` file.

## 1. Results table: `results_<sample>_NNN.csv`

- One file per image pair, with one row per detected rosette.
- `NNN` is a 3-digit counter that increments so earlier results are not overwritten.
- Areas are in pixels; there is no conversion to mm².
- The 6th column depends on the mode, which is set from the filename: a name starting with `07` → 7 DPI mode, anything else → "14" mode.

| Column | Content | How it is obtained |
|---|---|---|
| `row` | Tray row of the rosette (0-based) | rosette index after sorting `// COLUMNS` |
| `column` | Tray column (0-based) | rosette index after sorting `% COLUMNS` |
| `rosette area` | Total plant area (px) | pixels in the HSV range `YELLOW[0]..GREEN[1]` inside the rosette contour |
| `green area` | Healthy tissue area (px) | `rosette area − yellow area` |
| `yellow area` | Chlorotic/diseased tissue area (px) | pixels in the `YELLOW` HSV range inside the contour, after a 10×10 morphological open then close |
| `no. hydathodes` *(7 DPI mode)* | Number of luminescent spots on the plant | spots from the ES image (threshold 127, size between `NOISE_THRESHOLD_MIN` and `NOISE_THRESHOLD_MAX`) whose centre is inside the rosette's enclosing circle |
| `luminescence area` *(other mode)* | Colonised area (px) | sum of the areas of those same spots. Caveat: each spot's area is about the size of its bounding box, not of the blob itself |
| `global intensity` | Mean luminescence signal of the whole image | 7 DPI: `n/a`. Otherwise: mean grey value of the raw ES pixels above the image minimum. Same value on every row. |
| `warning` | Image-level quality flag (`True`/`False`) | `True` if any luminescent object exceeded `NOISE_THRESHOLD_MAX`. Those objects are excluded. Same value on every row. |

Limitations to consider improving on:
- The number of rows equals the number of rosettes *detected*. A missing plant shifts the row/column labels, and empty positions are not reported.
- Each image pair gets its own CSV. There is no combined table, and no sample or DPI column.
- The luminescence measures are per image (`global intensity`) or based on spot counts and areas. There is no per-plant integrated or mean intensity.

## 2. Overlay image: `debug_<sample>_NNN.png`

One file per image pair, the same size as the RGB image. It is a visual check of the segmentation and detection.

| Element | Appearance | Meaning |
|---|---|---|
| Background | Greyed out (flat grey outside the crop window) | Not plant |
| Plant pixels | Original RGB colours, at 50% weight | Rosette mask |
| Blue circles (thick) | Enclosing circle of each rosette, +5 px | Detected rosettes |
| Luminescence | ES image (histogram-equalised, warped onto RGB) in the colour `LUMINESCENCE_BGR` | Bacterial signal |
| Red circles (thin) | Enclosing circle of each luminescent spot, blended at `LUMINESCENCE_ALPHA` | Detected spots (hydathodes / lesions) |

Not shown: the rosette outlines, row/column labels, the split between yellow and green tissue, and the values from the CSV.

## 3. Calibration matrix: an entry in the `.env` file (from `calibrator.py`)

| Item | Content |
|---|---|
| `CALIBRATION='[[…]]'` | 3×3 perspective-transform matrix mapping ES image coordinates onto RGB image coordinates. It is written as a NumPy-printed string. |

- The matrix comes from the 4 corner dots of a 4×11 asymmetric dot-grid calibration sheet, imaged with both cameras.
- An overlay of the warped ES on the RGB is shown on screen as a check, but not saved.

## 4. Console messages (not saved)

| Message | When |
|---|---|
| `analyzing images` / `writing csv data` | for each image pair |
| `creating csv directory` | `CSV_DIR` did not exist |
| `too many rosettes: restarting analysis` | more than `ROWS*COLUMNS` rosettes found; the RGB image is re-processed with gamma/saturation correction |
| `too many contours: merging` | the enclosing circles of some rosettes overlap, so contours are merged |
| `ERROR: Incorrect image path …` | an image failed to load; the whole batch stops |

## Not produced
- No combined or summary table across samples or time points.
- No saved intermediate masks.
- No log file.
- No plots.

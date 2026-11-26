

### List of issues Claude came across while reading code


- [ES data] No raw sensor data available: all copies (`…_RawFiles`, `images_CR/`, `ORIGINALS_NOT-annotated/`) are identical 8-bit LemnaTec display exports. They are clipped at a floor of 2, saturate at 255 in 3 images, carry a gamma tag of 0.45455 (possibly non-linear) and have no exposure metadata. The calibration `.tif` is the 8-bit PNG × 257. → Ask the LemnaTec operator for a linear 16-bit export and whether exposure/display settings were constant.
- [ES data] Clipped floor (min = mode = 2 in all 44 images) → background noise can't be estimated; the σ-from-below estimate is always 0 and the cutoff falls back to `min_sigma` (10% of mode, which ties it to an export offset rather than a signal level).
- [ES data] Saturation: C01 tray2 7 DPI, C02 tray1 14 DPI and C02 tray7 7 DPI reach 233–255 → `lum_max` and intensity sums are underestimated there.
- [Digital_phenotyper `image.py`] `equalizeHist` before the fixed threshold 127 → the effective threshold differs per image; spot results can't be compared across images.
- [Digital_phenotyper `image.py`] Spot "area" = non-zero pixels in the spot's bounding box of the equalised image ≈ bbox area, not blob area.
- [Digital_phenotyper] Row/column come from sorting the detected rosettes → a missing plant shifts all labels; empty positions are not reported.
- [`arafromtop/postprocessing.py`, old version] `process_segmask_directory` overwrote its `config4` argument with a hardcoded config; the seg dir lacked `_2` (older seg run?); the seed mask path was relative; `n_labels` was the highest label instead of the plant count; `imshow` had no `vmin/vmax` (mask/text colours mismatch); no error handling in the batch loop. (Fixed in the rewrite.)
- [`pipelineclean_phase4_postprocessing_exploratory.py`] The comment on `min_size=600` ("diameter 7 → 572") describes the dilated area; the new cleanup measures undilated area, so the threshold is much stricter than the comment suggests.
- [`dev/angle_correction_exploratory.py`] Calibration sheet doesn't cover the full field → ~10–17 px extrapolation error at the image edges; luminescence is visibly slightly misaligned (worked around with expanded plant ROIs).
- [`color_classification.py`] The old 10×10 open/close removes ~all yellow on unblurred pixels (21 DPI: 1.1% → 0.02%); the old tool blurred first (`BLUR=10`). Default set to 3 → needs visual tuning.
- [`luminescence.py`] At 7 DPI most hydathode spots are single ES pixels, so `min_spot_size=1`; singleton noise can't be removed by size without also losing them.
- [`luminescence.py`] Many spots lie 40–270 RGB px from any plant (7 DPI tray1: 45 of 55) → off-plant signal or noise, counted in `n_spots_unassigned`; check what they are.
- [`luminescence.py`] `spot_median_dimmest` is usually 2 (= cutoff 3 + 1 − mode 2) → it mostly reflects the detection limit.
- [`plotting.py`] Matplotlib's default `image.composite_image=True` shifted the full-size RGB image relative to the contours in the overview PDF when an overlay image was in the same axes. Fixed by setting it to False in `FIG_RC`.
- [`plotting.py`] Overview PDFs are ~19 MB each (3 full-resolution images) → consider downsampling the display images.
- [Results] Several 7 DPI trays have only 8–9 plants (C01 tray5–8). Many trays have exactly 12 (maybe by design). Check whether small 7 DPI seedlings are removed by `min_size_cluster=600` / `min_size_wholeplant=400`.
- [`*_seg.npz`] `prepr_info` is an object array → loading all keys fails without `allow_pickle`; load only `img_pred_lbls`.

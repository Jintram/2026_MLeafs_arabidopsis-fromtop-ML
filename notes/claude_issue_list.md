

### List of issues Claude came across while reading code


- [ES data] No raw sensor data available: all copies (`…_RawFiles`, `images_CR/`, `ORIGINALS_NOT-annotated/`) are identical 8-bit LemnaTec display exports. They are clipped at a floor of 2, saturate at 255 in 3 images, carry a gamma tag of 0.45455 (possibly non-linear) and have no exposure metadata. The calibration `.tif` is the 8-bit PNG × 257. → Ask the LemnaTec operator for a linear 16-bit export and whether exposure/display settings were constant.
"""Convert tiffs from given directory to npz files.

Example usage:
python convert_tifs_to_npz.py /path/to/your/files tif
"""

################################################################################
# %% libs

import argparse
import glob
from pathlib import Path

import imageio.v3 as iio
import numpy as np

################################################################################
# %%

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all images with a given extension in a directory to .npz files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing image files.",
    )
    parser.add_argument(
        "extension",
        type=str,
        help="Image extension (e.g., tif, .tif, TIF).",
    )
    return parser.parse_args()

################################################################################
# %%

# Get input arguments
args = parse_args()
input_dir = args.input_dir
extension = args.extension
if not extension.startswith("."): 
    extension = f".{extension}"

# Collect files
file_list = [Path(x) for x in glob.glob(str(input_dir) + "/*" + extension.lower())]
print(f"Found {len(file_list)} files with extension {extension} in {input_dir}.")

# Create output dir
output_dir = (input_dir.parent / f"{input_dir.name}_NPZ")
output_dir.mkdir(parents=True, exist_ok=True)

# Now save them again
for src in file_list:

    # Load
    img = iio.imread(src)

    # Save    
    np.savez_compressed(
        output_dir / f"{src.stem}.npz",
        image=img
    )

################################################################################
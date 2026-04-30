


## Notes on what I did


### 30-04-2026

- Obtained annotation files for 11 RGB images of plants.
    - Only leafs where annotated
    - Where delivered in tif format. I converted them from tif to npy using
    `helpers/convert_tifs_to_npy.py` (custom script). Then copied and renamed
    them (`_tile_seg` instead `_plantmask` suffix) as required to be used as 
    segmentation input. 
    - See folder `Training_set_Ara_Xantho_202604/` for original annotations,
    matching images I collected in `ORIGINALS_sel_202604/`. Conversion can be 
    found in `TRAINING-singlelayer/`
- Created separate soil annotation for these files myself;
    - By creating a new annotation script `advanced/phase1_annotate-soil.py`
    - See folder `Training_set_Ara_Xantho_202604_soil_MW/`
- Merged the two annotations using
    - `advanced/merging-two-annotations.py`
    - See folder `./TRAINING_DATA/TRAINING_2layers/`    
- Finalized training folder by
    - going over all images with `pipelineclean_phase1_example_arabtop.py`
    - See folder `./TRAINING_DATA/TRAINING_2layers/`

# %%

import numpy as np
import napari

labels = np.zeros((256, 256), dtype=np.uint16)

viewer = napari.Viewer()
viewer.add_labels(labels, name="Labels-empty")

napari.run()
import numpy as np


filepath = "/data/dataset/AnyScaleGrasp/BimanBODex/shadow/right_full/core_bottle_47ebee26ca51177ac3fdab075b98c5d8/tabletop_ur10e/scale012_pose002_0.npy"
data = np.load(filepath, allow_pickle=True).item()

a = 1

import numpy as np

fx = 1415.46202
fy = 1425.59895
cx = 1205.72982
cy = 1021.93691

K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float64)

#src\two_view_vo\camera_4seasons.py
import numpy as np

fx = 501.4757919305817
fy = 501.4757919305817
cx = 421.7953735163109
cy = 167.65799492501083

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)

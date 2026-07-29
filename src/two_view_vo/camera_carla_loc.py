#src\two_view_vo\camera_carla_loc.py
import numpy as np

# CARLA-Loc 相机参数（来自论文）
# Resolution: 1280 x 720
# Horizontal FOV: 90 degrees
# No distortion

fx = 640.0
fy = 640.0
cx = 640.0
cy = 360.0

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,   0,  1]
], dtype=np.float64)
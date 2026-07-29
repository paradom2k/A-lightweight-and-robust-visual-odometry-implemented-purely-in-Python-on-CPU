#src\two_view_vo\camera_spray
import numpy as np

# Image resolution
W = 2048
H = 1088

# Recommended intrinsics for statistical Sampson residual analysis
fx = fy = W   # 2048
cx = W / 2.0  # 1024
cy = H / 2.0  # 544

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,   0,  1]
], dtype=np.float64)

print("K =\n", K)


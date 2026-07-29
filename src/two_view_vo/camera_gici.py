# src/two_view_vo/camera_gici.py
#
# GICI-LIB 数据集相机参数（Onsemi MT9V034，752×480）
# 原始畸变参数来自 intrinsics_and_extrinsics.yaml:
#   k1=-0.36472, k2=0.11530, p1=0.00060, p2=0.00202
#
# 本文件中的内参为去畸变后的新内参，
# 由 cv2.getOptimalNewCameraMatrix(alpha=0) 计算得到，
# 与解码脚本 decode_gici_camera.py 中的去畸变操作完全对应。
# alpha=0 表示裁掉黑边、保留全部有效像素区域。

import numpy as np

fx = 388.3139325727301
fy = 464.0774555090246
cx = 351.6387159287324
cy = 240.4100043687178

K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float64)
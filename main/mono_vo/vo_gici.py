# vo_4seasons.py
import cv2
import glob
import os
from src.two_view_vo.camera_gici import K
from src.two_view_vo.vo_pipeline import VisualOdometry

# ==============================
# 图像路径
# ==============================
img_dir = r"G:\GICI\gici_dataset_open\1.1\camera"
img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

# ==============================
# 初始化 VO
# ==============================
vo = VisualOdometry(K)

# ==============================
# 主循环
# ==============================
for img_path in img_files:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    vo.process_frame(img)

    if vo.traj is not None:
        cv2.imshow("Trajectory", vo.traj)
    if vo.vis is not None:
        cv2.imshow("Feature Tracking", vo.vis)

    key = cv2.waitKey(1)
    if key == 27:
        break

# ==============================
# 保存最终轨迹图
# ==============================
cv2.imwrite("../trajectory_final.png", vo.traj)
print("最终轨迹图已保存为 trajectory_final.png")

cv2.destroyAllWindows()

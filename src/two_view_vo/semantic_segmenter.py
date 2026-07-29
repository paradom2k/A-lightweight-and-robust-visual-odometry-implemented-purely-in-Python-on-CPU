# src/two_view_vo/semantic_segmenter.py
import numpy as np
import cv2
import torch
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)

# torchvision LR-ASPP MobileNetV3-Large 使用的是 COCO-with-VOC-labels 21类
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# 语义静态类别集合 C_static：偏"固定家具/背景"类视为静态，
# 生物与交通工具类视为潜在动态（可按你的实际场景调整这个集合）
STATIC_CLASS_NAMES = {
    "background", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",
}
STATIC_CLASS_IDS = [i for i, name in enumerate(VOC_CLASSES) if name in STATIC_CLASS_NAMES]


class SemanticSegmenter:
    """
    轻量语义分割器（LR-ASPP + MobileNetV3-Large, CPU）。
    每帧在降采样图像上跑一次前向，输出静态类别置信度图 p_sem(x,y)，
    对应 SGFR 公式中的 p_sem(p_l^j) = sum_{c in C_static} P(c | I_j, p_l^j)。
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        self.model = lraspp_mobilenet_v3_large(weights=weights)
        self.model.eval().to(device)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.last_p_sem = None  # 缓存最近一次静态置信度图

    @torch.no_grad()
    def compute_static_confidence(self, img):
        """
        输入：降采样后的图像（灰度或BGR），与 feature_tracker.add_new_features
        中使用的 downsample_half 输出尺度一致。
        输出：p_sem，形状 (H, W)，取值范围 [0, 1]。
        """
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - self.mean) / self.std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        out = self.model(tensor)["out"]              # (1, 21, H, W)
        probs = torch.softmax(out, dim=1)[0]           # (21, H, W)
        static_probs = probs[STATIC_CLASS_IDS].sum(dim=0)  # (H, W)

        p_sem = static_probs.cpu().numpy().astype(np.float32)
        self.last_p_sem = p_sem
        return p_sem

    @staticmethod
    def sample(p_sem_map, pts_xy):
        """
        在 p_sem_map 上按最近邻采样一批像素坐标 (N,2)=(x,y)。
        pts_xy 必须已经映射到 p_sem_map 所在的降采样坐标系。
        """
        if p_sem_map is None or len(pts_xy) == 0:
            return np.ones(len(pts_xy), dtype=np.float32)

        h, w = p_sem_map.shape
        xs = np.clip(np.round(pts_xy[:, 0]).astype(np.int32), 0, w - 1)
        ys = np.clip(np.round(pts_xy[:, 1]).astype(np.int32), 0, h - 1)
        return p_sem_map[ys, xs]
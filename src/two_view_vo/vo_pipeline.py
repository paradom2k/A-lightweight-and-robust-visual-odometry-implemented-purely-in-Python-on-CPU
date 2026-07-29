# src/two_view_vo/vo_pipeline.py
import cv2
import numpy as np
from src.two_view_vo.feature_tracker import FeatureTracker
from src.two_view_vo.semantic_segmenter import SemanticSegmenter
from src.two_view_vo.vo_utils import (
    decompose_E_and_choose,
    draw_trajectory,
    downsample_half,
    compute_epipolar_deviation,
    compute_sgfr_weights,
    weighted_fundamental_matrix,
)


class VisualOdometry:
    def __init__(self, K, sgfr_nu=5.0, sgfr_sigma=0.01, use_sgfr=True,
                 log_weights=False, weight_log_path="sgfr_weights.txt"):
        self.K = K
        self.tracker = FeatureTracker()

        self.R_f = np.eye(3)
        self.t_f = np.zeros((3, 1))
        self.positions = []

        self.traj = None
        self.vis = None
        self.scale_initialized = False
        self.ref_median_depth = 1.0
        self.last_scale = 1.0

        # ---------- SGFR 相关 ----------
        self.use_sgfr = use_sgfr
        self.sgfr_nu = sgfr_nu       # Student-t 自由度
        self.sgfr_sigma = sgfr_sigma  # 极线偏差尺度参数（归一化平面下，需按相机标定调）
        self.semantic_segmenter = SemanticSegmenter(device="cpu") if use_sgfr else None
        self.last_p_sem_map = None
        self.last_sgfr_weights = None  # 供外部调试/可视化查看
        # ---------- 权重观察日志 ----------
        self.log_weights = log_weights
        self.frame_idx = 0
        self.weight_log_file = None
        if self.log_weights:
            self.weight_log_file = open(weight_log_path, "w", encoding="utf-8")
            self.weight_log_file.write(
                "frame_idx feature_idx x_prev y_prev x_next y_next p_sem e_g weight\n"
            )

    def process_frame(self, img):
        h, w = img.shape

        # ---------- 每帧在降采样图上跑一次语义分割 ----------
        p_sem_map = None
        if self.use_sgfr:
            small_img = downsample_half(img)
            p_sem_map = self.semantic_segmenter.compute_static_confidence(small_img)
            self.last_p_sem_map = p_sem_map

        res = self.tracker.track(img)
        good_prev = res.good_prev
        good_next = res.good_next
        good_track_cnt = res.good_track_cnt
        good_ids = res.good_ids

        # RANSAC 初始估计
        if len(good_prev) >= 32:
            F, mask = cv2.findFundamentalMat(
                good_prev, good_next, cv2.FM_RANSAC, 1, 0.99
            )
            if F is not None and mask is not None:
                mask = mask.reshape(-1).astype(bool)
                good_prev = good_prev[mask]
                good_next = good_next[mask]

                E = self.K.T @ F @ self.K
                if np.linalg.matrix_rank(E) >= 2:

                    # ---------- SGFR: 语义-几何联合特征加权 ----------
                    if self.use_sgfr and p_sem_map is not None and len(good_next) >= 8:
                        pts_small = good_next / 2.0  # 映射到降采样语义图坐标
                        p_sem = self.semantic_segmenter.sample(p_sem_map, pts_small)
                        e_g = compute_epipolar_deviation(E, good_prev, good_next, self.K)
                        w_sgfr = compute_sgfr_weights(
                            p_sem, e_g, nu=self.sgfr_nu, sigma=self.sgfr_sigma
                        )
                        self.last_sgfr_weights = w_sgfr
                        if self.use_sgfr and p_sem_map is not None and len(good_next) >= 8:
                            pts_small = good_next / 2.0
                            p_sem = self.semantic_segmenter.sample(p_sem_map, pts_small)
                            e_g = compute_epipolar_deviation(E, good_prev, good_next, self.K)
                            w_sgfr = compute_sgfr_weights(
                                p_sem, e_g, nu=self.sgfr_nu, sigma=self.sgfr_sigma
                            )
                            self.last_sgfr_weights = w_sgfr

                            # ---------- 新增：把每个特征的权重写到txt ----------
                            if self.log_weights and self.weight_log_file is not None:
                                for i in range(len(w_sgfr)):
                                    self.weight_log_file.write(
                                        f"{self.frame_idx} {i} "
                                        f"{good_prev[i, 0]:.2f} {good_prev[i, 1]:.2f} "
                                        f"{good_next[i, 0]:.2f} {good_next[i, 1]:.2f} "
                                        f"{p_sem[i]:.4f} {e_g[i]:.6f} {w_sgfr[i]:.6f}\n"
                                    )
                                self.weight_log_file.flush()

                            F_star = weighted_fundamental_matrix(good_prev, good_next, w_sgfr)
                            if F_star is not None:
                                E_star = self.K.T @ F_star @ self.K
                                if np.linalg.matrix_rank(E_star) >= 2:
                                    F, E = F_star, E_star

                        F_star = weighted_fundamental_matrix(good_prev, good_next, w_sgfr)
                        if F_star is not None:
                            E_star = self.K.T @ F_star @ self.K
                            if np.linalg.matrix_rank(E_star) >= 2:
                                F, E = F_star, E_star

                    R, t = decompose_E_and_choose(E, good_prev, good_next, self.K)
                    if R is not None:
                        s = self._estimate_scale_from_depth(R, t, good_prev, good_next)
                        t_scaled = t * s
                        self.t_f = self.t_f + self.R_f @ t_scaled
                        self.R_f = R @ self.R_f

                        x, z = self.t_f[0, 0], self.t_f[2, 0]
                        self.positions.append((x, z))
                        self.traj = draw_trajectory(self.positions)

        # 特征补充
        cur_pts, track_cnt, ids = self.tracker.add_new_features(
            img, good_next, good_track_cnt, good_ids
        )

        # 可视化
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pt in cur_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        self.vis = vis

    def _estimate_scale_from_depth(self, R, t, pts1, pts2):
        if len(pts1) < 10:
            return float(self.last_scale)

        disp = np.linalg.norm(pts1 - pts2, axis=1)
        parallax_mask = disp > 3
        if np.sum(parallax_mask) < 10:
            return float(self.last_scale)

        pts1_sel = pts1[parallax_mask]
        pts2_sel = pts2[parallax_mask]

        P0 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = self.K @ np.hstack((R, t))

        X_h = cv2.triangulatePoints(P0, P1, pts1_sel.T, pts2_sel.T)
        X = X_h[:3] / (X_h[3] + 1e-8)

        z = X[2, :]
        depth_mask = (z > 0.5) & (z < 200)
        z_valid = z[depth_mask]

        if len(z_valid) < 10:
            return float(self.last_scale)

        median_depth = float(np.median(z_valid))
        if median_depth <= 1e-6:
            return float(self.last_scale)

        if not self.scale_initialized:
            self.ref_median_depth = median_depth
            self.scale_initialized = True
            self.last_scale = 1.0
            return 1.0

        s_raw = self.ref_median_depth / median_depth
        s_clamped = max(min(s_raw, 1000.0), 0.05)

        alpha = 0.3
        s = (1.0 - alpha) * self.last_scale + alpha * s_clamped
        self.last_scale = float(s)

        return float(s)
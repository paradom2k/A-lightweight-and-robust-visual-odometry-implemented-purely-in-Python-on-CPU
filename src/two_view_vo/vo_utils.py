# src/two_view_vo/vo_utils.py
import numpy as np
import cv2

TRAJ_H, TRAJ_W = 800, 800


def downsample_half(img):
    """统一的半分辨率降采样，供角点检测与语义分割共用同一尺度。"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2))


def decompose_E_and_choose(E, pts1, pts2, K):
    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]
    E = U @ np.diag(S) @ Vt
    U, _, Vt = np.linalg.svd(E)

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
        t = -t
    if np.linalg.det(R2) < 0:
        R2 = -R2

    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    K_inv = np.linalg.inv(K)

    best_R, best_t, best_count = None, None, -1
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    for R, t_vec in candidates:
        t_vec = t_vec.reshape(3, 1)
        P1 = K @ np.hstack((R, t_vec))

        X_h = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
        X = X_h[:3] / (X_h[3] + 1e-8)

        z1 = X[2]
        z2 = (R @ X + t_vec)[2]

        count = np.sum((z1 > 0) & (z2 > 0))
        if count > best_count:
            best_count = count
            best_R = R
            best_t = t_vec

    return best_R, best_t


# ----------------------------------------------------------------------
# SGFR: Semantic-Geometric Feature Reweighting
# ----------------------------------------------------------------------

def compute_epipolar_deviation(E, pts1, pts2, K):
    """
    e_g(l) = | \bar{p}_l^{c_j,T} E \bar{p}_l^{c_i} |
    pts1: host帧 i 的像素坐标 (N,2)   pts2: 当前帧 j 的像素坐标 (N,2)
    """
    K_inv = np.linalg.inv(K)

    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # (3,N)
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T

    x1 = K_inv @ pts1_h
    x2 = K_inv @ pts2_h

    # 归一化到单位球（bearing vector）
    x1 = x1 / (np.linalg.norm(x1, axis=0, keepdims=True) + 1e-12)
    x2 = x2 / (np.linalg.norm(x2, axis=0, keepdims=True) + 1e-12)

    Ex1 = E @ x1                              # (3,N)
    e_g = np.abs(np.sum(x2 * Ex1, axis=0))    # (N,)
    return e_g


def compute_sgfr_weights(p_sem, e_g, nu=5.0, sigma=0.01):
    """
    w_l = p_sem(l) * (1 + e_g(l)^2 / (nu * sigma^2))^(-(nu+1)/2)
    对应论文 Eq. (likelihood) x P(z_l=1)
    """
    p_sem = np.clip(p_sem.astype(np.float64), 1e-6, 1.0)
    likelihood = (1.0 + (e_g.astype(np.float64) ** 2) / (nu * sigma ** 2)) ** (-(nu + 1.0) / 2.0)
    w = p_sem * likelihood
    return w.astype(np.float32)


def _hartley_normalize(pts):
    """质心归零 + 平均距离 sqrt(2) 的各向同性归一化，提升DLT数值稳定性。"""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / mean_dist if mean_dist > 1e-8 else 1.0

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def weighted_fundamental_matrix(pts1, pts2, weights):
    """
    SGFR加权8点法求解 F*（论文 Eq. wfundamental）：
        diag(sqrt(w_l)) A f* = 0
    内部做 Hartley 归一化以保证数值稳定，返回反归一化后的 F。
    """
    N = pts1.shape[0]
    if N < 8:
        return None

    pts1_n, T1 = _hartley_normalize(pts1)
    pts2_n, T2 = _hartley_normalize(pts2)

    x1, y1 = pts1_n[:, 0], pts1_n[:, 1]
    x2, y2 = pts2_n[:, 0], pts2_n[:, 1]

    A = np.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1,      y1,      np.ones_like(x1),
    ], axis=1)  # (N, 9)

    sqrt_w = np.sqrt(np.clip(weights, 1e-8, None)).reshape(-1, 1)
    Aw = A * sqrt_w

    _, _, Vt = np.linalg.svd(Aw)
    F_n = Vt[-1].reshape(3, 3)

    # rank-2 约束
    U, S, Vt2 = np.linalg.svd(F_n)
    S[2] = 0
    F_n = U @ np.diag(S) @ Vt2

    # 反归一化回原始像素坐标系
    F = T2.T @ F_n @ T1

    if abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]

    return F


def draw_trajectory(positions):
    img = np.ones((TRAJ_H, TRAJ_W, 3), dtype=np.uint8) * 255
    if len(positions) < 2:
        return img

    xs = np.array([p[0] for p in positions])
    zs = np.array([p[1] for p in positions])

    min_x, max_x = xs.min(), xs.max()
    min_z, max_z = zs.min(), zs.max()

    margin = 40
    scale_x = (TRAJ_W - 2 * margin) / max(max_x - min_x, 1e-6)
    scale_z = (TRAJ_H - 2 * margin) / max(max_z - min_z, 1e-6)
    scale = min(scale_x, scale_z)

    center_x = (max_x + min_x) / 2
    center_z = (max_z + min_z) / 2

    for x, z in positions:
        u = int((x - center_x) * scale + TRAJ_W / 2)
        v = int((z - center_z) * scale + TRAJ_H / 2)
        cv2.circle(img, (u, v), 2, (0, 0, 255), -1)

    return img
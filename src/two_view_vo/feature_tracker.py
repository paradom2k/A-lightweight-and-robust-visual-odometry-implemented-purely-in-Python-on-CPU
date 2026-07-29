# src/two_view_vo/feature_tracker.py
import cv2
import numpy as np
from src.two_view_vo.vo_utils import downsample_half
MAX_CNT = 200
MIN_DIST = 15
WIN_SIZE = (50, 50)

# SSPC scale-space configuration
SSPC_SCALES = [1.0, 1.5, 2.0, 3.0, 4.0]
SSPC_BLOCK_SIZE = 3
SSPC_KSIZE = 3
SSPC_TAU_REL = 0.001  # relative threshold for corner strength at finest scale


class FeatureTrackResult:
    """
    统一返回结构，兼容旧代码（通过属性访问）
    """
    def __init__(self, good_prev, good_next, good_track_cnt, good_ids, good_err):
        self.good_prev = good_prev
        self.good_next = good_next
        self.good_track_cnt = good_track_cnt
        self.good_ids = good_ids
        self.good_err = good_err


def _compute_sspc_corner_responses(img, scales):
    """
    在多尺度高斯金字塔上计算 Shi-Tomasi 角点响应 R_s(p) = min(lambda1, lambda2)。

    使用 cv2.cornerMinEigenVal 直接得到最小特征值，对应 R_s。
    """
    responses = {}
    for s in scales:
        # 高斯平滑构造尺度空间 I_s = G(s) * I
        # 这里将尺度 s 映射为核大小，简单采用 ksize = int(2 * s + 1)
        ksize = max(3, int(2 * s + 1) | 1)  # 保证为奇数且至少 3
        blurred = cv2.GaussianBlur(img, (ksize, ksize), s)

        # Shi-Tomasi 最小特征值响应
        R_s = cv2.cornerMinEigenVal(
            blurred,
            blockSize=SSPC_BLOCK_SIZE,
            ksize=SSPC_KSIZE
        )
        responses[s] = R_s
    return responses


def _sspc_select_corners(img, mask, max_corners, min_dist):
    """
    Scale-Space Persistent Corner Selection (SSPC) + Shi-Tomasi 角点提取。

    步骤：
    1. 使用 Shi-Tomasi 在最细尺度上检测初始候选角点集合 P。
    2. 在多尺度上计算角点响应 R_s(p)。
    3. 对每个候选点计算尺度持久度 s*(p) 和综合评分 S(p) = s*(p) * R_{s_min}(p)。
    4. 按 S(p) 降序排序，并在距离约束下选取前 max_corners 个点。
    """
    if max_corners <= 0:
        return None

    # 初始候选角点（CornerDetector）
    initial_corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=max_corners * 4,  # 多取一些，后面再筛选
        qualityLevel=0.01,
        minDistance=min_dist,
        mask=mask
    )
    if initial_corners is None:
        return None

    pts = initial_corners.reshape(-1, 2)
    h, w = img.shape

    # 计算多尺度角点响应 R_s
    responses = _compute_sspc_corner_responses(img, SSPC_SCALES)
    s_min = SSPC_SCALES[0]
    R_min = responses[s_min]

    # 计算阈值 tau（基于最细尺度的最大响应）
    max_R_min = float(np.max(R_min)) if R_min.size > 0 else 0.0
    if max_R_min <= 0.0:
        return None
    tau = SSPC_TAU_REL * max_R_min

    scores = []
    s_star_list = []

    for p in pts:
        x = int(round(p[0]))
        y = int(round(p[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            scores.append(0.0)
            s_star_list.append(0.0)
            continue

        # 初始尺度响应 R_{s_min}(p)
        R_smin_p = float(R_min[y, x])
        if R_smin_p <= tau:
            scores.append(0.0)
            s_star_list.append(0.0)
            continue

        # 计算尺度持久度 s*(p)
        s_star = s_min
        for s in SSPC_SCALES:
            R_s_map = responses[s]
            R_sp = float(R_s_map[y, x])
            if R_sp > tau:
                s_star = s
            else:
                break

        s_star_list.append(s_star)
        scores.append(s_star * R_smin_p)
    # ============================
    # ★★★ 调试输出：统计每个 s* 的数量 ★★★
    # ============================
    s_star_arr = np.array(s_star_list)
    unique_s, counts = np.unique(s_star_arr, return_counts=True)

    print("\n[SSPC DEBUG] Scale Persistence Extent 分布：")
    for s_val, cnt in zip(unique_s, counts):
        print(f"  s* = {s_val:.1f}  →  {cnt} 个特征点")
    print("--------------------------------------------------\n")

    scores = np.array(scores, dtype=np.float32)
    s_star_list = np.array(s_star_list, dtype=np.float32)

    # 去掉评分为 0 的伪特征
    valid_mask = scores > 0.0
    if not np.any(valid_mask):
        return None

    pts_valid = pts[valid_mask]
    scores_valid = scores[valid_mask]

    # 按评分 S(p) 降序排序
    order = np.argsort(-scores_valid)
    pts_sorted = pts_valid[order]

    # 距离约束下的最终选点（类似 goodFeaturesToTrack 的 minDistance）
    selected = []
    used_mask = mask.copy()

    for p in pts_sorted:
        if len(selected) >= max_corners:
            break
        x = int(round(p[0]))
        y = int(round(p[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        # 检查当前 mask 是否允许
        if used_mask[y, x] == 0:
            continue

        selected.append(p)
        # 在 mask 上画圆，抑制邻近区域
        cv2.circle(used_mask, (x, y), min_dist, 0, -1)

    if len(selected) == 0:
        return None

    return np.array(selected, dtype=np.float32).reshape(-1, 1, 2)


class FeatureTracker:
    def __init__(self):
        self.prev_img = None
        self.prev_pts = np.empty((0, 2), dtype=np.float32)
        self.track_cnt = np.array([], dtype=np.int32)
        self.ids = np.array([], dtype=np.int32)
        self.next_id = 0

    def track(self, img):
        h, w = img.shape

        # 第一帧：返回空
        if self.prev_img is None or len(self.prev_pts) == 0:
            self.prev_img = img.copy()
            return FeatureTrackResult(
                np.empty((0, 2), np.float32),
                np.empty((0, 2), np.float32),
                np.array([], np.int32),
                np.array([], np.int32),
                np.array([], np.float32)
            )

        # LK 光流
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img,
            self.prev_pts.reshape(-1, 1, 2), None,
            WIN_SIZE, maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
        )

        # LK 失败：清空
        if next_pts is None or status is None or err is None:
            self.prev_img = img.copy()
            self.prev_pts = np.empty((0, 2), np.float32)
            self.track_cnt = np.array([], np.int32)
            self.ids = np.array([], np.int32)
            return FeatureTrackResult(
                np.empty((0, 2), np.float32),
                np.empty((0, 2), np.float32),
                np.array([], np.int32),
                np.array([], np.int32),
                np.array([], np.float32)
            )

        status = status.reshape(-1)
        err = err.reshape(-1)

        # ============================
        # 强制所有数组长度对齐
        # ============================
        N = min(len(self.prev_pts), len(self.track_cnt), len(self.ids),
                len(status), len(next_pts), len(err))

        if N == 0:
            self.prev_img = img.copy()
            self.prev_pts = np.empty((0, 2), np.float32)
            self.track_cnt = np.array([], np.int32)
            self.ids = np.array([], np.int32)
            return FeatureTrackResult(
                np.empty((0, 2), np.float32),
                np.empty((0, 2), np.float32),
                np.array([], np.int32),
                np.array([], np.int32),
                np.array([], np.float32)
            )

        prev_pts = self.prev_pts[:N]
        track_cnt = self.track_cnt[:N]
        ids = self.ids[:N]
        next_pts = next_pts[:N]
        status = status[:N]
        err = err[:N]

        # 只保留 status == 1
        good_mask = (status == 1)
        good_prev = prev_pts[good_mask]
        good_next = next_pts.reshape(-1, 2)[good_mask]
        good_track_cnt = track_cnt[good_mask]
        good_ids = ids[good_mask]
        good_err = err[good_mask]

        # 边界过滤
        if len(good_next) > 0:
            mask_border = (
                (good_next[:, 0] >= 1) & (good_next[:, 0] < w - 1) &
                (good_next[:, 1] >= 1) & (good_next[:, 1] < h - 1)
            )
            good_prev = good_prev[mask_border]
            good_next = good_next[mask_border]
            good_track_cnt = good_track_cnt[mask_border]
            good_ids = good_ids[mask_border]
            good_err = good_err[mask_border]

        return FeatureTrackResult(
            good_prev, good_next, good_track_cnt, good_ids, good_err
        )

    def add_new_features(self, img, good_next, good_track_cnt, good_ids):
        h, w = img.shape
        mask = np.ones_like(img, dtype=np.uint8) * 255

        for pt in good_next:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), MIN_DIST, 0, -1)

        n_needed = MAX_CNT - len(good_next)
        new_pts = []

        if n_needed > 0:
            # 使用与语义分割共用的降采样函数，保证尺度一致
            small_img = downsample_half(img)
            small_mask = downsample_half(mask)

            small_corners_sspc = _sspc_select_corners(
                small_img,
                small_mask,
                max_corners=n_needed,
                min_dist=MIN_DIST // 2
            )
            if small_corners_sspc is not None:
                new_pts.extend((small_corners_sspc.reshape(-1, 2) * 2))

            if len(new_pts) < n_needed:
                remain = n_needed - len(new_pts)
                corners_full_sspc = _sspc_select_corners(
                    img,
                    mask,
                    max_corners=remain,
                    min_dist=MIN_DIST
                )
                if corners_full_sspc is not None:
                    new_pts.extend(corners_full_sspc.reshape(-1, 2))



        if len(new_pts) > 0:
            new_pts = np.array(new_pts, dtype=np.float32)
            new_ids = np.arange(self.next_id, self.next_id + len(new_pts))
            new_track_cnt = np.ones(len(new_pts), dtype=np.int32)
            self.next_id += len(new_pts)

            cur_pts = np.vstack((good_next, new_pts))
            track_cnt = np.hstack((good_track_cnt + 1, new_track_cnt))
            ids = np.hstack((good_ids, new_ids))
        else:
            cur_pts = good_next
            track_cnt = good_track_cnt + 1
            ids = good_ids

        self.prev_img = img.copy()
        self.prev_pts = cur_pts.astype(np.float32)
        self.track_cnt = track_cnt
        self.ids = ids

        return cur_pts, track_cnt, ids

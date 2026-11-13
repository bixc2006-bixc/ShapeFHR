from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm


# =========================
# 一、候选形状子生成（auto_piss_extractor）
# =========================

def pd_distance(
    point: Tuple[float, float],
    line_p1: Tuple[float, float],
    line_p2: Tuple[float, float],
) -> float:
    """
    对应源码 pd_distance：计算点到直线段（两端点定义直线）的垂直距离。
    """
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + (x2 * y1 - y2 * x1))
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return float(numerator / denominator)


def auto_piss_extractor(
    ts: np.ndarray,
    num_pip: float = 0.2,
    j: int = -1,
    return_pip: int = 0,
) -> List[np.ndarray] | List[int]:
    """
    严格按用户给定流程实现。
    注意：返回 PIS 时不做归一化。
    """
    del j  # 仅保留接口一致性（源码里常用于打印）

    ts = np.asarray(ts, dtype=np.float64)
    L = ts.shape[0]

    max_no_pip = int(num_pip * L)
    if max_no_pip < 5:
        max_no_pip = 5

    list_index_z = stats.zscore(list(range(L)))

    pips: List[int] = [0, L - 1]
    remain_pos: List[int] = list(range(1, L - 1))
    piss: List[np.ndarray] = []

    for _ in range(max_no_pip - 2):
        biggest_dist = -1.0
        biggest_dist_pos = -1

        for jj in remain_pos:
            p2 = min([x for x in pips if x > jj])
            p3 = max([x for x in pips if x < jj])

            dist = pd_distance(
                (float(list_index_z[jj]), float(ts[jj])),
                (float(list_index_z[p3]), float(ts[p3])),
                (float(list_index_z[p2]), float(ts[p2])),
            )
            if dist > biggest_dist:
                biggest_dist = dist
                biggest_dist_pos = jj

        if biggest_dist_pos < 0:
            break

        pips.append(biggest_dist_pos)
        pips = sorted(pips)
        remain_pos.remove(biggest_dist_pos)

        pos = pips.index(biggest_dist_pos)

        # 新逻辑：每次仅生成“包含新 PIP 的连续三元组”对应候选
        # 若 len(pips) < 3，不生成候选
        if len(pips) >= 3:
            # 左三元组：(pips[pos-2], pips[pos-1], pips[pos]) -> [pips[pos-2], pips[pos]]
            if pos >= 2:
                piss.append(np.array([pips[pos - 2], pips[pos]], dtype=np.int64))

            # 右三元组：(pips[pos], pips[pos+1], pips[pos+2]) -> [pips[pos], pips[pos+2]]
            if pos <= len(pips) - 3:
                piss.append(np.array([pips[pos], pips[pos + 2]], dtype=np.int64))

    if return_pip:
        return pips
    return piss


# =========================
# 二、复杂度（CI）计算
# =========================

def auto_ci_extractor(
    ts: np.ndarray,
    piss: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[float]]:
    """
    对应步骤：
    ts_ci = (ts[:-1] - ts[1:]) ** 2
    ci = sqrt(sum(ts_ci[s:e-1]) + 0.001)
    """
    ts = np.asarray(ts, dtype=np.float64)
    ts_ci = (ts[:-1] - ts[1:]) ** 2

    list_ci_piss: List[float] = []
    for pis in piss:
        s, e = int(pis[0]), int(pis[1])
        if e - 1 <= s:
            val = 0.0
        else:
            val = float(np.sum(ts_ci[s : e - 1]))
        list_ci_piss.append(float(np.sqrt(val + 0.001)))

    return ts_ci, list_ci_piss


# =========================
# 三、PCS 扩展（pcs_extractor）
# =========================

def pcs_extractor(pis: Sequence[int], w: int, len_of_ts: int) -> np.ndarray:
    if w == 0:
        return np.array([int(pis[0]), int(pis[1])], dtype=np.int64)

    start_pos = int(pis[0]) - w + 1
    if start_pos < 0:
        start_pos = 0

    end_pos = int(pis[1]) + w
    if end_pos > len_of_ts:
        end_pos = len_of_ts

    return np.array([start_pos, end_pos], dtype=np.int64)


# =========================
# 四、距离矩阵计算（calculate_matrix）
# =========================

def calculate_matrix(
    t1: np.ndarray,
    t2: np.ndarray,
    w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    严格按步骤中的公式实现（CPU/NumPy版本，作为基准逻辑）。
    """
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)

    L1 = t1.shape[0]
    L2 = t2.shape[0]
    if L1 != L2:
        raise ValueError("calculate_matrix 要求 t1 与 t2 等长（与给定流程一致）")

    matrix_1 = np.full((L1, 2 * w + 1), np.inf, dtype=np.float64)
    matrix_2 = np.full((L2, 2 * w + 1), np.inf, dtype=np.float64)

    list_dist = (t1 - t2) ** 2
    matrix_1[:, w] = list_dist
    matrix_2[:, w] = list_dist

    for i in range(1, w + 1):
        list_dist = (t1[i:] - t2[:-i]) ** 2
        matrix_1[i:, w - i] = list_dist
        matrix_2[:-i, w + i] = list_dist

        list_dist = (t2[i:] - t1[:-i]) ** 2
        matrix_2[i:, w - i] = list_dist
        matrix_1[:-i, w + i] = list_dist

    return matrix_1, matrix_2


# =========================
# 五、最小距离计算（find_min_dist）
# =========================

def find_min_dist(
    pis: Sequence[int],
    pcs: Sequence[int],
    matrix_t1: np.ndarray,
    list_start_pos: np.ndarray,
    list_end_pos: np.ndarray,
    pis_ci: float,
    pcs_ci_list: np.ndarray,
) -> float:
    """
    严格按用户给定公式实现（CPU/NumPy版本）。
    """
    s, e = int(pis[0]), int(pis[1])

    start_col = int(list_start_pos[s])
    end_col = int(list_end_pos[e - 1])

    block = matrix_t1[s:e, start_col:end_col]
    if block.size == 0:
        return float("inf")

    sdist = np.sqrt(np.sum(block, axis=0))

    pcs_ci_list = np.asarray(pcs_ci_list, dtype=np.float64)
    pcs_cs = np.cumsum(pcs_ci_list)

    ww = int(pcs[1] - pcs[0] - (e - s))
    if ww < 0:
        return float("inf")

    d_ci = pcs_cs[-ww - 1 :].copy()
    if ww > 0:
        d_ci[-ww:] = d_ci[-ww:] - pcs_cs[:ww]

    d_ci = np.sqrt(d_ci + 0.001)
    d_ci = np.maximum(d_ci, pis_ci) / np.minimum(d_ci, pis_ci)

    m = min(sdist.shape[0], d_ci.shape[0])
    if m <= 0:
        return float("inf")

    return float(np.min(sdist[:m] * d_ci[:m]))


# =========================
# 六、信息增益（find_best_split_point_and_info_gain）
# =========================

def _entropy_binary(pos: float, neg: float) -> float:
    total = pos + neg
    if total <= 0:
        return 0.0

    p_pos = pos / total
    p_neg = neg / total

    ent = 0.0
    if p_pos > 0:
        ent -= p_pos * np.log2(p_pos)
    if p_neg > 0:
        ent -= p_neg * np.log2(p_neg)
    return float(ent)


def find_best_split_point_and_info_gain(
    list_sub_dist: Sequence[float],
    input_label: Sequence[int],
    target_class: int,
) -> float:
    """
    严格按用户给定流程：
    - 按距离排序
    - 仅在 g1_pos/t_pos 四舍五入到 0.1 后命中 0.2,0.4,... 时计算 IG
    - 取最大值
    """
    pairs = list(zip(input_label, list_sub_dist))
    pairs = sorted(pairs, key=lambda x: x[1])

    t_pos = sum(1 for lb in input_label if lb == target_class)
    total = len(input_label)
    t_neg = total - t_pos

    g1_pos, g1_neg = 0, 0
    g2_pos, g2_neg = t_pos, t_neg

    best_ig = -1.0
    rate = 0.2

    total_entropy = _entropy_binary(float(t_pos), float(t_neg))

    for idx in range(len(pairs) - 1):
        lb, _ = pairs[idx]
        if lb == target_class:
            g1_pos += 1
            g2_pos -= 1
        else:
            g1_neg += 1
            g2_neg -= 1

        if t_pos > 0 and round(g1_pos / t_pos, 1) == rate:
            g1_len = g1_pos + g1_neg
            g2_len = g2_pos + g2_neg

            e1 = _entropy_binary(float(g1_pos), float(g1_neg))
            e2 = _entropy_binary(float(g2_pos), float(g2_neg))

            inforgain = (
                total_entropy
                - (g1_len / total) * e1
                - (g2_len / total) * e2
            )

            if inforgain > best_ig:
                best_ig = float(inforgain)

            rate += 0.2
            if rate > 1.0:
                break

    return float(best_ig)


# =========================
# 七、最终形状子选择（find_c_shapelet_non_overlab）
# =========================

def find_c_shapelet_non_overlab(
    list_ppi: List[List[Any]],
    c: int,
    p: float = 0.1,
    p_inner: float = 0.1,
    len_ts: Optional[int] = None,
) -> List[List[Any]]:
    """
    输入格式：每项至少包含
    [ts_pos, start, end, score, label, dim, ...]
    其中 score 可是 info_gain 或 final_score（本实现中用于 final_score）。
    """
    if len(list_ppi) == 0 or c <= 0:
        return []

    # 按 score（索引3）降序
    sorted_idx = sorted(range(len(list_ppi)), key=lambda i: list_ppi[i][3], reverse=True)

    list_group_shapelet: List[List[Any]] = [list_ppi[sorted_idx[0]]]
    list_pos = {sorted_idx[0]}

    for idx in sorted_idx[1:]:
        s2_obj = list_ppi[idx]
        _, s2, e2, _, _, d2 = s2_obj[:6]

        # 源码中有该长度范围注释，按要求保留（默认不启用）
        # if len_ts is not None:
        #     seg_len = e2 - s2
        #     if not (0 <= seg_len <= 0.2 * len_ts):
        #         continue

        skip = False
        for s1_obj in list_group_shapelet:
            _, s1, e1, _, _, d1 = s1_obj[:6]

            if d1 == d2:
                len1 = max(e1 - s1, 1)
                len2 = max(e2 - s2, 1)

                if s1 <= s2 and e1 >= e2:
                    if (e2 - s2) / len1 > p_inner:
                        skip = True
                        break

                elif s2 <= s1 and e2 >= e1:
                    if (e1 - s1) / len2 > p_inner:
                        skip = True
                        break

                elif s1 < s2 and e1 < e2:
                    inter = e1 - s2
                    if inter > 0 and inter / min(len2, len1) > p:
                        skip = True
                        break

                elif s1 > s2 and e1 > e2:
                    inter = e2 - s1
                    if inter > 0 and inter / min(len2, len1) > p:
                        skip = True
                        break

        if not skip:
            list_group_shapelet.append(s2_obj)
            list_pos.add(idx)
            if len(list_group_shapelet) >= c:
                break

    # 不足 c 时，按 score 补齐
    if len(list_group_shapelet) < c:
        for idx in sorted_idx:
            if idx in list_pos:
                continue
            list_group_shapelet.append(list_ppi[idx])
            list_pos.add(idx)
            if len(list_group_shapelet) >= c:
                break

    return list_group_shapelet[:c]


# =========================
# 十、临床先验集成
# =========================

def _longest_true_run(mask: np.ndarray) -> int:
    """返回布尔序列中最长连续 True 长度。"""
    if mask.size == 0:
        return 0
    m = mask.astype(np.int32)
    max_run = 0
    cur = 0
    for v in m:
        if v == 1:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return int(max_run)


def clinical_consistency(subseq: np.ndarray, label: int, sample_rate: float) -> float:
    """
    输入：子序列（原始值，未归一化），标签（0/1），采样率Hz。
    输出：标签一致性分数。
    """
    subseq = np.asarray(subseq, dtype=np.float64)
    if subseq.size == 0:
        return 0.0

    baseline = float(np.median(subseq))
    variability = float(np.std(subseq))

    decel_mask = subseq < (baseline - 15.0)
    accel_mask = subseq > (baseline + 15.0)

    longest_decel = _longest_true_run(decel_mask)
    longest_accel = _longest_true_run(accel_mask)

    abnormality = (
        0.35 * float((baseline < 110.0) or (baseline > 160.0))
        + 0.25 * float((variability < 5.0) or (variability > 25.0))
        + 0.30 * float(longest_decel >= 2 * 60 * sample_rate)
        + 0.10 * float(longest_accel < 15 * sample_rate)
    )

    if int(label) == 1:
        return float(abnormality)
    return float(1.0 - abnormality)


@dataclass
class _CandidatePack:
    ts_pos: np.ndarray  # [C]
    start: np.ndarray   # [C]
    end: np.ndarray     # [C]
    pis_ci: np.ndarray  # [C]


class OfflineShapeletDiscovery:
    """
    GPU 优先的离线 shapelet 发现器。

    与源码对应关系：
    - auto_piss_extractor / auto_ci_extractor / pcs_extractor / calculate_matrix / find_min_dist / IG / 非重叠选择
    - fit 中流程严格按给定 discovery 描述组织

    GPU 要点：
    - 使用 unfold 生成滑动窗口，批量构建“距离矩阵”效果（避免双层循环）
    - 候选批 x 目标批同时计算 PISD
    - IG 使用 GPU 批量排序和并行熵计算
    - 动态 batch：OOM 时自动减半重试
    - autocast：在 CUDA 上可启用混合精度（默认开启）
    """

    def __init__(
        self,
        pip_ratio: float = 0.2,
        window_size: int = 20,
        max_candidates_per_series: int = 50,
        clinical_weight: float = 0.0,
        sample_rate: float = 4,
        overlap_outer: float = 0.1,
        overlap_inner: float = 0.1,
        device_preference: str = "auto",
    ) -> None:
        self.pip_ratio = float(pip_ratio)
        self.window_size = int(window_size)
        self.max_candidates_per_series = int(max_candidates_per_series)
        self.clinical_weight = float(clinical_weight)
        self.sample_rate = float(sample_rate)
        self.overlap_outer = float(overlap_outer)
        self.overlap_inner = float(overlap_inner)
        self.device_preference = str(device_preference)

        self.device = self._resolve_device(device_preference)
        self.use_amp = self.device.type == "cuda"

        self.candidate_cache_: Dict[int, List[List[Any]]] = {}
        self.shapelets_: Dict[int, List[Tuple[np.ndarray, float, int, int, int]]] = {}

    @staticmethod
    def _resolve_device(pref: str) -> torch.device:
        pref = pref.lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device_preference='cuda' 但当前环境无 CUDA")
            return torch.device("cuda")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _validate_and_stack_train_data(
        x_train: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], int]:
        if len(x_train) == 0:
            raise ValueError("x_train 不能为空")

        arr_list = [np.asarray(x, dtype=np.float64).reshape(-1) for x in x_train]
        L = arr_list[0].shape[0]
        for i, x in enumerate(arr_list):
            if x.ndim != 1:
                raise ValueError(f"x_train[{i}] 不是单变量序列")
            if x.shape[0] != L:
                raise ValueError("当前实现要求所有时间序列长度一致")
        return arr_list, L

    @staticmethod
    def _build_start_end_pos(len_of_ts: int, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        list_start_pos = np.ones(len_of_ts, dtype=np.int64)
        list_end_pos = np.ones(len_of_ts, dtype=np.int64) * (2 * window_size + 1)

        for i in range(window_size):
            list_end_pos[-(i + 1)] -= window_size - i

        for i in range(window_size - 1):
            list_start_pos[i] += window_size - i - 1

        return list_start_pos, list_end_pos

    def _pick_initial_batches(self, n_candidates: int, n_targets: int, length: int) -> Tuple[int, int]:
        # 初始启发式大小，OOM 时会自动缩小
        c_batch = min(max(8, self.max_candidates_per_series), max(n_candidates, 1))
        t_batch = min(max(8, 64), max(n_targets, 1))

        if self.device.type == "cuda":
            try:
                free_mem, _ = torch.cuda.mem_get_info(self.device)
                # 粗略估算：每元素8字节（float64），保守估计 6 倍中间张量
                per_pair = max(length * (2 * self.window_size + 1), 1)
                approx = per_pair * 8 * 6
                max_pairs = max(int(free_mem * 0.4 / max(approx, 1)), 1)
                # 让 c_batch * t_batch 不超过 max_pairs
                base = int(np.sqrt(max_pairs))
                c_batch = min(c_batch, max(base, 1))
                t_batch = min(t_batch, max(base, 1))
            except Exception:
                pass

        return max(c_batch, 1), max(t_batch, 1)

    def _compute_pisd_distances_gpu(
        self,
        candidate_pack: _CandidatePack,
        x_tensor: torch.Tensor,         # [N, L]
        ci_tensor: torch.Tensor,        # [N, L-1]
        list_start_pos_t: torch.Tensor, # [L]
        list_end_pos_t: torch.Tensor,   # [L]
    ) -> torch.Tensor:
        """
        批量计算候选到所有训练序列的最小 PISD 距离。
        返回：[C, N]

        GPU实现关键点：
        - y 端使用 pad + unfold 构建滑动窗口，等价于 calculate_matrix 的带偏移列
        - 对候选组（同 start/end）与目标组同时并行
        """
        C = int(candidate_pack.start.shape[0])
        N, L = x_tensor.shape

        if C == 0:
            return torch.empty((0, N), dtype=torch.float64, device=self.device)

        out = torch.empty((C, N), dtype=torch.float64, device=self.device)

        c_batch, t_batch = self._pick_initial_batches(C, N, L)

        start_arr = candidate_pack.start
        end_arr = candidate_pack.end
        ts_pos_arr = candidate_pack.ts_pos
        pis_ci_arr = candidate_pack.pis_ci

        c_ptr = 0
        while c_ptr < C:
            c_end = min(c_ptr + c_batch, C)

            local_indices = np.arange(c_ptr, c_end)
            local_starts = start_arr[c_ptr:c_end]
            local_ends = end_arr[c_ptr:c_end]
            local_ts_pos = ts_pos_arr[c_ptr:c_end]
            local_pis_ci = pis_ci_arr[c_ptr:c_end]

            # 按 (start, end) 分组，组内候选可并行
            group_map: Dict[Tuple[int, int], List[int]] = {}
            for li, (s, e) in enumerate(zip(local_starts, local_ends)):
                key = (int(s), int(e))
                group_map.setdefault(key, []).append(li)

            t_ptr = 0
            while t_ptr < N:
                t_end = min(t_ptr + t_batch, N)

                try:
                    y = x_tensor[t_ptr:t_end]  # [Tb, L]
                    Tb = y.shape[0]

                    # 利用 unfold 生成 [Tb, L, 2w+1] 的偏移窗口
                    y_win = F.pad(
                        y,
                        pad=(self.window_size, self.window_size),
                        mode="constant",
                        value=float("inf"),
                    ).unfold(dimension=-1, size=2 * self.window_size + 1, step=1)

                    ci_tb = ci_tensor[t_ptr:t_end]  # [Tb, L-1]

                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if self.use_amp
                        else torch.autocast(device_type=self.device.type, enabled=False)
                    )

                    with amp_ctx:
                        for (s, e), li_list in group_map.items():
                            gi = np.array(li_list, dtype=np.int64)
                            global_idx = local_indices[gi]

                            src_idx = torch.as_tensor(local_ts_pos[gi], dtype=torch.long, device=self.device)
                            xg = x_tensor[src_idx]  # [Cg, L]

                            # [Cg, Tb, L, 2w+1]
                            diff = (xg[:, None, :, None] - y_win[None, :, :, :]) ** 2

                            col0 = int(list_start_pos_t[s].item())
                            col1 = int(list_end_pos_t[e - 1].item())

                            # sdist = sqrt(sum(diff[s:e, col_range], axis=0))
                            sdist = torch.sqrt(diff[:, :, s:e, col0:col1].sum(dim=2))  # [Cg, Tb, M1]

                            pcs_s = max(0, s - self.window_size + 1)
                            pcs_e = min(L, e + self.window_size)
                            pis_len = e - s

                            # pcs_ci_list: ts_ci[pcs_s:pcs_e-1]
                            pcs_ci = ci_tb[:, pcs_s : pcs_e - 1]  # [Tb, m]

                            if pis_len <= 1:
                                # 极短片段，按公式 sqrt(0 + 0.001)
                                num_align = pcs_e - pcs_s - pis_len + 1
                                d_ci = torch.sqrt(
                                    torch.full(
                                        (Tb, max(num_align, 1)),
                                        0.001,
                                        dtype=torch.float64,
                                        device=self.device,
                                    )
                                )
                            else:
                                win = pis_len - 1
                                # [Tb, num_align]
                                dsum = pcs_ci.unfold(-1, win, 1).sum(dim=-1)
                                d_ci = torch.sqrt(dsum + 0.001)

                            pis_ci = torch.as_tensor(
                                local_pis_ci[gi],
                                dtype=torch.float64,
                                device=self.device,
                            ).view(-1, 1, 1)

                            d_ci = d_ci.unsqueeze(0)  # [1, Tb, M2]
                            ratio = torch.maximum(d_ci, pis_ci) / torch.minimum(d_ci, pis_ci)

                            m = min(sdist.shape[-1], ratio.shape[-1])
                            if m <= 0:
                                min_dist = torch.full((len(gi), Tb), float("inf"), dtype=torch.float64, device=self.device)
                            else:
                                min_dist = (sdist[..., :m] * ratio[..., :m]).amin(dim=-1)  # [Cg, Tb]

                            out_idx = torch.as_tensor(global_idx, dtype=torch.long, device=self.device)
                            out[out_idx[:, None], torch.arange(t_ptr, t_end, device=self.device)[None, :]] = min_dist

                    t_ptr = t_end

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and t_batch > 1:
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        t_batch = max(1, t_batch // 2)
                        continue
                    if "out of memory" in str(exc).lower() and c_batch > 1:
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        c_batch = max(1, c_batch // 2)
                        c_end = min(c_ptr + c_batch, C)
                        local_indices = np.arange(c_ptr, c_end)
                        local_starts = start_arr[c_ptr:c_end]
                        local_ends = end_arr[c_ptr:c_end]
                        local_ts_pos = ts_pos_arr[c_ptr:c_end]
                        local_pis_ci = pis_ci_arr[c_ptr:c_end]
                        group_map = {}
                        for li, (s, e) in enumerate(zip(local_starts, local_ends)):
                            key = (int(s), int(e))
                            group_map.setdefault(key, []).append(li)
                        t_ptr = 0
                        continue
                    raise

            c_ptr = c_end

        return out

    @staticmethod
    def _entropy_torch(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        total = pos + neg
        eps = torch.finfo(pos.dtype).eps
        p1 = pos / torch.clamp(total, min=eps)
        p2 = neg / torch.clamp(total, min=eps)

        ent = torch.zeros_like(total)
        mask1 = p1 > 0
        mask2 = p2 > 0
        ent = ent - torch.where(mask1, p1 * torch.log2(torch.clamp(p1, min=eps)), torch.zeros_like(p1))
        ent = ent - torch.where(mask2, p2 * torch.log2(torch.clamp(p2, min=eps)), torch.zeros_like(p2))
        return ent

    def _batch_info_gain_gpu(
        self,
        dist_mat: torch.Tensor,   # [C, N]
        y_tensor: torch.Tensor,   # [N]
        target_class: int,
    ) -> torch.Tensor:
        """
        GPU 批量 IG：
        - 每个候选一行距离，按距离排序
        - 在 rate = 0.2,0.4,... 的触发点（round(g1_pos/t_pos,1)==rate）计算 IG
        - 返回每候选 best_ig
        """
        C, N = dist_mat.shape
        if C == 0:
            return torch.empty((0,), dtype=torch.float64, device=self.device)

        order = torch.argsort(dist_mat, dim=1)
        ys = y_tensor[order]  # [C, N]

        pos_mask = (ys == int(target_class)).to(torch.float64)
        neg_mask = 1.0 - pos_mask

        # split 只到 N-1
        cum_pos = torch.cumsum(pos_mask, dim=1)[:, :-1]
        cum_neg = torch.cumsum(neg_mask, dim=1)[:, :-1]

        t_pos = torch.sum((y_tensor == int(target_class)).to(torch.float64))
        t_neg = torch.tensor(float(N), dtype=torch.float64, device=self.device) - t_pos

        if t_pos.item() <= 0:
            return torch.full((C,), -1.0, dtype=torch.float64, device=self.device)

        g1_pos = cum_pos
        g1_neg = cum_neg
        g2_pos = t_pos - g1_pos
        g2_neg = t_neg - g1_neg

        total_entropy = self._entropy_torch(t_pos.expand(C), t_neg.expand(C))  # [C]

        rates = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float64, device=self.device)
        rounded = torch.round((g1_pos / t_pos) * 10.0) / 10.0  # [C, N-1]

        # mask[c, r, i] = 是否在 i 位置命中 rate_r
        mask = rounded.unsqueeze(1) == rates.view(1, -1, 1)
        has = mask.any(dim=2)  # [C, R]

        # 首次命中位置
        first_idx = torch.argmax(mask.to(torch.int64), dim=2)  # [C, R]

        def gather_at(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            # x: [C, N-1], idx: [C, R] -> [C, R]
            return torch.gather(x, dim=1, index=idx)

        g1p = gather_at(g1_pos, first_idx)
        g1n = gather_at(g1_neg, first_idx)
        g2p = gather_at(g2_pos, first_idx)
        g2n = gather_at(g2_neg, first_idx)

        g1_len = g1p + g1n
        g2_len = g2p + g2n

        e1 = self._entropy_torch(g1p, g1n)
        e2 = self._entropy_torch(g2p, g2n)

        inforgain = (
            total_entropy.unsqueeze(1)
            - (g1_len / float(N)) * e1
            - (g2_len / float(N)) * e2
        )  # [C, R]

        inforgain = torch.where(has, inforgain, torch.full_like(inforgain, -1.0))
        best_ig = torch.max(inforgain, dim=1).values
        return best_ig

    def fit(
        self,
        x_train: Sequence[np.ndarray],
        y_train: Sequence[int],
        top_k_total: int = 30,
    ) -> Dict[int, List[Tuple[np.ndarray, float, int, int, int]]]:
        """
        主流程（对应 shapelet_discovery.py discovery）：
        返回：
        Dict[label, List[(shapelet_subseq, score, start, end, ts_pos)]]
        """
        x_list, len_of_ts = self._validate_and_stack_train_data(x_train)
        y_np = np.asarray(y_train)
        if y_np.shape[0] != len(x_list):
            raise ValueError("y_train 长度必须与 x_train 一致")

        list_labels = np.unique(y_np)

        # 3) list_start_pos/list_end_pos
        list_start_pos, list_end_pos = self._build_start_end_pos(len_of_ts, self.window_size)

        # 4) auto_piss_extractor
        train_data_piss: List[List[np.ndarray]] = []
        for i, ts in enumerate(tqdm(x_list, desc="Extract PIS", leave=False)):
            train_data_piss.append(auto_piss_extractor(ts, num_pip=self.pip_ratio, j=i, return_pip=0))

        # 5) auto_ci_extractor
        train_data_ci: List[np.ndarray] = []
        train_data_ci_piss: List[List[float]] = []
        for i, ts in enumerate(tqdm(x_list, desc="Extract CI", leave=False)):
            ts_ci, list_ci_piss = auto_ci_extractor(ts, train_data_piss[i])
            train_data_ci.append(ts_ci)
            train_data_ci_piss.append(list_ci_piss)

        # 6) 按标签分组（结构保持）
        group_train_data: Dict[int, List[np.ndarray]] = {}
        group_train_data_pos: Dict[int, List[int]] = {}
        group_train_data_piss: Dict[int, List[List[np.ndarray]]] = {}
        group_train_data_ci_piss: Dict[int, List[List[float]]] = {}

        for lb in list_labels:
            idxs = np.where(y_np == lb)[0].tolist()
            group_train_data[int(lb)] = [x_list[i] for i in idxs]
            group_train_data_pos[int(lb)] = idxs
            group_train_data_piss[int(lb)] = [train_data_piss[i] for i in idxs]
            group_train_data_ci_piss[int(lb)] = [train_data_ci_piss[i] for i in idxs]

        # GPU tensors
        x_tensor = torch.as_tensor(np.stack(x_list, axis=0), dtype=torch.float64, device=self.device)
        ci_tensor = torch.as_tensor(np.stack(train_data_ci, axis=0), dtype=torch.float64, device=self.device)
        y_tensor = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
        list_start_pos_t = torch.as_tensor(list_start_pos, dtype=torch.long, device=self.device)
        list_end_pos_t = torch.as_tensor(list_end_pos, dtype=torch.long, device=self.device)

        shapelets_per_class = max(1, int(top_k_total / max(len(list_labels), 1)))

        result: Dict[int, List[Tuple[np.ndarray, float, int, int, int]]] = {}
        self.candidate_cache_ = {}

        # 7) 按类挖掘候选 + 计算所有序列距离 + IG + 临床分
        for lb in tqdm(list_labels.tolist(), desc="Discover per class"):
            class_label = int(lb)
            ts_positions = group_train_data_pos[class_label]

            list_ppi: List[List[Any]] = []

            for ts_pos in tqdm(ts_positions, desc=f"Class {class_label}", leave=False):
                piss = train_data_piss[ts_pos]
                ci_piss = train_data_ci_piss[ts_pos]

                if len(piss) == 0:
                    continue

                # 限制每条序列候选数（按参数要求）
                if self.max_candidates_per_series > 0 and len(piss) > self.max_candidates_per_series:
                    piss = piss[: self.max_candidates_per_series]
                    ci_piss = ci_piss[: self.max_candidates_per_series]

                valid = [(p, ci) for p, ci in zip(piss, ci_piss) if p[1] - p[0] >= 8]
                if not valid:
                    continue
                piss, ci_piss = zip(*valid)

                starts = np.array([int(p[0]) for p in piss], dtype=np.int64)
                ends = np.array([int(p[1]) for p in piss], dtype=np.int64)
                ts_pos_arr = np.full_like(starts, ts_pos)
                pis_ci_arr = np.array(ci_piss, dtype=np.float64)

                cand_pack = _CandidatePack(
                    ts_pos=ts_pos_arr,
                    start=starts,
                    end=ends,
                    pis_ci=pis_ci_arr,
                )

                # 计算该批候选到所有训练序列的最小 PISD 距离
                dist_mat = self._compute_pisd_distances_gpu(
                    candidate_pack=cand_pack,
                    x_tensor=x_tensor,
                    ci_tensor=ci_tensor,
                    list_start_pos_t=list_start_pos_t,
                    list_end_pos_t=list_end_pos_t,
                )  # [C, N]

                # IG（GPU 批量）
                ig_scores = self._batch_info_gain_gpu(
                    dist_mat=dist_mat,
                    y_tensor=y_tensor,
                    target_class=class_label,
                ).detach().cpu().numpy()

                # 临床一致性（按要求使用原始子序列）
                cc_scores = np.zeros_like(ig_scores, dtype=np.float64)
                if self.clinical_weight != 0.0:
                    src_ts = x_list[ts_pos]
                    for k in range(len(starts)):
                        subseq = src_ts[starts[k] : ends[k]]
                        cc_scores[k] = clinical_consistency(subseq, class_label, self.sample_rate)

                final_scores = ig_scores + self.clinical_weight * cc_scores

                for k in range(len(starts)):
                    list_ppi.append(
                        [
                            int(ts_pos),
                            int(starts[k]),
                            int(ends[k]),
                            float(final_scores[k]),  # 用于排序与重叠抑制
                            int(class_label),
                            0,                      # dim 固定 0
                            float(ig_scores[k]),
                            float(cc_scores[k]),
                        ]
                    )

            self.candidate_cache_[class_label] = list_ppi

            # 每个类别选 shapelets_per_class
            selected = find_c_shapelet_non_overlab(
                list_ppi=list_ppi,
                c=shapelets_per_class,
                p=self.overlap_outer,
                p_inner=self.overlap_inner,
                len_ts=len_of_ts,
            )

            class_shapelets: List[Tuple[np.ndarray, float, int, int, int]] = []
            for item in selected:
                ts_pos, s, e, final_score = int(item[0]), int(item[1]), int(item[2]), float(item[3])
                subseq = x_list[ts_pos][s:e].copy()
                class_shapelets.append((subseq, final_score, s, e, ts_pos))

            result[class_label] = class_shapelets

        self.shapelets_ = result
        return result

    def save(self, path: str) -> None:
        payload = {
            "config": {
                "pip_ratio": self.pip_ratio,
                "window_size": self.window_size,
                "max_candidates_per_series": self.max_candidates_per_series,
                "clinical_weight": self.clinical_weight,
                "sample_rate": self.sample_rate,
                "overlap_outer": self.overlap_outer,
                "overlap_inner": self.overlap_inner,
                "device_preference": self.device_preference,
            },
            "candidate_cache": self.candidate_cache_,
            "shapelets": self.shapelets_,
        }
        torch.save(payload, path)

    def load(self, path: str) -> "OfflineShapeletDiscovery":
        payload = torch.load(path, map_location="cpu")
        self.candidate_cache_ = payload.get("candidate_cache", {})
        self.shapelets_ = payload.get("shapelets", {})
        return self

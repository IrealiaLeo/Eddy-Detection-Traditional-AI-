import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from typing import Tuple, List

def find_local_extrema(ssh: np.ndarray, xq: np.ndarray, yq: np.ndarray, nb: int, 
                       is_max: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    在二维SSH场上查找局部极值点（最大值或最小值）。
    这些点是涡旋中心的候选位置。

    Args:
        ssh (np.ndarray): 海面高度（Sea Surface Height）二维数组。
        xq, yq (np.ndarray): 网格坐标。
        nb (int): 边界排除的格点数，以避免边缘效应。
        is_max (bool): True表示查找局部最大值，False表示查找局部最小值。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 极值点的经度(lon)和纬度(lat)数组。
    """
    # 1. 使用scipy的滤波器快速找到局部极值
    if is_max:
        # 3x3窗口内的最大值等于自身，即为局部最大值
        extrema_mask = (maximum_filter(ssh, size=3) == ssh)
    else:
        # 3x3窗口内的最小值等于自身，即为局部最小值
        extrema_mask = (minimum_filter(ssh, size=3) == ssh)
    
    # 2. 获取极值点的索引
    yh_idx, xh_idx = np.where(extrema_mask)

    # 3. 筛选掉边界附近的点
    # 首先是基于索引的边界检查
    ny, nx = ssh.shape
    valid_mask_idx = (xh_idx >= nb) & (xh_idx < nx - nb) & \
                     (yh_idx >= nb) & (yh_idx < ny - nb)
    xh_idx, yh_idx = xh_idx[valid_mask_idx], yh_idx[valid_mask_idx]

    # 其次是基于坐标值的边界检查，确保点在有效地理区域内
    # （此步假设nb已经足够，但作为双重保险）
    valid_mask_coord = (xq[0, xh_idx] > xq[0, nb]) & (xq[0, xh_idx] < xq[0, -nb]) & \
                       (yq[yh_idx, 0] > yq[nb, 0]) & (yq[yh_idx, 0] < yq[-nb, 0])
    xh_idx, yh_idx = xh_idx[valid_mask_coord], yh_idx[valid_mask_coord]

    # 4. 返回极值点的经纬度坐标
    return xq[0, xh_idx], yq[yh_idx, 0]


def find_velocity_minima(u_orig: np.ndarray, v_orig: np.ndarray, vorticity: np.ndarray,
                         xq: np.ndarray, yq: np.ndarray, nb: int, 
                         nu_stack: List[int], ru: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于流场几何特征检测涡旋中心，核心思想来自 Nencioli et al. (2010)。
    1. 在速度幅值场上寻找局部最小值。
    2. 对每个最小值点，施加一系列几何约束（如速度矢量在特定方向反向）。
    3. 使用多尺度邻域(nu_stack)进行检测，增加鲁棒性。
    4. 根据涡度符号将通过筛选的中心点分为冷涡（气旋）和暖涡（反气旋）。
    
    此函数经过重构，以提高代码清晰度和执行效率。

    Args:
        u_orig, v_orig: 原始u, v速度分量，位于交错网格。
        vorticity: 中心网格上的涡度场。
        xq, yq: 网格坐标。
        nb: 边界排除范围。
        nu_stack: 多尺度检测的邻域半径列表。
        ru: 速度比值阈值。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - cold_centers (Nc, 2): 冷涡中心坐标 [lon, lat]。
        - warm_centers (Nw, 2): 暖涡中心坐标 [lon, lat]。
    """
    # 1. 插值速度到中心网格，并计算速度幅值
    # u(yh, xq) -> u_center(ny-1, nx-1)
    u_center = 0.5 * (u_orig[:, :-1] + u_orig[:, 1:])
    u_center = u_center[1:, :]  # 调整y轴对齐
    # v(yq, xh) -> v_center(ny-1, nx-1)
    v_center = 0.5 * (v_orig[:-1, :] + v_orig[1:, :])
    v_center = v_center[:, 1:]  # 调整x轴对齐
    
    ny, nx = vorticity.shape
    u_center, v_center = u_center[:ny, :nx], v_center[:ny, :nx]
    
    velocity_mag = np.sqrt(u_center**2 + v_center**2)
    
    # 2. 检测速度幅值的局部最小值
    minima_mask = (minimum_filter(velocity_mag, size=3) == velocity_mag)
    
    # 3. 排除边界点
    minima_mask[:nb, :] = minima_mask[-nb:, :] = minima_mask[:, :nb] = minima_mask[:, -nb:] = False
    
    # 4. 获取候选中心的索引
    y_candidates, x_candidates = np.where(minima_mask)
    
    # 5. 多尺度几何约束检查
    valid_centers_indices = []
    max_nu = max(nu_stack) if nu_stack else 0

    for y_idx, x_idx in zip(y_candidates, x_candidates):
        # 提前进行一次性边界检查，如果最大邻域都越界，则跳过
        if not (max_nu < y_idx < ny - max_nu and max_nu < x_idx < nx - max_nu):
            continue

        is_valid_for_any_nu = False
        for nu in nu_stack:
            # a. 提取邻域边界上的速度幅值
            boundary_n = velocity_mag[y_idx + nu, x_idx - nu : x_idx + nu + 1]
            boundary_s = velocity_mag[y_idx - nu, x_idx - nu : x_idx + nu + 1]
            boundary_w = velocity_mag[y_idx - nu : y_idx + nu + 1, x_idx - nu]
            boundary_e = velocity_mag[y_idx - nu : y_idx + nu + 1, x_idx + nu]
            
            U_min_boundary = min(np.min(boundary_n), np.min(boundary_s), np.min(boundary_w), np.min(boundary_e))
            U_max_boundary = max(np.max(boundary_n), np.max(boundary_s), np.max(boundary_w), np.max(boundary_e))

            # b. 几何约束1：速度分量反号
            # v在南北边界反号，u在东西边界反号
            cond_sign_reversal = (
                (v_center[y_idx + nu, x_idx] * v_center[y_idx - nu, x_idx] < 0) and
                (u_center[y_idx, x_idx + nu] * u_center[y_idx, x_idx - nu] < 0)
            )
            
            # c. 几何约束2：速度比值条件
            if U_min_boundary < 1e-9: U_min_boundary = 1e-9  # 避免除以零
            
            cond_ratio = (
                max(np.max(boundary_n) / np.max(boundary_s), np.max(boundary_s) / np.max(boundary_n)) < ru and
                max(np.max(boundary_w) / np.max(boundary_e), np.max(boundary_e) / np.max(boundary_w)) < ru and
                (U_max_boundary / U_min_boundary < ru)
            )

            if cond_sign_reversal and cond_ratio:
                is_valid_for_any_nu = True
                break  # 只要在一个尺度上满足条件即可，跳出nu循环

        if is_valid_for_any_nu:
            valid_centers_indices.append((y_idx, x_idx))

    if not valid_centers_indices:
        return np.empty((0, 2)), np.empty((0, 2))

    # 6. 根据涡度对有效中心进行分类
    valid_indices = np.array(valid_centers_indices)
    y_valid, x_valid = valid_indices[:, 0], valid_indices[:, 1]
    
    center_vorticity = vorticity[y_valid, x_valid]
    
    # 中心网格坐标
    x_center_coords = xq[0, 1:-1][:nx]
    y_center_coords = yq[1:-1, 0][:ny]
    
    lon_valid = x_center_coords[x_valid]
    lat_valid = y_center_coords[y_valid]
    
    cold_mask = center_vorticity > 0
    warm_mask = center_vorticity < 0
    
    cold_centers = np.column_stack((lon_valid[cold_mask], lat_valid[cold_mask]))
    warm_centers = np.column_stack((lon_valid[warm_mask], lat_valid[warm_mask]))
    
    return cold_centers, warm_centers
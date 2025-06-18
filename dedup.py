import numpy as np
from scipy.spatial import KDTree
from typing import Tuple

def filter_and_deduplicate(
    vortex_centers: np.ndarray, 
    ssh_extrema: np.ndarray, 
    spatial_tolerance: float = 0.5,
    dedup_radius: float = 0.25
) -> np.ndarray:
    """
    对涡旋候选中心进行筛选和去重。
    1. 筛选：只保留那些与SSH极值点足够近的涡旋中心。
    2. 去重：使用KDTree移除彼此过于接近的涡旋中心点。

    Args:
        vortex_centers (np.ndarray): (N, 2) 数组，由速度法检测出的涡旋中心 [lon, lat]。
        ssh_extrema (np.ndarray): (M, 2) 数组，由SSH场检测出的极值点 [lon, lat]。
        spatial_tolerance (float): 筛选步骤中，涡旋中心与SSH极值点的最大允许距离（度）。
        dedup_radius (float): 去重步骤中，用于判断点是否重复的半径（度）。

    Returns:
        np.ndarray: (K, 2) 数组，经过筛选和去重后的最终涡旋中心。
    """
    if vortex_centers.size == 0 or ssh_extrema.size == 0:
        return np.empty((0, 2))

    # 1. 筛选：向量化计算每个涡旋中心与所有SSH极值点的距离
    # 使用广播 (N, 1, 2) 和 (1, M, 2) 来创建一个 (N, M, 2) 的差值矩阵
    delta_lon = np.abs(vortex_centers[:, 0, None] - ssh_extrema[None, :, 0])
    delta_lat = np.abs(vortex_centers[:, 1, None] - ssh_extrema[None, :, 1])

    # 检查是否存在任何一个SSH极值点在容忍范围内
    # (delta_lon <= tol) & (delta_lat <= tol) 形成一个 (N, M) 的布尔矩阵
    # np.any(..., axis=1) 检查每一行（每个涡旋中心），看它是否与任何SSH极值点匹配
    is_close_to_ssh = np.any((delta_lon <= spatial_tolerance) & (delta_lat <= spatial_tolerance), axis=1)
    
    filtered_centers = vortex_centers[is_close_to_ssh]

    if filtered_centers.size == 0:
        return filtered_centers

    # 2. 去重：使用KDTree高效查找邻近点
    tree = KDTree(filtered_centers)
    
    # query_ball_point返回一个列表的列表，其中neighbors[i]是点i半径内的所有点的索引
    neighbors = tree.query_ball_point(filtered_centers, r=dedup_radius)
    
    n = filtered_centers.shape[0]
    keep = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not keep[i]:
            continue
        # 将所有比i大且在i邻域内的点的keep标记设为False
        # 这样可以保留每个聚集区中的第一个点（索引最小的那个）
        for j in neighbors[i]:
            if j > i:
                keep[j] = False
                
    return filtered_centers[keep]
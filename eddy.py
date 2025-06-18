from datetime import datetime, timedelta
import numpy as np
from typing import Tuple, List
from ow import compute_ow
from extrema import find_local_extrema, find_velocity_minima
from dedup import filter_and_deduplicate

def detect_eddy(
    day_idx: int,
    xq: np.ndarray, yq: np.ndarray, xh: np.ndarray, yh: np.ndarray,
    ssh_all: np.ndarray, u_all: np.ndarray, v_all: np.ndarray,
    base_date_str: str,
    nb: int,
    nub_stack: List[int],
    ru: float
) -> Tuple:
    """
    执行单日涡旋检测的全流程。
    该函数整合了OW计算、多源涡旋中心检测、筛选和去重等步骤。

    Args:
        day_idx (int): 当前处理日期在年度数据中的索引。
        xq, yq, xh, yh: 交错网格坐标。
        ssh_all, u_all, v_all: 包含一年中所有天数的原始数据。
        base_date_str (str): 年度数据的起始日期（如 "19960101"）。
        nb (int): 边界排除范围。
        nub_stack (List[int]): 速度法检测的多尺度邻域半径。
        ru (float): 速度法检测的速度比值阈值。

    Returns:
        Tuple: 包含所有检测结果的元组。
            - result_cold (np.ndarray): 最终冷涡中心 [lon, lat]。
            - result_warm (np.ndarray): 最终暖涡中心 [lon, lat]。
            - ssh_center (np.ndarray): 插值到中心网格的SSH。
            - vorticity_center (np.ndarray): 中心网格的涡度。
            - ow_center (np.ndarray): 中心网格的OW参数。
            - ow0_for_contour (float): 用于绘制等值线的OW阈值。
            - u_center (np.ndarray): 插值到中心网格的u速度。
            - v_center (np.ndarray): 插值到中心网格的v速度。
            - new_date_str (str): 当前处理的日期字符串 "YYYYMMDD"。
    """
    # 1. 提取当天数据
    u_day, v_day, ssh_day = u_all[day_idx], v_all[day_idx], ssh_all[day_idx]
    
    # 2. 计算OW, 涡度和OW阈值
    ow_center, ow0_std, vorticity_center = compute_ow(u_day, v_day, xq, yq, xh, yh)
    # 根据论文，涡旋边界由 OW = -0.2 * σ_ow 定义
    ow0_for_contour = -0.2 * ow0_std

    # 3. 方法一：基于SSH极值检测涡旋中心候选
    lon_max_ssh, lat_max_ssh = find_local_extrema(ssh_day, xh, yh, nb, is_max=True)
    lon_min_ssh, lat_min_ssh = find_local_extrema(ssh_day, xh, yh, nb, is_max=False)
    
    # 4. 方法二：基于速度场几何特征检测涡旋中心
    cold_v, warm_v = find_velocity_minima(
        u_day, v_day, vorticity_center, xq, yq, nb, nub_stack, ru
    )

    # 5. 整合与筛选：结合两种方法的结果并去重
    ssh_extrema_warm = np.column_stack((lon_max_ssh, lat_max_ssh))
    ssh_extrema_cold = np.column_stack((lon_min_ssh, lat_min_ssh))
    
    result_warm = filter_and_deduplicate(warm_v, ssh_extrema_warm)
    result_cold = filter_and_deduplicate(cold_v, ssh_extrema_cold)

    # 6. 计算当天日期
    date_obj = datetime.strptime(base_date_str, "%Y%m%d")
    new_date = (date_obj + timedelta(days=day_idx)).strftime("%Y%m%d")

    # 7. 准备用于保存的、对齐到中心网格的原始物理量
    # SSH 插值到中心
    ssh_center = 0.25 * (ssh_day[:-1, :-1] + ssh_day[:-1, 1:] + ssh_day[1:, :-1] + ssh_day[1:, 1:])
    # U/V 插值到中心
    ny_c, nx_c = ow_center.shape
    u_center = 0.5 * (u_day[:, :-1] + u_day[:, 1:])[1:, :][:ny_c, :nx_c]
    v_center = 0.5 * (v_day[:-1, :] + v_day[1:, :])[:, 1:][:ny_c, :nx_c]

    return (
        result_cold, result_warm, ssh_center, vorticity_center, ow_center, 
        ow0_for_contour, u_center, v_center, new_date
    )
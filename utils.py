import numpy as np
from typing import Tuple

def find_index_range(coords_1d: np.ndarray, min_val: float, max_val: float) -> Tuple[int, int]:
    """
    在一维有序坐标数组中，查找给定范围对应的索引区间。

    Args:
        coords_1d (np.ndarray): 一维单调递增的坐标数组。
        min_val (float): 范围的最小值。
        max_val (float): 范围的最大值。

    Returns:
        Tuple[int, int]: 对应于[min_val, max_val]区间的起始和结束索引。
    """
    # 使用searchsorted高效地找到插入点索引
    # 'left'确保包含所有>=min_val的值
    start_index = np.searchsorted(coords_1d, min_val, side='left')
    # 'right'确保包含所有<=max_val的值
    end_index = np.searchsorted(coords_1d, max_val, side='right')
    return start_index, end_index
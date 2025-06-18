import os
import numpy as np
import matplotlib.path as mpath
from skimage.measure import find_contours
from shapely.geometry import Polygon, Point
from typing import List, Dict, Union

def filter_and_match_contours(contours: List[np.ndarray], centers: np.ndarray, 
                              tolerance: float = 1e-4) -> List[np.ndarray]:
    """
    筛选等值线，只保留那些内部至少包含一个给定涡旋中心的闭合等值线。

    Args:
        contours (List[np.ndarray]): find_contours找到的等值线列表。
        centers (np.ndarray): (N, 2) 的涡旋中心坐标数组 [lon, lat]。
        tolerance (float): 用于处理浮点精度问题的缓冲值。

    Returns:
        List[np.ndarray]: 包含涡旋中心的有效等值线列表。
    """
    if centers.size == 0:
        return []
        
    valid_contours = []
    for contour in contours:
        # 将等值线顶点转换为Shapely多边形对象
        poly = Polygon(contour)
        # 检查多边形是否有效（例如，没有自相交）
        if not poly.is_valid:
            poly = poly.buffer(0) # 尝试修复无效多边形
            if not poly.is_valid:
                continue
        
        # 检查是否有任何一个中心点落在多边形内部
        # poly.contains()对于边界点可能返回False，buffer可以解决此问题
        if any(poly.buffer(tolerance).contains(Point(center)) for center in centers):
            valid_contours.append(contour)
            
    return valid_contours


def save_contour(xq: np.ndarray, yq: np.ndarray, ow_data: np.ndarray, ow0_threshold: float,
                 result_warm: np.ndarray, result_cold: np.ndarray, new_date: str,
                 save_contour_flag: bool = False) -> np.ndarray:
    """
    基于OW阈值提取涡旋边界，并根据涡旋中心填充区域，生成标签矩阵。

    Args:
        xq, yq: 网格坐标。
        ow_data (np.ndarray): 中心网格上的Okubo-Weiss参数。
        ow0_threshold (float): 用于定义涡旋核心的OW阈值。
        result_warm, result_cold: 暖涡和冷涡的中心坐标。
        new_date (str): 日期字符串，用于保存文件名。
        save_contour_flag (bool): 是否将涡旋边界保存为.npz文件。

    Returns:
        np.ndarray: 涡旋标签矩阵 (0: 背景, 1: 暖涡, 2: 冷涡)。
    """
    # 1. 创建中心网格坐标
    x_center = xq[0, 1:-1]
    y_center = yq[1:-1, 0]
    x_grid, y_grid = np.meshgrid(x_center, y_center)
    
    ny, nx = ow_data.shape
    x_grid, y_grid = x_grid[:ny, :nx], y_grid[:ny, :nx]

    # 2. 识别涡旋核心区域
    # 根据论文，涡旋核心是OW为负且小于某个负阈值的区域
    # ow_data <= ow0_threshold (ow0_threshold本身是负数)
    is_vortex_core = (ow_data <= ow0_threshold)
    
    # 3. 查找涡旋核心区域的边界
    # find_contours工作在布尔矩阵上，0.5是标准水平面
    all_contours_indices = find_contours(is_vortex_core, 0.5)

    # 4. 将等值线从索引坐标转换为地理坐标
    all_contours_coords = []
    for contour in all_contours_indices:
        # 插值以获得更精确的坐标
        y_coords = np.interp(contour[:, 0], np.arange(ny), y_center)
        x_coords = np.interp(contour[:, 1], np.arange(nx), x_center)
        all_contours_coords.append(np.column_stack([x_coords, y_coords]))

    # 5. 匹配等值线与涡旋中心
    warm_contours = filter_and_match_contours(all_contours_coords, result_warm)
    cold_contours = filter_and_match_contours(all_contours_coords, result_cold)
    
    # 6. 创建并填充标签矩阵
    label_grid = np.zeros(x_grid.shape, dtype=np.uint8)
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    def fill_contours(contours_to_fill: List[np.ndarray], fill_value: int):
        for contour in contours_to_fill:
            if len(contour) < 3: continue
            path = mpath.Path(contour, closed=True)
            # contains_points是一个非常高效的函数，用于判断一批点是否在路径内
            mask = path.contains_points(grid_points).reshape(label_grid.shape)
            label_grid[mask] = fill_value

    fill_contours(warm_contours, 1)  # 1 代表暖涡
    fill_contours(cold_contours, 2)  # 2 代表冷涡

    # 7. (可选) 保存涡旋边界
    if save_contour_flag:
        save_path = "./eddies_contour"
        os.makedirs(save_path, exist_ok=True)
        boundary_data = {
            'warm_eddies': np.array(warm_contours, dtype=object),
            'cold_eddies': np.array(cold_contours, dtype=object)
        }
        np.savez(os.path.join(save_path, f'eddies_contour_{new_date}.npz'),
                 **boundary_data, allow_pickle=True)
                 
    return label_grid
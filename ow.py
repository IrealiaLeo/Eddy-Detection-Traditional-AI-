import numpy as np
from typing import Tuple
from preprocess import standardize_ow

def compute_ow(u: np.ndarray, v: np.ndarray, xq: np.ndarray, yq: np.ndarray, 
               xh: np.ndarray, yh: np.ndarray, L_standardize: int = 50, 
               min_samples: int = 5, smooth_method: str = 'median', 
               smooth_size: int = 3) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    计算Okubo-Weiss (OW)参数、其阈值以及涡度场。
    OW参数用于衡量流场中旋转与变形的相对强度，是涡旋检测的关键指标。
    OW = (sn^2 + ss^2) - ζ^2，其中sn为法向应变，ss为切向应变，ζ为相对涡度。

    Args:
        u, v: u(yh, xq)和v(yq, xh)网格上的速度分量。
        xq, yq, xh, yh: 交错网格的坐标。
        L_standardize, min_samples, smooth_method, smooth_size: 传递给standardize_ow的参数。

    Returns:
        Tuple[np.ndarray, float, np.ndarray]:
        - ow (np.ndarray): 计算并插值到中心网格的OW参数。
        - ow0 (float): OW参数的全局标准差，用作涡旋判定的阈值。
        - vorticity (np.ndarray): 相对涡度场。
    """
    R = 6371000.0  # 地球半径 (米)

    # 1. 计算网格间距 (米)
    # 注意cos(lat)修正，将经度差转换为实际距离
    xu = np.radians(np.diff(xq, axis=1)) * R * np.cos(np.radians(yh))
    xv = np.radians(np.diff(xh, axis=1)) * R * np.cos(np.radians(yq[1:-1, :])) # yq需对齐
    yu = np.radians(np.diff(yh, axis=0)) * R
    yv = np.radians(np.diff(yq, axis=0)) * R

    # 2. 计算速度梯度 (一阶差分近似)
    ux = np.diff(u, axis=1) / xu
    uy = np.diff(u, axis=0) / yu[:, 1:-1] # yu需对齐
    vx = np.diff(v, axis=1) / xv
    vy = np.diff(v, axis=0) / yv

    # 3. 将所有梯度分量插值到网格中心
    # 采用简单的四点平均（box average）
    vy_center = 0.25 * (vy[:-1, :-1] + vy[:-1, 1:] + vy[1:, :-1] + vy[1:, 1:])
    ux_center = 0.25 * (ux[:-1, :-1] + ux[:-1, 1:] + ux[1:, :-1] + ux[1:, 1:])
    vx_center = vx[1:-1, :]
    uy_center = uy[:, 1:-1]

    # 4. 计算应变(strain)和涡度(vorticity)
    # sn: 法向应变 (stretching deformation)
    # ss: 切向应变 (shearing deformation)
    # vorticity: 相对涡度 (ζ = vx - uy)
    sn = ux_center - vy_center
    ss = vx_center + uy_center
    vorticity = vx_center - uy_center

    # 5. 计算OW参数
    # OW = strain^2 - vorticity^2
    ow = sn**2 + ss**2 - vorticity**2

    # 6. 对OW进行局部标准化（可选）
    # 填充边界以处理卷积边缘效应，之后再移除
    ow_padded = np.pad(ow, pad_width=1, mode='constant', constant_values=np.nan)
    if L_standardize is not None:
        ow_standardized = standardize_ow(ow_padded, L=L_standardize, min_samples=min_samples,
                                       smooth_method=smooth_method, smooth_size=smooth_size)
    else:
        ow_standardized = ow_padded

    # 7. 计算OW阈值并返回最终结果
    ow0 = np.nanstd(ow_standardized)
    
    # 移除填充的边界，使结果与vorticity等中心场对齐
    ow_final = ow_standardized[1:-1, 1:-1]
    
    return ow_final, ow0, vorticity
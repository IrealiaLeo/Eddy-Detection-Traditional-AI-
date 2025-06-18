import numpy as np
from scipy import ndimage, signal
from typing import Optional

def smooth_edges(z: np.ndarray, mask: np.ndarray, filter_size: int = 3, method: str = 'median') -> np.ndarray:
    """
    对标准化后的参数在NaN值（原始陆地/缺测区域）的边缘进行平滑处理。
    这可以减少因标准化在数据边缘产生的人为噪声。

    Args:
        z (np.ndarray): 待平滑的二维数组，可能包含NaN。
        mask (np.ndarray): 原始的NaN掩码（True表示NaN）。
        filter_size (int): 平滑滤波器的尺寸。
        method (str): 平滑方法，'median'或'gaussian'。

    Returns:
        np.ndarray: 边缘平滑后的数组。
    """
    # 确定需要被平滑像素替换的区域（即原始的NaN区域）
    edge_mask = mask | np.isnan(z)
    # 填充NaN以便滤波器可以处理
    z_filled = np.nan_to_num(z, nan=0.0)

    # 应用指定的平滑滤波器
    if method == 'median':
        z_smoothed = ndimage.median_filter(z_filled, size=filter_size)
    elif method == 'gaussian':
        z_smoothed = ndimage.gaussian_filter(z_filled, sigma=filter_size / 2.0)
    else:
        raise ValueError("平滑方法必须是 'median' 或 'gaussian'")

    # 仅在原始NaN区域使用平滑后的值，保留其他区域的原始值
    return np.where(edge_mask, z_smoothed, z)

def standardize_ow(ow: np.ndarray, L: int = 50, min_samples: int = 5,
                   smooth_method: Optional[str] = 'median', smooth_size: int = 3) -> np.ndarray:
    """
    对Okubo-Weiss (OW)参数进行局部标准化，计算 z = (x - μ) / σ。
    其中μ和σ是在每个点周围 L x L 的窗口内计算的局部均值和标准差。
    此版本增强了数值稳定性。

    Args:
        ow (np.ndarray): 原始OW参数二维数组。
        L (int): 用于计算局部统计量的正方形窗口边长。
        min_samples (int): 窗口内有效（非NaN）数据点的最小数量。
        smooth_method (Optional[str]): 是否及如何对结果进行边缘平滑。
        smooth_size (int): 边缘平滑的滤波器尺寸。

    Returns:
        np.ndarray: 标准化后的OW参数。
    """
    # 1. 准备工作：创建卷积核和处理NaN
    kernel = np.ones((L, L))
    original_mask = np.isnan(ow)
    ow_no_nan = np.nan_to_num(ow, nan=0.0)
    valid_mask = (~original_mask).astype(float) # 1代表有效数据，0代表NaN

    # 2. 计算局部统计量
    # n0: 窗口内的有效数据点数
    n0 = signal.convolve2d(valid_mask, kernel, mode='same', boundary='symm')
    
    # m0: 窗口内值的总和
    m0 = signal.convolve2d(ow_no_nan, kernel, mode='same', boundary='symm')
    
    # v0: 窗口内值的平方和
    v0 = signal.convolve2d(ow_no_nan**2, kernel, mode='same', boundary='symm')

    # 3. 安全地计算均值和标准差
    # 对于有效样本数过少的区域，将其视为无效
    n0 = np.where(n0 < min_samples, np.nan, n0)
    
    # 局部均值 mu = m0 / n0
    mu = np.divide(m0, n0, where=(~np.isnan(n0)) & (n0 > 0), out=np.full_like(m0, np.nan))
    
    # 局部方差 variance = (v0 / n0) - mu**2
    variance = np.divide(v0, n0, where=(~np.isnan(n0)) & (n0 > 0), out=np.full_like(v0, np.nan)) - mu**2
    variance = np.maximum(variance, 0)  # 确保方差非负，避免浮点误差
    
    # 局部标准差 sigma
    sigma = np.sqrt(variance)
    # 防止除以零，将零标准差替换为极小值
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    
    # 4. 计算标准化后的OW (z-score)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (ow - mu) / sigma
    
    # 5. 后处理
    # 恢复原始的NaN掩码
    z[original_mask] = np.nan
    
    # 可选的边缘平滑
    if smooth_method is not None:
        z = smooth_edges(z, original_mask, filter_size=smooth_size, method=smooth_method)
    
    return z
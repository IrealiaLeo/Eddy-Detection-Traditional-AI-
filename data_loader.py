import os
import numpy as np
import netCDF4 as nc
from typing import Tuple
from utils import find_index_range

def load_data(extent: Tuple[float, float, float, float], date: str) -> Tuple[np.ndarray, ...]:
    """
    根据给定的范围和年份，加载并裁剪海洋模型日平均数据。

    Args:
        extent (Tuple[float, float, float, float]): 经纬度范围 [lon_min, lon_max, lat_min, lat_max]。
        date (str): 数据的年份，格式为 "YYYY"。

    Returns:
        Tuple[np.ndarray, ...]: 裁剪后的 xq, yq, xh, yh, time, ssh, u, v。
                                如果文件不存在，则会引发 FileNotFoundError。
    """
    # 构建NetCDF文件的完整路径
    nc_file = f'/home/kangdj/MOM6/NWA25/ocean_daily/{date}.ocean_daily.nc'
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"数据文件未找到: {nc_file}")

    with nc.Dataset(nc_file, 'r') as dataset:
        # 读取 staggered grid (交错网格) 的坐标
        # xh/yh: h-points, 速度和厚度的中心
        # xq/yq: q-points, 涡度和输运的角点
        xh = dataset.variables['xh'][:].reshape(1, -1)
        xq = dataset.variables['xq'][:].reshape(1, -1)
        yh = dataset.variables['yh'][:].reshape(-1, 1)
        yq = dataset.variables['yq'][:].reshape(-1, 1)

        # 读取变量
        ssh = dataset.variables['zos'][:]  # Sea Surface Height
        u = dataset.variables['ssu'][:]   # Zonal velocity
        v = dataset.variables['ssv'][:]   # Meridional velocity
        time = dataset.variables['time'][:]

    lon_min, lon_max, lat_min, lat_max = extent

    # 为了完整包含边缘数据，对查找范围进行微小扩展（0.041是为了让最后裁切出的网格为500*300）
    xh_start, xh_end = find_index_range(xh.ravel(), lon_min, lon_max + 0.041)
    xq_start, xq_end = find_index_range(xq.ravel(), lon_min - 0.041, lon_max + 0.041)
    yh_start, yh_end = find_index_range(yh.ravel(), lat_min, lat_max + 0.041)
    yq_start, yq_end = find_index_range(yq.ravel(), lat_min - 0.041, lat_max + 0.041)

    # 创建切片对象
    xh_slice = slice(xh_start, xh_end)
    xq_slice = slice(xq_start, xq_end)
    yh_slice = slice(yh_start, yh_end)
    yq_slice = slice(yq_start, yq_end)

    # 裁剪坐标和变量
    xh_cropped = xh[:, xh_slice]
    xq_cropped = xq[:, xq_slice]
    yh_cropped = yh[yh_slice, :]
    yq_cropped = yq[yq_slice, :]

    ssh_cropped = ssh[:, yh_slice, xh_slice]
    u_cropped = u[:, yh_slice, xq_slice]
    v_cropped = v[:, yq_slice, xh_slice]

    # 将MaskedArray中的掩码值（通常是缺测值）填充为NaN
    ssh_filled = ssh_cropped.filled(np.nan)
    u_filled = u_cropped.filled(np.nan)
    v_filled = v_cropped.filled(np.nan)

    return xq_cropped, yq_cropped, xh_cropped, yh_cropped, time, ssh_filled, u_filled, v_filled
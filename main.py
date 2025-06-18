import os
import time
import h5py
import numpy as np
import traceback
from typing import Dict, Any, List

# 假设这些模块在同一路径或Python路径下
from data_loader import load_data
from eddy import detect_eddy
from contour import save_contour

# --- 全局配置参数 ---
# 数据和路径设置
HDF5_SAVE_PATH = "/home/kangdj_gbr2021/eddy_detect_by_ow/data_400x500/h5_final"
HDF5_FILENAME = "eddy_data_daily_raw_combined.h5"
START_YEAR = 1996
END_YEAR = 2020
GEOGRAPHIC_EXTENT = [-75.0, -55.0, 29.8, 42.64]  # [lon_min, lon_max, lat_min, lat_max]

# 涡旋检测算法参数
PARAMS_EDDY = {
    "nb": 5,          # 边界排除格点数
    "nub_stack": [4, 8, 20], # 速度法多尺度邻域半径
    "ru": 10,         # 速度法比值阈值
}

# --- HDF5 元数据 ---
METADATA_H5: Dict[str, Any] = {
    "description": "Daily raw ocean data and corresponding eddy labels for ML training.",
    "source_paper": "Kang, D., & Curchitser, E. N. (2013). Gulf Stream eddy characteristics in a high-resolution ocean model.",
    "variables": ["u", "v", "velocity_mag", "ow", "vorticity", "ssh"],
    "coordinates": {
        "x": "Longitude (center grid, degrees_east)",
        "y": "Latitude (center grid, degrees_north)",
        "time": "Date (yyyymmdd integer format)",
    },
    "class_labels": {0: "background", 1: "warm_eddy (anticyclonic)", 2: "cold_eddy (cyclonic)"},
}

def initialize_hdf5_file(hf: h5py.File, x_center: np.ndarray, y_center: np.ndarray, shape: Tuple[int, int]):
    """在HDF5文件中创建数据集和元数据。"""
    print("首次写入：正在创建HDF5文件结构...")
    ny, nx = shape
    num_features = len(METADATA_H5["variables"])

    hf.create_dataset('x', data=x_center.astype(np.float32), compression="gzip")
    hf.create_dataset('y', data=y_center.astype(np.float32), compression="gzip")
    hf.create_dataset('time', shape=(0,), maxshape=(None,), dtype='i4', chunks=(128,), compression="gzip")
    
    # 定义合理的块大小以优化I/O性能
    chunk_shape_features = (1, max(1, ny // 8), max(1, nx // 8), num_features)
    chunk_shape_labels = (1, max(1, ny // 8), max(1, nx // 8))

    hf.create_dataset('features', shape=(0, ny, nx, num_features), maxshape=(None, ny, nx, num_features),
                      dtype=np.float32, chunks=chunk_shape_features, compression="gzip")
    hf.create_dataset('labels', shape=(0, ny, nx), maxshape=(None, ny, nx),
                      dtype=np.uint8, chunks=chunk_shape_labels, compression="gzip")

    # 将元数据写入文件属性
    for key, value in METADATA_H5.items():
        if isinstance(value, dict):
            hf.attrs[key] = str(value)  # 字典存为字符串
        else:
            hf.attrs[key] = value

def main():
    """主执行函数，循环处理每年的每一天，并将结果存入HDF5文件。"""
    os.makedirs(HDF5_SAVE_PATH, exist_ok=True)
    h5_filepath = os.path.join(HDF5_SAVE_PATH, HDF5_FILENAME)
    
    first_write = not os.path.exists(h5_filepath)

    for year in range(START_YEAR, END_YEAR + 1):
        year_str = str(year)
        print(f"\n{'='*20} 开始处理年份: {year_str} {'='*20}")

        # --- 1. 加载年度数据 ---
        try:
            print(f"正在加载 {year_str} 年的数据...")
            xq, yq, xh, yh, time_coords, ssh_all, u_all, v_all = load_data(GEOGRAPHIC_EXTENT, year_str)
        except FileNotFoundError as e:
            print(f"警告: {e}. 跳过年份 {year_str}.")
            continue
        except Exception as e:
            print(f"加载 {year_str} 年数据时发生未知错误: {e}. 跳过年份。")
            continue

        # --- 2. 定义中心网格坐标和目标形状 ---
        try:
            x_center = xq[0, 1:-1]
            y_center = yq[1:-1, 0]
            ny, nx = len(y_center), len(x_center)
            expected_shape = (ny, nx)
            if nx <= 0 or ny <= 0: raise ValueError("中心网格维度无效。")
        except (IndexError, ValueError) as e:
            print(f"定义 {year_str} 年中心网格时出错: {e}. 跳过年份。")
            continue

        # --- 3. 按年处理，打开HDF5文件准备追加 ---
        try:
            with h5py.File(h5_filepath, 'a') as hf:
                if first_write:
                    initialize_hdf5_file(hf, x_center, y_center, expected_shape)
                    first_write = False

                current_time_size = hf['time'].shape[0]
                total_days = ssh_all.shape[0]

                # --- 4. 循环处理当年的每一天 ---
                for day_idx in range(total_days):
                    day_start_time = time.time()
                    try:
                        # --- a. 核心涡旋检测 ---
                        (result_cold, result_warm, ssh_center, vorticity_center,
                         ow_center, ow0_for_contour, u_center, v_center, new_date_str
                        ) = detect_eddy(
                            day_idx, xq, yq, xh, yh, ssh_all, u_all, v_all,
                            f"{year_str}0101", **PARAMS_EDDY
                        )
                        
                        # --- b. 准备特征矩阵 ---
                        velocity_mag_center = np.sqrt(u_center**2 + v_center**2)
                        
                        # 按METADATA中定义的顺序堆叠特征
                        feature_list = [
                            u_center, v_center, velocity_mag_center,
                            ow_center, vorticity_center, ssh_center
                        ]
                        
                        # 检查所有特征维度是否一致
                        if any(f.shape != expected_shape for f in feature_list):
                            print(f"错误: {new_date_str} 特征维度不匹配. 跳过当天。")
                            continue

                        # 填充NaN值并堆叠成 (ny, nx, num_features)
                        features_day = np.stack(
                            [np.nan_to_num(f.filled(np.nan) if hasattr(f, 'filled') else f) for f in feature_list],
                            axis=-1
                        ).astype(np.float32)

                        # --- c. 生成标签矩阵 ---
                        label_matrix = save_contour(
                            xq, yq, ow_center, ow0_for_contour,
                            result_warm, result_cold, new_date_str
                        )
                        if label_matrix.shape != expected_shape:
                            print(f"错误: {new_date_str} 标签维度不匹配. 跳过当天。")
                            continue

                        # --- d. 写入HDF5文件 ---
                        new_size = current_time_size + 1
                        hf['time'].resize(new_size, axis=0)
                        hf['features'].resize(new_size, axis=0)
                        hf['labels'].resize(new_size, axis=0)

                        hf['time'][current_time_size] = int(new_date_str)
                        hf['features'][current_time_size, ...] = features_day
                        hf['labels'][current_time_size, ...] = label_matrix
                        
                        current_time_size = new_size
                        elapsed_day = time.time() - day_start_time
                        print(f"  > 已处理并保存: {new_date_str} ({day_idx + 1}/{total_days}), 耗时 {elapsed_day:.2f}s")

                    except Exception:
                        print(f"---!!! 处理日期索引 {day_idx} (年份 {year_str}) 时发生严重错误 !!!---")
                        traceback.print_exc()
                        print(f"---!!! 跳过此日期 !!!---")
                        continue
        
        except Exception:
            print(f"---!!! 处理年份 {year_str} 期间发生HDF5文件操作或其他重大错误 !!!---")
            traceback.print_exc()
            print(f"---!!! 跳过此年份剩余部分 !!!---")
            continue

        print(f"{'-'*20} 完成处理年份: {year_str} {'-'*20}")

    print(f"\n处理完成。所有数据已保存至: {h5_filepath}")

if __name__ == "__main__":
    main()
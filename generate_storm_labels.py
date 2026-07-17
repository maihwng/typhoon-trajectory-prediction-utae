import pandas as pd
import numpy as np
import cv2
import h5py
import logging
import warnings
import os
from tqdm import tqdm

# Cấu hình logging khoa học
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# --- SIÊU THAM SỐ CẤU HÌNH ---
LON_MIN, LON_MAX = 100.0, 125.0
LAT_MIN, LAT_MAX = 0.0, 25.0
SPATIAL_RES = 0.25 
IMG_SIZE = 96
T_OBSERVATIONS = 5  # Số bước thời gian nhìn về quá khứ (Chuẩn WSTS+) 

def coord_to_pixel(lat, lon):
    x = int((lon - LON_MIN) / SPATIAL_RES)
    y = int((LAT_MAX - lat) / SPATIAL_RES)
    return x, y

def load_and_preprocess_ibtracs(csv_path):
    """Xử lý dữ liệu IBTrACS dựa trên chuẩn WMO[cite: 534, 546]."""
    logging.info(f"Nạp dữ liệu IBTrACS: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
    
    # Lọc bão theo mùa (Tháng 6-12) và mốc 6h [cite: 534]
    df = df[
        (df['ISO_TIME'].dt.year.between(2021, 2025)) &
        (df['ISO_TIME'].dt.month.between(6, 12)) &
        (df['ISO_TIME'].dt.hour.isin([0, 6, 12, 18])) &
        (df['LON'].between(LON_MIN, LON_MAX)) &
        (df['LAT'].between(LAT_MIN, LAT_MAX)) &
        (df['NATURE'] != 'ET')
    ]

    # Tính bán kính gió trung bình (R34) [cite: 546]
    radii_cols = ['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']
    for col in radii_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['RADIUS_DEG'] = df[radii_cols].mean(axis=1) / 60.0

    # Nội suy bán kính theo cường độ gió (Khớp bảng quy chuẩn North West Pacific)

    def impute_radius(row):
        if pd.notna(row['RADIUS_DEG']): return row['RADIUS_DEG']
        wind = row['USA_WIND']
        if pd.isna(wind) or wind < 34: return 1.5
        elif 34 <= wind < 64: return 2.0
        else: return 2.5      
    df['RADIUS_DEG'] = df.apply(impute_radius, axis=1)
    return df

def create_hdf5_wsts_format(df_storms, output_path):
    """Đóng gói dữ liệu thành chuỗi thời gian (T=5) chuẩn HDF5."""
    logging.info("Bắt đầu sinh ảnh Masks và đóng gói Sequence...") 
    all_frames = [] 

    # 1. Sinh tất cả các frame thô
    for year in range(2021, 2026):
        time_axis = pd.date_range(start=f'{year}-06-01', end=f'{year}-12-31 18:00', freq='6h')
        for current_time in tqdm(time_axis, desc=f"Rendering {year}"):
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            active_storms = df_storms[df_storms['ISO_TIME'] == current_time]          
            for _, storm in active_storms.iterrows():
                px_x, px_y = coord_to_pixel(storm['LAT'], storm['LON'])
                px_radius = max(1, int(storm['RADIUS_DEG'] / SPATIAL_RES))
                cv2.circle(mask, (px_x, px_y), px_radius, 1.0, -1)
            if not active_storms.empty:
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                if mask.max() > 0: mask /= mask.max()       
            all_frames.append(mask)

    # 2. Tạo Sliding Window (N, T, H, W)
    # Lưu ý: 'inputs' trong thực tế sẽ là ERA5, ở đây ta demo đóng gói chuỗi Label
    inputs_seq = []
    labels_target = []
    for i in range(len(all_frames) - T_OBSERVATIONS - 3):
        # Lấy 5 frame làm đầu vào 
        inputs_seq.append(np.stack(all_frames[i : i + T_OBSERVATIONS]))
        # Lấy frame kế tiếp làm nhãn mục tiêu
        labels_target.append(all_frames[i + T_OBSERVATIONS + 3])

    # 3. Ghi vào HDF5 nén
    logging.info(f"Đang ghi file H5: {output_path}")
    with h5py.File(output_path, 'w') as f:
        # 'inputs' và 'labels' là các key mô hình UTAE/ConvLSTM sẽ tìm
        f.create_dataset('inputs', data=np.array(inputs_seq), compression="gzip")
        f.create_dataset('labels', data=np.array(labels_target), compression="gzip")

if __name__ == "__main__":
    storm_df = load_and_preprocess_ibtracs("ibtracs.WP.list.v04r01.csv")
    create_hdf5_wsts_format(storm_df, "SCS_Typhoon_Dataset_T5.h5")
    logging.info("HOÀN TẤT.")
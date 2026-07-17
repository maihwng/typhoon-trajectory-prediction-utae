import cdsapi
import time
import os
import xarray as xr

URL = "https://cds.climate.copernicus.eu/api"
KEY = "79218e52-0a03-450f-b115-5fdc4ab09ee2"
c = cdsapi.Client(url=URL, key=KEY)

# Biển Đông
bounding_box = [25, 100, 0, 125]

years = ['2021', '2022', '2023', '2024']
months = ['06', '07', '08', '09', '10', '11', '12']
days = [str(i).zfill(2) for i in range(1, 32)]
times = ['00:00', '06:00', '12:00', '18:00']

os.makedirs('ERA5_Data', exist_ok=True)
os.makedirs('ERA5_Data/temp', exist_ok=True)

for year in years:
    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU TẢI DỮ LIỆU NĂM {year}")
    print(f"{'='*50}")

    file_surface = f"ERA5_Data/surface_{year}.nc"
    file_pressure = f"ERA5_Data/pressure_{year}.nc"

    # =====================================================
    # 1. TẢI SURFACE LEVELS (Toàn bộ năm)
    # =====================================================
    print(f"[1/2] Đang tải SURFACE variables năm {year}...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': [
                    'mean_sea_level_pressure',
                    'sea_surface_temperature',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'total_column_water_vapour',
                    'total_precipitation',
                ],
                'year': year,
                'month': months,
                'day': days,
                'time': times,
                'area': bounding_box,
            },
            file_surface
        )
        print("✅ Tải thành công Surface")
        print("⏳ Nghỉ 10 giây...")
        time.sleep(10)
    except Exception as e:
        print(f"Lỗi khi tải Surface: {e}")

    # =====================================================
    # 2. TẢI PRESSURE LEVELS (Chia theo từng tháng để tránh quá tải API)
    # =====================================================
    print(f"[2/2] Đang tải PRESSURE variables (chia tháng) năm {year}...")
    monthly_files = []
    
    for month in months:
        temp_file = f"ERA5_Data/temp/pressure_{year}_{month}.nc"
        monthly_files.append(temp_file)
        
        print(f"  -> Tải tháng {month}/{year}...")
        try:
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'data_format': 'netcdf',
                    'variable': [
                        'relative_humidity',
                        'vorticity',
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'geopotential',
                    ],
                    'pressure_level': ['200', '500', '850'],
                    'year': year,
                    'month': month,
                    'day': days,
                    'time': times,
                    'area': bounding_box,
                },
                temp_file
            )
        except Exception as e:
            print(f"Lỗi khi tải tháng {month}: {e}")
        
        print("  ⏳ Chờ 10 giây tránh rate limit...")
        time.sleep(10)

    # =====================================================
    # GỘP CÁC FILE THÁNG BẰNG XARRAY (Không cần dask)
    # =====================================================
    print(f"Đang gộp các file tháng thành {file_pressure}...")
    try:
        datasets = [xr.open_dataset(f, engine='netcdf4') for f in monthly_files]
        ds_merged = xr.concat(datasets, dim='valid_time')
        ds_merged.to_netcdf(file_pressure)
        
        # Đóng file để tránh khóa
        ds_merged.close()
        for ds in datasets:
            ds.close()
            
        print(f"✅ Gộp thành công {file_pressure}")
        
        # Xóa file tạm cho gọn
        for f in monthly_files:
            try:
                os.remove(f)
            except:
                pass
    except Exception as e:
        print(f"Lỗi khi gộp file: {e}")

    print(f"✅ Hoàn tất năm {year}")
    if year != years[-1]:
        print("⏳ Chờ 20 giây trước khi sang năm tiếp theo...")
        time.sleep(20)

print("\n🎉 HOÀN TẤT TOÀN BỘ ERA5 PIPELINE")

import xarray as xr

ds = xr.open_dataset(
    r"D:\BTL_TS\ERA5_Data\pressure_2021.nc",
    engine="netcdf4"
)

print(ds)
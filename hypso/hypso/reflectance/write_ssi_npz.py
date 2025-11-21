
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import csv

import os

script_dir = os.path.realpath(__file__)
script_dir = os.path.dirname(script_dir)


# Load the NetCDF file containing TSIS-1 spectrum
file_path = os.path.join(script_dir, "hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.nc")

print(file_path)

ds = xr.open_dataset(str(file_path))

# Access specific variables (example)
print("\nVariables:")
for var in ds.data_vars:
    print(f"{var}: {ds[var].shape}")

#print(ds["SSI"].values)
#print(ds["SSI"].values.shape)
#print(ds["Vacuum Wavelength"].values)
#print(ds["Vacuum Wavelength"].values.shape)

solar_x = ds["Vacuum Wavelength"].values
solar_y = ds["SSI"].values

#print("\nGlobal Attributes:")
#print(ds.attrs)

# Close the dataset
ds.close()

# Set these to the range of wavelengths you would like to be included in the .npz SSI file for generating ToA reflectance
start_lambda = 350
end_lambda = 850

start_lambda_idx = np.abs(solar_x - start_lambda).argmin()
end_lambda_idx = np.abs(solar_x - end_lambda).argmin()

print("Start wavelength:")
print(start_lambda_idx)
print(solar_x[start_lambda_idx])
print(solar_y[start_lambda_idx])

print("End wavelength:")
print(end_lambda_idx)
print(solar_x[end_lambda_idx])
print(solar_y[end_lambda_idx])

truncated_solar_x = solar_x[start_lambda_idx:end_lambda_idx+1]
truncated_solar_y = solar_y[start_lambda_idx:end_lambda_idx+1]

print(len(truncated_solar_x))
print(len(truncated_solar_y))

print(truncated_solar_x[-1])
print(truncated_solar_y[-1])

fn = os.path.join(script_dir, 'hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.npz')

np.savez(fn, solar_x=truncated_solar_x, solar_y=truncated_solar_y)


data = np.load(fn)

solar_x = data["solar_x"]
solar_y = data["solar_y"]
print(solar_x)

from satpy import Scene

import numpy as np
import xarray as xr
import pandas as pd

#scn = Scene()
#scn['my_dataset'] = Dataset(my_data_array, **my_info)



# Create a NumPy array representing some data
data = np.random.rand(3, 2)  # Example 3D array (e.g., time, lat, lon)

# Define dimensions and coordinates
dims = ('latitudes', 'longitudes')
coords = {
    'latitudes': [10.0, 20.0, 30.0],
    'longitudes': [100.0, 110.0]
}

# Create an xarray Dataset
ds = xr.Dataset(
    {'example_data': (dims, data)},
    coords=coords
)

scn = Scene()

scn['example_data'] = ds

print(scn['example_data'])
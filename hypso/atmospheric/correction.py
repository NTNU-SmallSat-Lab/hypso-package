import xarray as xr
import numpy as np


def add_acolite(self, nc_path, overlapSatImg=False):
    print("\n\n-------  Loading L2 Acolite Cube  ----------")
    # Extract corrected hypercube
    self.hypercube['L2'] = np.empty_like(self.hypercube['L1'])

    with xr.open_dataset(nc_path) as ncdat:
        keys = [i for i in ncdat.data_vars]
        try:
            keys.remove('lat')
            keys.remove('lon')
        except:
            print("Couldn't find lat and lon keys on Acolite .nc")

        toa_keys = [k for k in keys if 'rhos' not in k]
        surface_keys = [kk for kk in keys if 'rhot' not in kk]

        # Add Cube
        for i, k in enumerate(surface_keys):
            self.hypercube['L2'][:, :, i] = ncdat.variables[k].values

    # Get Chlorophyll
    self.chlEstimator.ocx_estimation(
        satName='hypsoacolite', overlapSatImg=overlapSatImg)

    print("Done Importing Acolite L2 Data")

    return self.hypercube['L2']

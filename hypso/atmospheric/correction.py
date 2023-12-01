import xarray as xr
import numpy as np
from importlib.resources import files
from urllib.parse import urlparse, unquote
from hypso.utils import MyProgressBar
import urllib.request
import tarfile
from pathlib import Path
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

def get_acolite_repo():
    github_url = r"https://github.com/acolite/acolite/archive/refs/tags/20231023.0.tar.gz"
    filename = unquote(urlparse(github_url).path.split("/")[-1])

    output_filename=files(
        'hypso.atmospheric').joinpath(f'data/{filename}')

    try:
        urllib.request.urlretrieve(url=github_url,
                                   filename=output_filename,
                                   reporthook = MyProgressBar(filename))

    except Exception as err:
        print(f"Download Failed. {err}")
        print(f"Deleting {filename}")
        # If fail, delete
        output_filename.unlink(missing_ok=True)

    # Uncompress
    tar = tarfile.open(output_filename, 'r:gz')
    dst_path = files(
        'hypso.atmospheric').joinpath(f'data/')
    tar.extractall(dst_path)
    tar.close()

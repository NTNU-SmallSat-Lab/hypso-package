# HYPSO
"hypso" is a simple, fast, visualization tool for the hyperspectral
images taken by the HYPSO mission from the Norwegian University of Science and
Technology (NTNU) for Python >3.9

- Documentation: https://ntnu-smallsat-lab.github.io/hypso-package/
  
- Anaconda URL: https://anaconda.org/conda-forge/hypso

- Anaconda Github Feedstock: https://github.com/conda-forge/hypso-feedstock

## How to Install (or Update)
It is recommended to use mamba as it is less prone to errors in dependency management than the default conda terminal (see https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

if conda installed:
```
conda install -c conda-forge conda-libmamba-solver 
conda config --set solver libmamba
conda create -n hypsoenv python=3.9
conda activate hypsoenv
conda install -c conda-forge hypso
```

if mamba installed:
```
mamba create -n hypsoenv python=3.9
mamba activate hypsoenv
mamba install -c conda-forge hypso
```

To update to the most recent version it is suggested to run the following code (change "mamba" for "conda" if needed):
```
mamba search -c conda-forge hypso
mamba update hypso
```

## Usage (see demo.ipynb for expanded version)
```
from hypso import Hypso

# Define HYPSO Image File and the .points from QGIS (if available)
hypso_file_path="/Documents/mjosa_2023-06-15_0948Z-l1a.nc"
points_path = "/Documents/mjosa_2023-06-15_0948Z-rgba_8bit.tif.points" #georeferencing

# Create Satellite Object
# l1b.nc is Generated when loading l1a.nc for the first time
satobj = Hypso(hypso_file_path, points_path=points_path)

from hypso.plot import show_rgb_map

# Show Image on top of map
show_rgb_map(satobj, plotTitle="Mjøsa 15-062023 09:48AM",dpi_input=250)
```
![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/d5cf9416-7843-47fc-b262-93227882d9f0)

```
from hypso.experimental.chlorophyll import start_chl_estimation

start_chl_estimation(sat_obj=satobj, model_path="/Users/alvaroflores/Documents/model_6sv1_aqua_tuned.joblib")

from hypso.plot import plot_array_overlay
# Plot Array on Map
plot_array_overlay(satobj,satobj.chl, plotTitle="6SV1 Estimation",cbar_title="Chlorophyll Values Sample 0 to 100", min_value=0.01, max_value=100)
```
![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/e5e905b3-8cd6-490d-9c66-50cfa0fa948c)

## Authors

- Package: @DevAlvaroF
- Correction Coefficients: Marie Henriksen, Joe Garett
- Georeferencing: Sivert Bakken, Dennis Langer


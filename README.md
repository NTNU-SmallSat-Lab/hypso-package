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

## Manually Geo-Referencing (QGIS)
If after plotting the image on the map (See demo.ipynb), a mismatch is observed, manual georeferencing needs to be done by selecting matching points between the HYPSO capture and a reference map (OSM is recommended).

At least 6 points are recommended to be selected, although more is better. If possible, select regions across the entire image for a better correction.

Steps:

0. In QGIS install the plugin "QuickMapServices" as it contains OpenStreeMaps (OSM) by default (see https://plugins.qgis.org/plugins/quick_map_services/)
1. Drag and drop into QGIS the GeoTiff which contains the name "rgba_8bit". You can use the one in the folder "geotiff" which is inside the HYPSO folder <br>
   ![1](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/8b7837e5-9555-4d02-a4ef-a1c6275e19bf)

2. Select the option in the menu from Web -> QuickMapServices -> OSM -> OSM Standard. The map will load as well as your image <br>
   ![2](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/9293deab-bff9-4c7f-9b4c-3ec000447a81) <br>


3. On the "Layer" window (left side) place the cursor on top of the *OSM standard* and verify the CRS as EPSG:3857 <br>
   ![4](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/a8f6143d-ab89-4f17-b7b1-074bb4de0579) <br>

4. Go to the menu `Layer -> Georeferencer...`<br>
   ![5](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/0e234a16-d275-4ff9-94be-2d0dd753c239) <br>

5. Click `File -> Open Raster...` <br>
   ![6](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/254454fe-b43f-4fc5-b8cd-69126834da55) 

6. You can deactivate the layer of the first image you dragged on the "Layers" window <br>
   ![7](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/e94ad7d6-063a-4d2e-a661-54bdfee91630) <br>

7. In the georeferencer window, click an area in the HYPSO image that you can observe on both the HYPSO capture and the OSM map (see green dot in image). A new window will appear called "Enter Map Coordinates". Click on the botton "From Map Canvas". The window will minimize and will wait for you to now click the same area but in the OSM map <br>
   ![8](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/a5394d0a-56c0-4d54-ad25-7b560e901cee) <br>

8. The window will appear again with coordinates and 2 green dots will be visible in the HYPSO capure and the OSM map. Click OK <br>
    ![9](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/7ac6ba83-5df1-48b3-a9f0-a11f8cefb33f) <br>

9. At the bottom of the Georeferencer window the match will appear, repeat the process until at least 6 points are selected. <br>
    ![10](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/96153ad4-28ab-4725-8556-e2045ae02467) <br>

10. Once you are done go to the menu in the georeferencer window `File -> Save GCP Points as...` and save the `"*.points"` file in the same directory as the HYPSO root folder (it has to be inside) <br>
    ![11](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/e7842ec7-db2d-456f-8afe-f30682b87d02) <br>

11. Reload the HYPSO image again in Python (if you are using Jupyter notebooks, restart the kernel). <br>

**IMPORTANT**: CRS to be used in the OSM map is 'epsg:3857', use default for HYPSO or the match will fail.

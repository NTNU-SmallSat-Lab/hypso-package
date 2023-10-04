# HYPSO
"hypso" is a simple, fast, visualization tool for the hyperspectral
images taken by the HYPSO mission from the Norwegian University of Science and
Technology (NTNU) for Python >3.9

## How to Install
It is recommended to use mamba as it is less prone to errors in dependency management than the default conda terminal (see https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install)

```
conda install hypso
```

if mamba installed:
```
mamba install hypso
```

## Pipeline (on Load)
1. Load metadata from existing files `get_metainfo()`
2. Gets "Raw Cube" (L1A) `get_raw_cube()`
3. Gets Calibration Coefficients (radiometric, smile and striping) Path depending on the image size `get_calibration_coefficients_path()`
4. Extracts Coefficients `get_coefficients_from_dict()`
5. Gets Spectral Coefficients Path (wavelengths) `get_spectral_coefficients_path()`
6. Extracts Spectral Coefficients `get_coefficients_from_file()`
7. Calibration and Correction of Cube to get L1B (Radiometric -> Smile -> Destriping) `get_calibrated_and_corrected_cube()`
8. Get Projection metadata from GeoTiff `get_projection_metadata()`
9. Coordinate Correction using the `"*.points*"`file (if exists) `start_coordinate_correction()` (Steps to do it manually are at the bottom of this README)
10. Georeference and create 120 bands GeoTiff (L1C) in a new "geotiff_full" directory `generate_full_geotiff()`

**NOTE:** If corrections are made to the `"*.points"` file, delete the "geotiff_full" directory and run the script again to create the GeoTiffs again.

WARNING: Do not extract the LAT and LON values from files in the "geotiff" folder, use the GeoTiffs in the "geotiff_full" directory instead because they are corrected.
        

## Sub-Modules (Alphabetical Order)

### a) calibration
-Radiometric correction as well as smile and destriping process
### b) classification
- Water mask is generated as a binary file here (original method from https://github.com/cordmaur/WaterDetect/)
- Further implementations of on-board classification (To Be Implemented)
### c) experimental
New features to be tested
- Atmospheric Correction 6SV1 (To be Implemented)
- Atmospheric Correction ACOLITE (To Be Implemented)

### d) exportfiles
- Export of .nc files
- Print attributes and groups of .nc files
### e) georeference
Used for:
- Creating 120-band GeoTiff
- Correction of latitude and longitude with `"*.points"` file

### f) plot
- Show RGB image on Map
- Export .png rgb
- Plot 2D array in same coordinates as image (to visualize results of any algorithm)
## Usage (see demo.ipynb for expanded version)
```
from hypso import Satellite

# HYPSO Image Directory
hypso_dir = r"/Users/alvaroflores/Documents/mjosa-06-15_0948Z"

# Create Satellite Object
satobj = SatelliteClass(hypso_dir)

from hypso.plot import show_rgb_map

# Show RGB on top of Map
show_rgb_map(satobj, plotTitle="Mjosa 15-062023 09:48AM",dpi_input=250)
```
![output](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/44f939bb-9435-4688-9194-6b08b172fc36)


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


3. On the "Layer" window (left side) place the cursor on top of the HYPSO image and check the CRS is EPSG:32632 <br>
   ![3](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/753eec2b-671f-4f5e-b78c-3495eccad831) <br>


4. Do the same for the OSM standard and verify the CRS as EPSG:3857 <br>
   ![4](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/a8f6143d-ab89-4f17-b7b1-074bb4de0579) <br>

5. Go to the menu `Layer -> Georeferencer...`<br>
   ![5](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/0e234a16-d275-4ff9-94be-2d0dd753c239) <br>

6. Click `File -> Open Raster...` <br>
   ![6](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/254454fe-b43f-4fc5-b8cd-69126834da55) 

7. You can deactivate the layer of the first image you dragged on the "Layers" window <br>
   ![7](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/e94ad7d6-063a-4d2e-a661-54bdfee91630) <br>

8. In the georeferencer window, click an area in the HYPSO image that you can observe on both the HYPSO capture and the OSM map (see green dot in image). A new window will appear called "Enter Map Coordinates". Click on the botton "From Map Canvas". The window will minimize and will wait for you to now click the same area but in the OSM map <br>
   ![8](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/a5394d0a-56c0-4d54-ad25-7b560e901cee) <br>

9. The window will appear again with coordinates and 2 green dots will be visible in the HYPSO capure and the OSM map. Click OK <br>
    ![9](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/7ac6ba83-5df1-48b3-a9f0-a11f8cefb33f) <br>

10. At the bottom of the Georeferencer window the match will appear, repeat the process until at least 6 points are selected. <br>
    ![10](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/96153ad4-28ab-4725-8556-e2045ae02467) <br>

11. Once you are done go to the menu in the georeferencer window `File -> Save GCP Points as...` and save the `"*.points"` file in the same directory as the HYPSO root folder (it has to be inside) <br>
    ![11](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/e7842ec7-db2d-456f-8afe-f30682b87d02) <br>

12. Reload the HYPSO image again in Python (if you are using Jupyter notebooks, restart the kernel). <br>

**IMPORTANT**: CRS to be used in the OSM map is 'epsg:3857', use default for HYPSO or the match will fail.

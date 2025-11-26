import numpy as np
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
#from matplotlib.path import Path
import matplotlib

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import Polygon
from dateutil import parser
import earthaccess
import dateutil

import Py6S
from tqdm import tqdm

from datetime import datetime, timedelta

from .dem import MeanDEM



def _extract_footprint_and_date(satobj):

    try:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
    except:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes


    # Extract edge pixels to make a precise footprint polygon
    edge_lons = np.concatenate([
        longitudes[0, :],        # top
        longitudes[:, -1],       # right
        longitudes[-1, ::-1],    # bottom reversed
        longitudes[::-1, 0]      # left reversed
    ])
    edge_lats = np.concatenate([
        latitudes[0, :],
        latitudes[:, -1],
        latitudes[-1, ::-1],
        latitudes[::-1, 0]
    ])

    footprint_precise = Polygon(zip(edge_lons, edge_lats))

    # Simple bounding rectangle (CCW)
    min_lon, min_lat, max_lon, max_lat = footprint_precise.bounds
    simple_polygon_ccw = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, max_lat),
        (max_lon, min_lat),
        (min_lon, min_lat)
    ]

    # Extract date from file name using regex
    file_date = satobj.unixtime
    temporal_range = (file_date - timedelta(hours=12), file_date + timedelta(hours=12))


    return footprint_precise, simple_polygon_ccw, temporal_range


def download_viirs_aot(footprint_polygon, temporal_range, local_path='data_aerosol'):
    """
    Search and download VIIRS AOT 550 nm granules for given footprint and temporal range.

    Parameters:
        footprint_polygon : shapely.geometry.Polygon
            The area of interest.
        temporal_range : tuple(date, date)
            Start and end date for search.
        local_path : str
            Folder to save downloaded granules.

    Returns:
        files : list of str
            Paths to downloaded granules.
    """
    # Log in (only once per session)
    earthaccess.login()

    print(earthaccess.status())
    
    # Satellites to query
    short_names = ['AERDB_L2_VIIRS_SNPP', 'AERDB_L2_VIIRS_NOAA20']
    
    # Extract bounding box from footprint
    min_lon, min_lat, max_lon, max_lat = footprint_polygon.bounds
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    
    all_results = []
    for sn in short_names:
        results = earthaccess.search_data(
            short_name=sn,
            temporal=temporal_range,
            bounding_box=bounding_box
        )
        all_results.extend(results)
        print(f"Found {len(results)} granules for {sn}")
    
    print(f"\nTotal granules found: {len(all_results)}")
    
    # Download the files
    try:
        files = earthaccess.download(all_results, local_path=local_path)
        print(f"Downloaded {len(files)} files to {local_path}")
    except ValueError as ex:
        print("[WARNING] No VIIRS files downloaded.")
        files = None
    
    return files


def get_aot_in_swath(files_f, footprint_precise, latitudes, longitudes, aot_var="Aerosol_Optical_Thickness_550_Land_Ocean", name="default", local_path='data_aerosol'):
    """
    Plot VIIRS AOT 550 nm data from a list of NetCDF files and filter by a swath footprint polygon.
    
    Parameters:
    - files_f: list of file paths (NetCDF files)
    - footprint_precise: shapely Polygon defining the footprint
    - bounding_box: optional [lon_min, lat_min, lon_max, lat_max] for zooming the plot
    - aot_var: string, name of the AOT variable in NetCDF
    """
    all_aot, all_lat, all_lon = [], [], []

    # Read all files
    for f in files_f:
        #print(f)
        ds = xr.open_dataset(f, decode_timedelta=False)
        if aot_var in ds:
            print("Found AOT data.")
            aot = ds[aot_var]
            lat = ds["Latitude"]
            lon = ds["Longitude"]

            mask = aot > 0
            all_aot.append(aot.where(mask).values.flatten())
            all_lat.append(lat.where(mask).values.flatten())
            all_lon.append(lon.where(mask).values.flatten())
        ds.close()

    
    # Concatenate
    all_aot = np.concatenate(all_aot)
    all_lat = np.concatenate(all_lat)
    all_lon = np.concatenate(all_lon)

    lon_min = np.min(longitudes)
    lat_min = np.min(latitudes)

    lon_max = np.max(longitudes)
    lat_max = np.max(latitudes)


    savefig_path = Path(local_path)
    savefig_path.joinpath(str(name) + ".png")

    #'''
    # Scatter plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': cartopy.crs.PlateCarree()})
    pcm = ax.scatter(all_lon, all_lat, c=all_aot, cmap='viridis', s=1, alpha=0.7)

    # Coastlines, borders, grid
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Polygon boundary
    lon_boundary, lat_boundary = footprint_precise.boundary.xy

    ax.plot(lon_boundary, lat_boundary, color='red', linewidth=2, label='Swath Footprint')
    #ax.set_xlim(bounding_box[0] - 5, bounding_box[2] + 5)  # lon_min, lon_max
    #ax.set_ylim(bounding_box[1] - 5, bounding_box[3] + 5)  # lat_min, lat_max
    ax.set_xlim(lon_min - 5, lon_max + 5)  # lon_min, lon_max
    ax.set_ylim(lat_min - 5, lat_max + 5)  # lat_min, lat_max

    # Colorbar
    fig.colorbar(pcm, ax=ax, label='AOT 550 nm')
    ax.set_title('VIIRS Deep Blue Aerosol Optical Thickness (550 nm) - All Granules')


    ax.legend()
    #plt.show()
    #plt.show(block=True) 
    plt.savefig(savefig_path)
    plt.close()
    #'''


    poly_coords = np.array(footprint_precise.exterior.coords)
    poly_path = matplotlib.path.Path(poly_coords)
    points = np.vstack((all_lon, all_lat)).T

    # Vectorized check: True if inside footprint
    inside_mask = poly_path.contains_points(points)

    # Apply mask to AOT data
    aot_inside = all_aot[inside_mask]

    return all_aot, all_lat, all_lon, aot_inside









def BasicParameters(wavelengths: np.ndarray, 
                    #radiance_cube: np.ndarray, 
                    lat_2d_array: np.ndarray,
                    lon_2d_array: np.ndarray, 
                    solar_azimuth_angles: np.ndarray,
                    solar_zenith_angles: np.ndarray,
                    sat_azimuth_angles: np.ndarray,
                    sat_zenith_angles: np.ndarray,
                    iso_time: str,
                    dem_path: Path = None,
                    x_coord: int = None,
                    y_coord: int = None
                    ) -> dict:
    """
    Get the parameters you need for 6s atmospheric correction

    :param wavelengths: Wavelengths corresponding to the spectral image
    :param radiance_cube: Hypercube of the spectral image L1B
    :param hypso_info: Dictionary containing the information of the spectral image
    :param lat_2d_array: 2D latitude array of the spectral image
    :param lon_2d_array: 2D longitude array of the spectral image
    :param time_capture: Time of the capture

    :return: Dictionary of the Basic Paramters for the PY6SV1 correction method
    """

    # -------------------------------------------------
    #               Solar Parameters
    # -------------------------------------------------
    SixsParameters = dict()


    #time_capture = parser.parse(iso_time)
    time_capture = dateutil.parser.parse(iso_time)
    SixsParameters['time'] = time_capture

    SixsParameters['wavelengths'] = wavelengths
    #SixsParameters['radiance_cube'] = radiance_cube

    # Solar zenith angle, azimuth (average)
    if x_coord is not None and y_coord is not None:
        SixsParameters["SolarZenithAngle"] = solar_zenith_angles[y_coord, x_coord]
        SixsParameters["SolarAzimuthAngle"] = solar_azimuth_angles[y_coord, x_coord]
    else:
        SixsParameters["SolarZenithAngle"] = np.mean(solar_zenith_angles)
        SixsParameters["SolarAzimuthAngle"] = np.mean(solar_azimuth_angles)

    # -------------------------------------------------
    #               Satellite Parameters
    # -------------------------------------------------
    # Satellite zenith angle, azimuth
    ViewZeniths = dict()
    ViewAzimuths = dict()
    # Make an 120 array with the average zenith and azimuth angle for every band
    # Ideally the average should be per band but we only have one 2D array so we use the same for every one

    if x_coord is not None and y_coord is not None:
        for i in range(120):
            ViewZeniths[i] = sat_zenith_angles[y_coord, x_coord]
            ViewAzimuths[i] = sat_azimuth_angles[y_coord, x_coord]

    else:
        for i in range(120):
            ViewZeniths[i] = np.mean(sat_zenith_angles)
            ViewAzimuths[i] = np.mean(sat_azimuth_angles)

    SixsParameters["SatZenithAngles"] = ViewZeniths
    SixsParameters["SatAzimuthAngles"] = ViewAzimuths

    # -------------------------------------------------
    #                      Date
    # -------------------------------------------------
    # Date:Month, Day
    Date = dateutil.parser.isoparse(iso_time)
    SixsParameters["ImgMonth"] = int(Date.month)
    SixsParameters["ImgDay"] = int(Date.day)

    # -------------------------------------------------
    #                Lat Lon Bounding Box
    # -------------------------------------------------
    lat = lat_2d_array
    lon = lon_2d_array

    # UPPER
    ULLat = lat[0, 0]
    # URLat = lat[0, -1]

    ULLon = lon[0, 0]
    # URLon = lon[0, -1]

    # BOTTOM
    # BLLat = lat[-1, 0]
    BRLat = lat[-1, -1]

    # BLLon = lon[-1, 0]
    BRLon = lon[-1, -1]

    min_lat = np.nanmin(lat)
    max_lat = np.nanmax(lat)

    min_lon = np.nanmin(lon)
    max_lon = np.nanmax(lon)

    print(f"ROI:\nMax Lat: {max_lat}  Min Lat: {min_lat}\nMax Lon: {max_lon}  Min Lon: {min_lon}")

    ImageCenterLon = (ULLon + BRLon) / 2
    ImageCenterLat = (ULLat + BRLat) / 2

    ImageCenterLon = np.mean([min_lon, max_lon])
    ImageCenterLat = np.mean([min_lat, max_lat])

    # Atmospheric mode type
    if -15 < ImageCenterLat <= 15:
        SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

    elif (15 < ImageCenterLat <= 45) or (-45 <= ImageCenterLat < -15):
        if 4 < SixsParameters["ImgMonth"] <= 9:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeSummer)
        else:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeWinter)

    elif (45 < ImageCenterLat <= 60) or (-60 <= ImageCenterLat < -45):
        if 4 < SixsParameters["ImgMonth"] <= 9:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticSummer)
        else:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticWinter)

    rounded_lat = round(ImageCenterLat, -1)

    # data from Table 2-2 in http://www.exelisvis.com/docs/FLAASH.html
    SAW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticWinter)
    SAS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticSummer)
    MLS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeSummer)
    MLW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeWinter)
    T = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

    ap_JFMA = {
        80: SAW,
        70: SAW,
        60: MLW,
        50: MLW,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_MJ = {
        80: SAW,
        70: MLW,
        60: MLW,
        50: SAS,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_JA = {
        80: MLW,
        70: MLW,
        60: SAS,
        50: SAS,
        40: MLS,
        30: T,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: MLS,
        -30: MLS,
        -40: SAS,
        -50: MLW,
        -60: MLW,
        -70: MLW,
        -80: SAW,
    }

    ap_SO = {
        80: MLW,
        70: MLW,
        60: SAS,
        50: SAS,
        40: MLS,
        30: T,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: MLS,
        -30: MLS,
        -40: SAS,
        -50: MLW,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_ND = {
        80: SAW,
        70: SAW,
        60: MLW,
        50: SAS,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_dict = {
        1: ap_JFMA,
        2: ap_JFMA,
        3: ap_JFMA,
        4: ap_JFMA,
        5: ap_MJ,
        6: ap_MJ,
        7: ap_JA,
        8: ap_JA,
        9: ap_SO,
        10: ap_SO,
        11: ap_ND,
        12: ap_ND,
    }

    SixsParameters["AtmosphericProfile"] = ap_dict[Date.month][rounded_lat]


    if dem_path is not None:
        # Find the DEM height by studying the range of the area.
        pointUL = dict()
        pointDR = dict()
        pointUL["lat"] = ULLat
        pointUL["lon"] = ULLon
        pointDR["lat"] = BRLat
        pointDR["lon"] = BRLon

        # Modifications made due to HYPSO 2D Lat/Lon array not being squares, they may be skewed
        pointUL["lat"] = max_lat
        pointUL["lon"] = min_lon
        pointDR["lat"] = min_lat
        pointDR["lon"] = max_lon

        # Look up elevation at a single point
        if x_coord is not None and y_coord is not None:

            pointUL["lat"] = lat[y_coord, x_coord]
            pointUL["lon"] = lon[y_coord, x_coord]
            pointDR["lat"] = lat[y_coord, x_coord]
            pointDR["lon"] = lon[y_coord, x_coord]

            pointLat = pointUL["lat"]
            pointLon = pointUL["lon"]

            print(f"Location: {pointLat}, {pointLon}")

            mean_elevation = (MeanDEM(pointUL, pointDR, dem_path)) * 0.001

        else:
            mean_elevation = (MeanDEM(pointUL, pointDR, dem_path)) * 0.001
            
        print("meanDEM:")
        print(mean_elevation)
        SixsParameters["meanDEM"] = mean_elevation
    
    else:
        SixsParameters["meanDEM"] = 0

    # -------------------------------------------------
    #                Other Parameters
    # -------------------------------------------------
    # aerosol type continent
    #SixsParameters["AeroProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Maritime)
    SixsParameters["AeroProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Continental)

    # Underlying surface type
    #SixsParameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(0.26)
    #SixsParameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(0.05)
    SixsParameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(Py6S.GroundReflectance.LakeWater)
    #SixsParameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(Py6S.GroundReflectance.)

    # 550nm aerosol optical thickness, obtained from MODIS based on date.
    # https://neo.gsfc.nasa.gov/analysis/index.php
    SixsParameters['aot550'] = 0.14497  # Constant value changed later if supplied
    SixsParameters['aot550'] = None

    # TOA Approach without SRF *****************************************************************

    # Non-homogeneous lower bedding surface, Lambertian
    # This should be used when the SRF is not known and it can be obtained from the TOA Reflectance
    # This is non tested example --------------------------------
    # Running for each wavelength with its respective reflectance value
    # for (i, j) in zip(wave, toa_reflec):
    #     s.wavelength = Wavelength(i)
    #     s.atmos_corr = Py6S.AtmosCorr.AtmosCorrLambertianFromReflectance(j)
    #     s.run()
    #     print "A18 wavelength: ", i, "Ref. : ", j, " ",
    #     boa_rec = s.outputs.atmos_corrected_reflectance_lambertian

    # s.atmos_corr = SixsInputParameter['AtmosCorrection']

    # Non-homogeneous lower bedding surface, Lambertian
    # SixsParameters['AtmosCorrection'] = Py6S.AtmosCorr.AtmosCorrLambertianFromReflectance(-0.1)
    # *****************************************************************

    return SixsParameters








def run_py6s(wavelengths: np.ndarray, 
             #radiance_cube: np.ndarray,
             reflectance_cube: np.ndarray, 
             lat_2d_array: np.ndarray,
             lon_2d_array: np.ndarray, 
             solar_azimuth_angles: np.ndarray,
             solar_zenith_angles: np.ndarray,
             sat_azimuth_angles: np.ndarray,
             sat_zenith_angles: np.ndarray,
             iso_time: str,
             py6s_dict: dict, 
             dem_path: Path = None
             ) -> np.ndarray:


    """
    Run the PY6S atmospheric correction on the Hypso spectral image

    :param wavelengths: Wavelengths corresponding to the spectral image
    :param radiance_cube: Hypercube of the spectral image L1B
    :param lat_2d_array: 2D latitude array of the spectral image
    :param lon_2d_array: 2D longitude array of the spectral image
    :param py6s_dict: Dictionary containing the PY6S information for atmospheric correction
    :param iso_time: Time of the capture

    :return: Return 3-channel surface reflectance spectral image
    """

    print("\n-------  Py6S Atmospheric Correction  ----------")

    # Original units mW  (m^{-2} sr^{-1} nm^{-1})
    # radiance_cube = radiance_cube / 1000 # mW to W -> W  (m^{-2} sr^{-1} nm^{-1})
    # radiance_cube = radiance_cube / 0.001

    cube = reflectance_cube

    height, width, depth = cube.shape



    rho_R_values = np.empty_like(depth)
    rho_A_R_values = np.empty_like(depth)
    Tg_H20_values = np.empty_like(depth)
    Tg_O3_values = np.empty_like(depth)
    #Tg_OG_values = np.empty_like(depth)
    Ts_Tv_values = np.empty_like(depth)
    S_atm_values = np.empty_like(depth)



    init_parameters = BasicParameters(wavelengths=wavelengths, 
                                    #radiance_cube=radiance_cube, 
                                    lat_2d_array=lat_2d_array, 
                                    lon_2d_array=lon_2d_array, 
                                    solar_azimuth_angles=solar_azimuth_angles,
                                    solar_zenith_angles=solar_zenith_angles,
                                    sat_azimuth_angles=sat_azimuth_angles,
                                    sat_zenith_angles=sat_zenith_angles,
                                    iso_time=iso_time,
                                    dem_path=dem_path
                                    #x_coord=x_coord,
                                    #y_coord=y_coord
                                    )

    # Combining two dictionaries into init_parameters
    init_parameters.update(py6s_dict)

    SixsInputParameter = init_parameters

    for BandId in tqdm(range(120)):


        # Part I
        # Run 6S to calculate Rayleigh reflectance
        # https://blog.rtwilson.com/calculating-rayleigh-reflectance-using-py6s/


        # 6S Models
        s = Py6S.SixS()

        # Enable Sensor type customization
        s.geometry = Py6S.Geometry.User()

        # Add Geometry Parameters
        s.geometry.solar_z = SixsInputParameter["SolarZenithAngle"]
        s.geometry.solar_a = SixsInputParameter["SolarAzimuthAngle"]
        s.geometry.view_z = SixsInputParameter["SatZenithAngles"][BandId]
        s.geometry.view_a = SixsInputParameter["SatAzimuthAngles"][BandId]

        # Date: Month, Day
        s.geometry.month = SixsInputParameter["ImgMonth"]
        s.geometry.day = SixsInputParameter["ImgDay"]


        s.altitudes.set_sensor_satellite_level()
        s.altitudes.set_target_sea_level()

        
        s.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.NoAerosols)
        s.atmos_profile = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.NoGaseousAbsorption)
        s.ground_reflectance = SixsInputParameter["GroundReflectance"]

        # Study area altitude, satellite sensor orbit altitude
        s.altitudes = Py6S.Altitudes()

        s.altitudes.set_target_custom_altitude(SixsInputParameter["meanDEM"])
        s.altitudes.set_sensor_satellite_level()


        current_band_wl = SixsInputParameter["wavelengths"][BandId] / 1000 # convert to micrometers

        s.wavelength = Py6S.Wavelength(current_band_wl)

        s.run()

        rho_R = s.outputs.atmospheric_intrinsic_reflectance






        # Part II
        # Run 6S with AOD

        # 6S Models
        s = Py6S.SixS()

        # Enable Sensor type customization
        s.geometry = Py6S.Geometry.User()

        # Add Geometry Parameters
        s.geometry.solar_z = SixsInputParameter["SolarZenithAngle"]
        s.geometry.solar_a = SixsInputParameter["SolarAzimuthAngle"]
        s.geometry.view_z = SixsInputParameter["SatZenithAngles"][BandId]
        s.geometry.view_a = SixsInputParameter["SatAzimuthAngles"][BandId]

        # Date: Month, Day
        s.geometry.month = SixsInputParameter["ImgMonth"]
        s.geometry.day = SixsInputParameter["ImgDay"]

        # Type of atmospheric pattern
        s.atmos_profile = SixsInputParameter["AtmosphericProfile"]

        # Target Features
        #SixsInputParameter["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(0)
        #SixsInputParameter["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(Py6S.GroundReflectance.LakeWater)
        s.ground_reflectance = SixsInputParameter["GroundReflectance"]

        # Aerosol Profile
        s.aero_profile = SixsInputParameter["AeroProfile"]  # Aerosol Type (Maritime Here)

        if 'aot550' in SixsInputParameter.keys():
            s.aot550 = SixsInputParameter['aot550']
            # Update AOT and Aero_profile if data provided
        elif 'aeronet' in SixsInputParameter.keys():
            s = Py6S.SixSHelpers.Aeronet.import_aeronet_data(s, SixsInputParameter['aeronet'], SixsInputParameter['time'])
        else:
            # Use Default Values
            s.aot550 = 0.14497  # Value checked from website for scene


        # Study area altitude, satellite sensor orbit altitude
        s.altitudes = Py6S.Altitudes()

        s.altitudes.set_target_custom_altitude(SixsInputParameter["meanDEM"])
        s.altitudes.set_sensor_satellite_level()


        current_band_wl = SixsInputParameter["wavelengths"][BandId] / 1000 # convert to micrometers

        s.wavelength = Py6S.Wavelength(current_band_wl)

        s.run()

        rho_A_R = s.outputs.atmospheric_intrinsic_reflectance
        Tg_H20 = s.outputs.trans['water'].total
        Tg_O3 = s.outputs.trans['ozone'].total
        Tg_OG = 1.0
        Ts_Tv = s.outputs.trans['total_scattering'].total
        S_atm = s.outputs.spherical_albedo.total



        #with open(str(BandId)+'_' + str(current_band_wl) + 'nm_output.txt', 'w') as file:
        #    file.write(s.outputs.fulltext)



        # Part III write outputs

        rho_R_values[BandId] = rho_R
        rho_A_R_values[BandId] = rho_A_R
        Tg_H20_values[BandId] = Tg_H20
        Tg_O3_values[BandId] = Tg_O3
        #Tg_OG_values[BandId] = Tg_OG
        Ts_Tv_values[BandId] = Ts_Tv
        S_atm_values[BandId] = S_atm


                
        # Linear 1D Interp to Fill Values skipped due to AOT Variances
        spectra = rho_R_values[:]
        wl = init_parameters['wavelengths']
        nans = np.isnan(spectra)
        spectra[nans] = np.interp(wl[nans], wl[~nans], spectra[~nans])
        interp_spectra = spectra
        rho_R_values[:] = interp_spectra

        spectra = rho_A_R_values[:]
        wl = init_parameters['wavelengths']
        nans = np.isnan(spectra)
        spectra[nans] = np.interp(wl[nans], wl[~nans], spectra[~nans])
        interp_spectra = spectra
        rho_A_R_values[:] = interp_spectra

        spectra = Tg_H20_values[:]
        wl = init_parameters['wavelengths']
        nans = np.isnan(spectra)
        spectra[nans] = np.interp(wl[nans], wl[~nans], spectra[~nans])
        interp_spectra = spectra
        Tg_H20_values[:] = interp_spectra

        spectra = Tg_O3_values[:]
        wl = init_parameters['wavelengths']
        nans = np.isnan(spectra)
        spectra[nans] = np.interp(wl[nans], wl[~nans], spectra[~nans])
        interp_spectra = spectra
        Tg_O3_values[:] = interp_spectra

        spectra = Ts_Tv_values[:]
        wl = init_parameters['wavelengths']
        nans = np.isnan(spectra)
        spectra[nans] = np.interp(wl[nans], wl[~nans], spectra[~nans])
        interp_spectra = spectra
        Ts_Tv_values[:] = interp_spectra

        spectra = S_atm_values[:]
        wl = init_parameters['wavelengths']
        nans = np.isnan(spectra)
        spectra[nans] = np.interp(wl[nans], wl[~nans], spectra[~nans])
        interp_spectra = spectra
        S_atm_values[:] = interp_spectra


    return rho_R_values, rho_A_R_values, Tg_H20_values, Tg_O3_values, Ts_Tv_values, S_atm_values







def run_6sv1_atmospheric_correction(satobj, dem_path: Path = None, VERBOSE: bool = True):

    if VERBOSE: 
        print("[INFO] Running 6SV1 atmospheric correction")

    try:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
    except Exception as ex:
        print(ex)
        print("[WARNING] 6SV1 defaulting to direct georeferencing.")
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes


    footprint, bbox, temporal = _extract_footprint_and_date(satobj=satobj)

    #print("Footprint polygon:", footprint)
    #print("Bounding rectangle:", bbox)

    if VERBOSE:
        print("Temporal range:", temporal)

    path = Path(satobj.capture_dir)
    path.joinpath("data_aerosol")
    path.mkdir(parents=True, exist_ok=True)

    files = download_viirs_aot(footprint_polygon=footprint, temporal_range=temporal, local_path=path)

    aot_inside_NOAA = None
    aot_inside_SNPP = None

    try:
        files_f = [f for f in files if 'NOAA' in str(f)]
        all_aot_NOAA, all_lat_NOAA, all_lon_NOAA, aot_inside_NOAA = get_aot_in_swath(files_f, footprint, latitudes, longitudes, name='NOAA')
        if VERBOSE:
            print(np.mean(aot_inside_NOAA))
    except Exception:
        pass

    try:
        files_f = [f for f in files if 'SNPP' in str(f)]
        all_aot_SNPP, all_lat_SNPP, all_lon_SNPP, aot_inside_SNPP = get_aot_in_swath(files_f, footprint, latitudes, longitudes, name='SNPP')
        if VERBOSE:
            print(np.mean(aot_inside_SNPP))
    except Exception:
        pass

    if aot_inside_NOAA is None and aot_inside_SNPP is not None:
        aot550 = np.mean(aot_inside_SNPP)

    elif aot_inside_SNPP is None and aot_inside_NOAA is not None:
        aot550 = np.mean(aot_inside_NOAA)
    
    elif aot_inside_SNPP is not None and aot_inside_NOAA is not None:
        aot550 = np.mean(np.concatenate(aot_inside_NOAA, aot_inside_SNPP))
    else:

        print("[WARNING] No AOT at 550nm value found.")
        aot550 = None

    if VERBOSE:
        print("AOT550 Value:")
        print(aot550)



    py6s_dict = {
        #'aot550': 0.0580000256
        'aot550': aot550
    }

    wavelengths = satobj.wavelengths
    l1d_cube = satobj.l1d_cube.to_numpy()

    solar_azimuth_angles = satobj.solar_azimuth_angles
    solar_zenith_angles = satobj.solar_zenith_angles

    sat_azimuth_angles = satobj.sat_azimuth_angles
    sat_zenith_angles = satobj.sat_zenith_angles

    iso_time = satobj.iso_time


    rho_R_values, \
    rho_A_R_values, \
    Tg_H20_values, \
    Tg_O3_values, \
    Ts_Tv_values, \
    S_atm_values = run_py6s(wavelengths=wavelengths, 
                            reflectance_cube=l1d_cube,
                            lat_2d_array=latitudes,
                            lon_2d_array=longitudes,
                            solar_azimuth_angles=solar_azimuth_angles,
                            solar_zenith_angles=solar_zenith_angles,
                            sat_azimuth_angles=sat_azimuth_angles,
                            sat_zenith_angles=sat_zenith_angles,
                            iso_time=iso_time,
                            py6s_dict=py6s_dict,
                            dem_path=dem_path
                            )
    

    cube = np.empty_like(l1d_cube)

    height, width, bands = cube.shape

    #for BandId in tqdm(range(120))
    for band in tqdm(range(0,bands)):
        for i in range(0,height):
            for j in range(0,width):

                #rho_toa = l1d_cube[i,j,band]

                #rho_R = rho_R_values[i,j,band]
                #rho_A_R = rho_A_R_values[i,j,band]
                #Tg_H20 = Tg_H20_values[i,j,band]
                #Tg_O3 = Tg_O3_values[i,j,band]
                #Ts_Tv = Ts_Tv_values[i,j,band]
                #S_atm = S_atm_values[i,j,band]

                rho_toa = l1d_cube[band]

                rho_R = rho_R_values[band]
                rho_A_R = rho_A_R_values[band]
                Tg_H20 = Tg_H20_values[band]
                Tg_O3 = Tg_O3_values[band]
                Ts_Tv = Ts_Tv_values[band]
                S_atm = S_atm_values[band]

                rho_atm = rho_R + (rho_A_R - rho_R)*Tg_H20

                Y = rho_toa - rho_atm * Tg_O3

                numerator = Y
                denominator = (S_atm * Y) + (Ts_Tv * Tg_O3 * Tg_H20)

                rho_s = numerator / denominator

                cube[i,j,band] = rho_s



    

    return cube





def toa_to_surface_reflectance(rho_toa, rho_ra, Tg, Ts, S):
    
    # Vermote (1997), Equation 1) inverted

    A = 1 / (Tg * Ts)
    B = -rho_ra / Ts
    Y = A*rho_toa + B 
    rho_s = Y / (1 + S*Y) 

    return rho_s
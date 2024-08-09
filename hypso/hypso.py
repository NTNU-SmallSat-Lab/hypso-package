from pathlib import Path
from typing import Union

class Hypso:

    def __init__(self, hypso_path: Union[str, Path], points_path: Union[str, Path, None] = None):

        """
        Initialization of HYPSO Class.

        :param hypso_path: Absolute path to "L1a.nc" file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """
        # Set NetCDF and .points file paths
        self._set_hypso_path(hypso_path=hypso_path)
        self._set_points_path(points_path=points_path)

        # Initialize platform and sensor names
        self.platform = None
        self.sensor = None

        # Initialize capture name and target
        self.capture_name = None
        self.capture_region = None

        # Initialize directory and file info
        self.tmp_dir = None
        self.nc_dir = None
        self.nc_file = None
        self.nc_name = None
        self.l1a_nc_file = None
        self.l1b_nc_file = None
        self.l2a_nc_file = None
        self.l1a_nc_name = None
        self.l1b_nc_name = None
        self.l2a_nc_name = None

        # Initialize datacubes
        self.l1a_cube = None
        self.l1b_cube = None
        self.l2a_cubes = {}

        # Initialize top of atmpshere (TOA) reflectance
        self.toa_reflectance = None

        # Initialize metadata dictionaries
        self.capture_config = {}
        self.timing = {}
        self.adcs = {}

        # Initialize timing info
        self.capture_datetime = None
        self.start_timestamp_capture = None
        self.end_timestamp_capture = None
        self.start_timestamp_adcs = None
        self.end_timestamp_adcs = None
        self.unixtime = None
        self.iso_time = None


        # Initialize dimensions
        self.capture_type = None
        self.spatial_dimensions = (956, 684)  # 1092 x variable
        self.standard_dimensions = {
            "nominal": 956,  # Along frame_count
            "wide": 1092  # Along image_height (row_count)
        }

        self.x_start = None
        self.x_stop = None
        self.y_start = None
        self.y_stop = None
        self.bin_factor = 8
        self.row_count = None
        self.frame_count = None
        self.column_count = None
        self.image_height = None
        self.image_width = None
        self.im_size = None
        self.bands = None
        self.lines = None
        self.samples = None

        # Misc metadata
        self.background_value = None
        self.exposure = None

        # Initialize georeferencing info
        self.projection_metadata = None
        self.latitudes = None
        self.longitudes = None
        self.datacube_flipped = False
        self.along_track_gsd = None
        self.across_track_gsd = None
        self.along_track_mean_gsd = None
        self.across_track_mean_gsd = None
        self.resolution = None
        self.bbox = None

       # Initialize calibration file paths:
        self.rad_coeff_file = None
        self.smile_coeff_file = None
        self.destriping_coeff_file = None
        self.spectral_coeff_file = None

        # Initialize calibration coefficients
        self.rad_coeffs = None
        self.smile_coeffs = None
        self.destriping_coeffs = None
        self.spectral_coeffs = None

        # Initialize wavelengths
        self.wavelengths = None
        self.wavelengths_units = r'$nm$'

        # Initialize spectral response function
        self.srf = None

        # Initilize land mask dict
        self.land_masks = {}
        self.active_land_mask = None
        self.active_land_mask_name = None
        
        # Initilize cloud mask dict
        self.cloud_masks = {}
        self.active_cloud_mask = None
        self.active_cloud_mask_name = None

        # Intialize active mask
        self.active_mask = None

        # Initialize chlorophyll estimates dict
        self.chl = {}

        # Initialize products dict

        # Initialize ADCS data
        self.adcs = None
        self.adcs_pos_df = None
        self.adcs_quat_df = None

        # Initialize geometry data
        self.framepose_df = None
        self.wkt_linestring_footprint = None
        self.prj_file_contents = None
        self.local_angles = None
        self.geometric_meta_info = None
        self.solar_zenith_angles = None
        self.solar_azimuth_angles = None
        self.sat_zenith_angles = None
        self.sat_azimuth_angles = None
        self.latitudes_original = None
        self.longitudes_original = None

        # DEBUG
        self.DEBUG = False
        self.VERBOSE = False

    def _set_hypso_path(self, hypso_path=None) -> None:

        if hypso_path is None:
            hypso_path = self.hypso_path

        # Make NetCDF file path an absolute path
        self.hypso_path = Path(hypso_path).absolute()

        return None

    def _set_points_path(self, points_path=None) -> None:

        # Make .points file path an absolute path (if possible)
        if points_path is not None:
            self.points_path = Path(points_path).absolute()
        else:
            self.points_path = None

        return None
        

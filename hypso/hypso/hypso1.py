
from importlib.resources import files
from pathlib import Path
from typing import Union

from hypso import Hypso
from hypso.DataArrayDict import DataArrayDict

class Hypso1(Hypso):

    def __init__(self, path: Union[str, Path], verbose=False) -> None:
        
        """
        Initialization of HYPSO-1 Class.

        :param path: Absolute path to NetCDF file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(path=path)

        # General -----------------------------------------------------
        self.platform = 'hypso1'
        self.sensor = 'hypso1_hsi'
        self.VERBOSE = verbose

        self._load_capture_file(path=path)

        product_attributes = {}

        products = DataArrayDict(dims_shape=self.spatial_dimensions, 
                                      attributes=product_attributes, 
                                      dim_names=self.dim_names_2d,
                                      num_dims=2
                                      )

        setattr(self, "products", products)

        return None



    def _set_calibration_coeff_files(self) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        match self.capture_type:

            case "custom":
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.npz"
                npz_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.npz"  
                npz_file_destriping = None
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case "nominal":
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case "wide":
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.npz"
                npz_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.npz"
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case _:
                npz_file_radiometric = None
                npz_file_smile = None
                npz_file_destriping = None
                npz_file_spectral = None

        self.rad_coeff_file = files('hypso.calibration').joinpath(f'hypso1_data/{npz_file_radiometric}')
        self.smile_coeff_file = files('hypso.calibration').joinpath(f'hypso1_data/{npz_file_smile}')
        self.destriping_coeff_file = files('hypso.calibration').joinpath(f'hypso1_data/{npz_file_destriping}')
        self.spectral_coeff_file = files('hypso.calibration').joinpath(f'hypso1_data/{npz_file_spectral}')














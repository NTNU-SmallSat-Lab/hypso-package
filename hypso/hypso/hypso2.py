
from importlib.resources import files
from pathlib import Path
from typing import Union

from hypso2_calibration import get_hypso2_calibration_files

from hypso import Hypso
from hypso.DataArrayDict import DataArrayDict

class Hypso2(Hypso):

    def __init__(self, path: Union[str, Path], verbose=False) -> None:
        
        """
        Initialization of HYPSO-2 Class.

        :param path: Absolute path to NetCDF file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(path=path)

        # General -----------------------------------------------------
        self.platform = 'hypso2'
        self.sensor = 'hypso2_hsi'
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

        capture_type = self.capture_type

        calibration_files = get_hypso2_calibration_files(capture_type)

        self.rad_coeff_file = calibration_files['radiometric']
        self.smile_coeff_file = calibration_files['smile']
        self.destriping_coeff_file = calibration_files['destriping']
        self.spectral_coeff_file = calibration_files['spectral']

        return None


















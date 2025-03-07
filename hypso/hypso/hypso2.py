from pathlib import Path
from typing import Union
import numpy as np

from .HypsoBase import HypsoBase
from hypso2_calibration import get_hypso2_calibration_files


class Hypso2(HypsoBase):

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

        self.fwhm = np.array([5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46,
                              5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46,
                              5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29,
                              3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29,
                              3.29, 3.29, 3.29, 3.29, 3.29, 3.32, 3.32, 3.32, 3.32, 3.32, 3.32,
                              3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.54, 3.54, 3.54, 3.54,
                              3.58, 3.58, 3.58, 3.59, 3.59, 3.59, 3.59, 3.59, 3.59, 3.59])

        self._load_capture_file(path=path)

        return None



    def _set_calibration_coeff_files(self) -> None:

        capture_type = self.capture_type

        calibration_files = get_hypso2_calibration_files(capture_type)

        self.rad_coeff_file = calibration_files['radiometric']
        self.smile_coeff_file = calibration_files['smile']
        self.destriping_coeff_file = calibration_files['destriping']
        self.spectral_coeff_file = calibration_files['spectral']

        return None


















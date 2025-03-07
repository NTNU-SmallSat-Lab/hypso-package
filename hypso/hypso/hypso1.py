from pathlib import Path
from typing import Union
import numpy as np

from .HypsoBase import HypsoBase
from hypso1_calibration import get_hypso1_calibration_files


class Hypso1(HypsoBase):

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

        self.fwhm = np.array([9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 
                              9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 6.6, 6.6, 6.6, 6.6, 6.6, 
                              6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 
                              8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 
                              5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 
                              5.8, 5.8, 5.8, 5.8, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 
                              4.1, 4.1, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 
                              4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

        self._load_capture_file(path=path)

        return None



    def _set_calibration_coeff_files(self) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        capture_type = self.capture_type

        calibration_files = get_hypso1_calibration_files(capture_type)

        self.rad_coeff_file = calibration_files['radiometric']
        self.smile_coeff_file = calibration_files['smile']
        self.destriping_coeff_file = calibration_files['destriping']
        self.spectral_coeff_file = calibration_files['spectral']

        return None















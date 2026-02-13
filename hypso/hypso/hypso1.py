from pathlib import Path
from typing import Union
import numpy as np

# Development use only:
# import sys
# sys.path.insert(0, '/home/ariaa/smallSatLab/hypso-package-new/hypso1_calibration/')

from .HypsoBase import HypsoBase
from hypso1_calibration import get_hypso1_calibration_files
from hypso.calibration import read_coeffs_from_file

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

        self.srf_wl = np.array([455 ,505 ,555 ,605 ,655 ,705, 755 ])
        self.srf_fwhm = np.array([9.6, 6.6, 8.2, 5.8, 5.8, 4.1, 4.0])

        self._load_capture_file(path=path)

        return None



    def _set_calibration_coeff_files(self, coeff_type='moved', **kwargs) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.
        :return: None.
        """       

        capture_type = self.capture_type

        #self.coeff_type = kwargs.get('coeff_type', 'moved')  # Default to 'original' if not provided
        print(f"[INFO] Setting calibration coefficient files with coeff_type: {coeff_type}")
        calibration_files = get_hypso1_calibration_files(capture_type, coeff_type=coeff_type)  # 'moved', 'adjusted', or 'original.'

        self.coeff_type = coeff_type
        self.rad_coeff_file = calibration_files['radiometric']
        self.smile_coeff_file = calibration_files['smile']
        self.destriping_coeff_file = calibration_files['destriping']
        self.spectral_coeff_file = calibration_files['spectral']

        return None


def get_hypso1_wavelengths(aoi_x=428, column_count=1080, bin_factor=9):

    calibration_files = get_hypso1_calibration_files()
    
    spectral_coeff_file = calibration_files["spectral"]
    
    x_start = aoi_x
    x_stop = aoi_x + column_count

    spectral_coeffs = read_coeffs_from_file(coeff_path=spectral_coeff_file, coeff_type='spectral', x_start=x_start, x_stop=x_stop, bin_factor=bin_factor)
    
    return spectral_coeffs












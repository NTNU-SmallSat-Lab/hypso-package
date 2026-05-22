from pathlib import Path
from typing import Union
import numpy as np

# Development use only:
# import sys
# sys.path.insert(0, '/home/ariaa/smallSatLab/hypso-package-new/hypso2_calibration/')

from .HypsoBase import HypsoBase
from hypso2_calibration import get_hypso2_calibration_files
from hypso.calibration import read_coeffs_from_file

class Hypso2(HypsoBase):

    def __init__(self, path: Union[str, Path], label: str = None, load_cube: bool = True, verbose=False) -> None:
        
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
        self.sat_id = 'HYPSO-2'
        self.VERBOSE = verbose
        self.label = label

        print("[INFO] Detected plaform: " + self.platform)
        print("[INFO] Detected sensor: " + self.sensor)        

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

        self.srf_wl =   [435.84 ,546.07 ,696.54 ,706.72 ,738.4 ,751.46 ,763.51 ,772.38 ,811.53 ,826.45 ,842.46 ,871.68 ,912 ]
        self.srf_fwhm = [5.46   ,3.34   ,3.29   ,3.32   ,3.42  ,3.54   ,3.58   ,3.59   ,4.16   ,4.06   ,4.66   ,4.47   ,5.06]

        self._load_capture_file(path=path, load_cube=load_cube)

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
        calibration_files = get_hypso2_calibration_files(capture_type, coeff_type=coeff_type) # 'moved', 'adjusted', or 'original.'

        #calibration_files = get_hypso2_calibration_files(capture_type, **kwargs)

        self.coeff_type = coeff_type
        self.rad_coeff_file = calibration_files['radiometric']
        self.smile_coeff_file = calibration_files['smile']
        self.destriping_coeff_file = calibration_files['destriping']
        self.spectral_coeff_file = calibration_files['spectral']

        return None


def get_hypso2_wavelengths(aoi_x=428, column_count=1080, bin_factor=9):

    calibration_files = get_hypso2_calibration_files()
    
    spectral_coeff_file = calibration_files["spectral"]
    
    x_start = aoi_x
    x_stop = aoi_x + column_count

    spectral_coeffs = read_coeffs_from_file(coeff_path=spectral_coeff_file, coeff_type='spectral', x_start=x_start, x_stop=x_stop, bin_factor=bin_factor)

    return spectral_coeffs















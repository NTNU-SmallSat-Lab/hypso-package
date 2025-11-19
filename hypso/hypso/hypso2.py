from pathlib import Path
from typing import Union
import numpy as np
import copy

# Development use only:
import sys
sys.path.insert(0, '/home/ariaa/smallSatLab/hypso-package-new/hypso2_calibration/')
# sys.path.append('../../hypso-package-new/hypso2_calibration/')

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

        self.srf_wl =   [435.84 ,546.07 ,696.54 ,706.72 ,738.4 ,751.46 ,763.51 ,772.38 ,811.53 ,826.45 ,842.46 ,871.68 ,912 ]
        self.srf_fwhm = [5.46   ,3.34   ,3.29   ,3.32   ,3.42  ,3.54   ,3.58   ,3.59   ,4.16   ,4.06   ,4.66   ,4.47   ,5.06]


        self._load_capture_file(path=path)

        return None



    def _set_calibration_coeff_files(self) -> None:

        calibration_files = get_hypso2_calibration_files(coeff_type='adjusted')

        self.rad_coeff_file = calibration_files['radiometric']
        self.smile_coeff_file = calibration_files['smile']
        self.destriping_coeff_file = calibration_files['destriping']
        self.spectral_coeff_file = calibration_files['spectral']

        return None


















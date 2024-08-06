from pathlib import Path
from typing import Union

from .hypso import Hypso

class Hypso2(Hypso):

    def __init__(self, hypso_path: Union[str, Path], points_path: Union[str, Path, None] = None) -> None:
        
        """
        Initialization of (planned) HYPSO-2 Class.

        :param hypso_path: Absolute path to "L1a.nc" file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(hypso_path=hypso_path, points_path=points_path)

        self.platform = 'hypso2'
        self.sensor = 'hypso2_hsi'




from typing import Union
import numpy as np
import pandas as pd
from importlib.resources import files
from pathlib import Path
from dateutil import parser
import netCDF4 as nc
import matplotlib.pyplot as plt
import xarray as xr
import re


from .hypso import Hypso
from .hypso1 import Hypso1


class Hypso1_Scene(Hypso1):

    def __init__(self, satobj: Hypso1) -> None:
        
        #super().__init__(hypso_path=hypso_path, points_path=points_path)


        start_time = satobj.capture_datetime
        end_time = satobj.capture_datetime

        return None
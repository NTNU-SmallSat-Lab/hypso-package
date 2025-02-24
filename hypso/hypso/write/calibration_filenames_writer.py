from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np

def calibration_filenames_writer(satobj, netfile: nc.Dataset) -> None:



    radiometric_file = None
    smile_file = None
    spectral_file = None
    destriping_file = None


    # Radiometric calibration file
    try:
        radiometric_file = str(Path(satobj.rad_coeff_file).name)
    except (AttributeError, TypeError):
        pass

    try:
        radiometric_file = satobj.nc_attrs['radiometric_file']
    except KeyError:
        pass

    set_or_create_attr(netfile,
                        attr_name="radiometric_file",
                        attr_value=str(radiometric_file))


    # Smile correction file
    try:
        smile_file = str(Path(satobj.smile_coeff_file).name)
    except (AttributeError, TypeError):
        pass

    try:
        smile_file = satobj.nc_attrs['smile_file']
    except KeyError:
        pass

    set_or_create_attr(netfile,
                        attr_name="smile_file",
                        attr_value=str(smile_file))


    # Destriping correction file
    try:
        destriping_file = str(Path(satobj.destriping_coeff_file).name)
    except (AttributeError, TypeError):
        pass

    try:
        destriping_file = satobj.nc_attrs['destriping_file']
    except KeyError:
        pass

    set_or_create_attr(netfile,
                        attr_name="destriping_file",
                        attr_value=str(destriping_file))
    

    # Spectral calibration file
    try:
        spectral_file = str(Path(satobj.spectral_coeff_file).name)
    except (AttributeError, TypeError):
        pass

    try:
        spectral_file = satobj.nc_attrs['spectral_file']
    except KeyError:
        pass

    set_or_create_attr(netfile, 
                       attr_name="spectral_file", 
                       attr_value=str(spectral_file))


    return None
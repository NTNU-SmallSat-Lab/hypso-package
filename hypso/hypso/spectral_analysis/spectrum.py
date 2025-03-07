
from ..geometry.nearest import get_nearest_pixel

from ..HypsoBase import HypsoBase
from ..hypso1 import Hypso1
from ..hypso2 import Hypso2

from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path


def get_l1a_spectrum(satobj: Union[Hypso1, Hypso2], 
                    latitude=None, 
                    longitude=None,
                    x: int = None,
                    y: int = None
                    ) -> xr.DataArray:

    if latitude is not None and longitude is not None:
        idx = get_nearest_pixel(target_latitude=latitude, 
                                target_longitude=longitude,
                                latitudes=satobj.latitudes,
                                longitudes=satobj.longitudes)

    elif x is not None and y is not None:
        idx = (x,y)

    else:
        return None

    spectrum = satobj.l1a_cube[idx[0], idx[1], :]

    return spectrum


def get_l1b_spectrum(satobj: Union[Hypso1, Hypso2], 
                    latitude=None, 
                    longitude=None,
                    x: int = None,
                    y: int = None
                    ) -> tuple[np.ndarray, str]:

    if latitude is not None and longitude is not None:
        idx = get_nearest_pixel(target_latitude=latitude, 
                                target_longitude=longitude,
                                latitudes=satobj.latitudes,
                                longitudes=satobj.longitudes)

    elif x is not None and y is not None:
        idx = (x,y)
    else:
        return None

    spectrum = satobj.l1b_cube[idx[0], idx[1], :]

    return spectrum

def get_l2a_spectrum(satobj: Union[Hypso1, Hypso2],
                    latitude=None, 
                    longitude=None,
                    x: int = None,
                    y: int = None
                    ) -> xr.DataArray:

    if latitude is not None and longitude is not None:
        idx = get_nearest_pixel(target_latitude=latitude, 
                                target_longitude=longitude,
                                latitudes=satobj.latitudes,
                                longitudes=satobj.longitudes)

    elif x is not None and y is not None:
        idx = (x,y)
    else:
        return None

    try:
        spectrum = satobj.l2a_cube[idx[0], idx[1], :]
    except KeyError:
        return None

    return spectrum



def plot_l1a_spectrum(satobj: Union[Hypso1, Hypso2], 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None,
                        save: bool = False
                        ) -> None:
    
    if latitude is not None and longitude is not None:
        idx = get_nearest_pixel(target_latitude=latitude, 
                                target_longitude=longitude,
                                latitudes=satobj.latitudes,
                                longitudes=satobj.longitudes)

    elif x is not None and y is not None:
        idx = (x,y)

    else:
        return None

    spectrum = satobj.l1a_cube[idx[0], idx[1], :]
    bands = range(0, len(spectrum))
    units = spectrum.attrs["units"]

    output_file = Path(satobj.parent_dir, satobj.capture_name + '_l1a_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(bands, spectrum)
    plt.ylabel(units)
    plt.xlabel("Band number")
    plt.title(f"L1a (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
    plt.grid(True)

    if save:
        plt.imsave(output_file)
    else:
        plt.show()

    return None


def plot_l1b_spectrum(satobj: Union[Hypso1, Hypso2], 
                    latitude=None, 
                    longitude=None,
                    x: int = None,
                    y: int = None,
                    save: bool = False
                    ) -> None:
    
    if latitude is not None and longitude is not None:
        idx = get_nearest_pixel(target_latitude=latitude, 
                                target_longitude=longitude,
                                latitudes=satobj.latitudes,
                                longitudes=satobj.longitudes)

    elif x is not None and y is not None:
        idx = (x,y)

    else:
        return None

    spectrum = satobj.l1b_cube[idx[0], idx[1], :]
    bands = satobj.wavelengths
    units = spectrum.attrs["units"]

    output_file = Path(satobj.parent_dir, satobj.capture_name + '_l1a_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(bands, spectrum)
    plt.ylabel(units)
    plt.xlabel("Wavelength (nm)")
    plt.title(f"L1b (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
    plt.grid(True)

    if save:
        # TODO: TypeError: imsave() missing 1 required positional argument: 'arr'
        plt.imsave(output_file)
    else:
        plt.show()

    return None


def plot_l2a_spectrum(satobj: Union[Hypso1, Hypso2],
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None,
                        save: bool = False
                        ) -> np.ndarray:
    
    if latitude is not None and longitude is not None:
        idx = get_nearest_pixel(target_latitude=latitude, 
                                target_longitude=longitude,
                                latitudes=satobj.latitudes,
                                longitudes=satobj.longitudes)

    elif x is not None and y is not None:
        idx = (x,y)

    else:
        return None

    try:
        spectrum = satobj.l2a_cube[idx[0], idx[1], :]
    except KeyError:
        return None

    bands = satobj.wavelengths
    units = spectrum.attrs["units"]

    output_file = Path(satobj.parent_dir, satobj.capture_name + '_l2a_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(bands, spectrum)
    plt.ylabel(units)
    plt.ylim([0, 1])
    plt.xlabel("Wavelength (nm)")
    plt.title(f"L2a (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
    plt.grid(True)

    if save:
        plt.imsave(output_file)
    else:
        plt.show()
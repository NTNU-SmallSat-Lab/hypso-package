from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np
from .navigation_group_writer import navigation_group_writer
from .calibration_filenames_writer import calibration_filenames_writer

from matplotlib import pyplot as plt

def write_products_nc_file(satobj, file_name: Path, overwrite: bool = False, **kwargs) -> None:
    
    path = Path(file_name)

    if Path(file_name).is_file() and not overwrite:

        if satobj.VERBOSE:
            print("[INFO] NetCDF file has already exists.")

        return None

    products_nc_writer(satobj=satobj, 
                    dst_nc=path, 
                    **kwargs)

    return None


def products_nc_writer(satobj, dst_nc: str, datacube: str = True) -> None:
    """
    Create a l1d.nc file using the top-of-atmosphere data.

    :return: Nothing.
    """

    # Create a new NetCDF file
    with (nc.Dataset(dst_nc, 'w', format='NETCDF4') as netfile):
        bands = satobj.image_width
        lines = satobj.nc_capture_config_attrs["frame_count"]  # AKA Frames AKA Rows
        samples = satobj.image_height  # AKA Cols

        # Set top level attributes -------------------------------------------------
        for md in satobj.nc_attrs:
            set_or_create_attr(netfile,
                                md,
                                satobj.nc_attrs[md])

        #set_or_create_attr(netfile, attr_name="processing_level", attr_value="L1D")

        # Add calibration file names
        calibration_filenames_writer(satobj=satobj, netfile=netfile)

        # Create dimensions
        netfile.createDimension('lines', lines)
        netfile.createDimension('samples', samples)

        # Create groups
        netfile.createGroup('products')


        # Set pseudoglobal vars like compression level
        COMP_SCHEME = 'zlib'  # Default: zlib
        COMP_LEVEL = 4  # Default (when scheme != none): 4
        COMP_SHUFFLE = True  # Default (when scheme != none): True


        for name, product in satobj.products.items():

            print('Writing ' + str(name))

            #coords = list(product.coords)

            variable = netfile.createVariable(
                'products/' + str(name),  
                'f4',
                #(coords[0], coords[1]),
                ('lines', 'samples'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            
            #variable.units = ""
            #variable.long_name = ""

            for attr_name, attr_value in product.attrs.items():
                setattr(variable, attr_name, attr_value)

            variable[:] = product.to_numpy()

        navigation_group_writer(satobj=satobj, netfile=netfile)

    return None
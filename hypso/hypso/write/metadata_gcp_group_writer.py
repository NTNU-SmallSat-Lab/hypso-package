from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np

def metadata_gcp_group_writer(satobj, netfile: nc.Dataset, COMP_SCHEME = 'zlib', COMP_LEVEL = 4, COMP_SHUFFLE = True) -> None:
    """
    Write GCP metadata group to NetCDF file. 

    :return: Nothing.
    """

    # Create GCP metadata Group --------------------------------------
    meta_gcp = netfile.createGroup('metadata/gcp')

    # Adding GCPs -------------------------------------------
    if (hasattr(satobj, 'nc_gcp_attrs') and satobj.nc_gcp_attrs is not None):

        try:
            for md in getattr(satobj, 'nc_gcp_attrs'):
                set_or_create_attr(meta_gcp,
                                    md,
                                    getattr(satobj, 'nc_gcp_attrs')[md])
        except Exception as ex:
            pass

            
    if (hasattr(satobj, 'nc_gcp_vars') and satobj.nc_gcp_vars is not None):

        keys = list(satobj.nc_gcp_vars.keys())

        if len(keys) != 0:

            length = len(satobj.nc_gcp_vars[keys[0]])

            netfile.createDimension('gcps', length)

            for key in keys:

                var = satobj.nc_gcp_vars[key]

                meta_gcp_latitude = netfile.createVariable(
                    'metadata/gcp/' + str(key), 'f4',
                    ('gcps',),
                    compression=COMP_SCHEME,
                    complevel=COMP_LEVEL,
                    shuffle=COMP_SHUFFLE)
                
                meta_gcp_latitude[:] = var

    return None
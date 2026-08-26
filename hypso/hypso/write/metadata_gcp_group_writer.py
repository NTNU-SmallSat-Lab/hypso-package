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
    if satobj.metadata.gcp.attrs is not None:

        try:
            for md in satobj.metadata.gcp.attrs:
                set_or_create_attr(meta_gcp,
                                    md,
                                    satobj.metadata.gcp.attrs[md])
        except Exception as ex:
            pass


    if satobj.metadata.gcp.vars is not None:

        keys = list(satobj.metadata.gcp.vars.keys())

        if len(keys) != 0:

            length = len(satobj.metadata.gcp.vars[keys[0]])

            netfile.createDimension('gcps', length)

            for key in keys:

                var = satobj.metadata.gcp.vars[key]

                meta_gcp_latitude = netfile.createVariable(
                    'metadata/gcp/' + str(key), 'f4',
                    ('gcps',),
                    compression=COMP_SCHEME,
                    complevel=COMP_LEVEL,
                    shuffle=COMP_SHUFFLE)
                
                meta_gcp_latitude[:] = var

    return None
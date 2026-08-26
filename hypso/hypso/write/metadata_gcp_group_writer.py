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

                # Prefixed "gcp_" - confirmed real bug found opening files in ESA SNAP:
                # a bare metadata/gcp/latitude + metadata/gcp/longitude pair (29 tie
                # points) is discovered by SNAP's CF geocoding scanner purely by
                # variable NAME (neither carries a standard_name/units of its own to
                # distinguish it), alongside the real per-pixel root latitude/longitude
                # (598x1092) - SNAP then computes a resolution estimate by indexing the
                # 29-element GCP array with a full-raster pixel offset, crashing with
                # ArrayIndexOutOfBoundsException. Confirmed via a minimal reproducer:
                # the file opens fine without this group, and crashes identically once
                # it's added back - see REFACTOR_PROGRESS.md for the full diagnosis.
                # load_gcp_from_nc_file (load/utils.py) reads whatever keys are present
                # generically, so this rename needs no reader-side change.
                meta_gcp_latitude = netfile.createVariable(
                    'metadata/gcp/gcp_' + str(key), 'f4',
                    ('gcps',),
                    compression=COMP_SCHEME,
                    complevel=COMP_LEVEL,
                    shuffle=COMP_SHUFFLE)
                
                meta_gcp_latitude[:] = var

    return None
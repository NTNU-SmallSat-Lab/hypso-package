import logging

from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)

def metadata_srf_group_writer(satobj, netfile: nc.Dataset, COMP_SCHEME = 'zlib', COMP_LEVEL = 4, COMP_SHUFFLE = True) -> None:
    """
    Write SRF group to NetCDF file. 

    :return: Nothing.
    """

    # Create geometry Group --------------------------------------
    #srf_group = netfile.createGroup('srf')
    srf_group = netfile.createGroup('metadata/srf')


    #self.srf = srf
    #self.srf_ssi = srf_ssi
    #self.srf_ssi_wl = srf_ssi_wl
    #self.esun = esun
    #self.esun_wl = esun_wl
    #self.effective_fwhm = effective_fwhm

    if (hasattr(satobj, 'esun') and satobj.esun is not None):

        try:
            esun = satobj.esun
            length = len(satobj.esun)
            netfile.createDimension('esun', length)
            esun_var = netfile.createVariable(
                'metadata/srf/esun', 'f8',
                ('esun',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            esun_var[:] = esun
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")
        

    if (hasattr(satobj, 'esun_wl') and satobj.esun_wl is not None):

        try:
            esun_wl = satobj.esun_wl
            length = len(satobj.esun_wl)
            netfile.createDimension('esun_wavelengths', length)
            esun_wl_var = netfile.createVariable(
                'metadata/srf/esun_wavelengths', 'f8',
                ('esun_wavelengths',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            esun_wl_var[:] = esun_wl
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")


    if (hasattr(satobj, 'effective_fwhm') and satobj.effective_fwhm is not None):

        try:
            effective_fwhm = satobj.effective_fwhm
            length = len(satobj.effective_fwhm)
            netfile.createDimension('effective_fwhm', length)
            effective_fwhm_var = netfile.createVariable(
                'metadata/srf/effective_fwhm', 'f8',
                ('effective_fwhm',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            effective_fwhm_var[:] = effective_fwhm
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")

    if (hasattr(satobj, 'effective_fwhm_unbinned') and satobj.effective_fwhm_unbinned is not None):

        try:
            effective_fwhm_unbinned = satobj.effective_fwhm_unbinned
            length = len(satobj.effective_fwhm_unbinned)
            netfile.createDimension('effective_fwhm_unbinned', length)
            effective_fwhm_unbinned_var = netfile.createVariable(
                'metadata/srf/effective_fwhm_unbinned', 'f8',
                ('effective_fwhm_unbinned',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            effective_fwhm_unbinned_var[:] = effective_fwhm_unbinned
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")



    return None
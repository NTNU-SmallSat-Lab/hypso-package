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



    # CSIRO

    if (hasattr(satobj, 'csiro_ssi') and satobj.csiro_ssi is not None):

        try:
            csiro_ssi = satobj.csiro_ssi
            length = len(satobj.csiro_ssi)
            netfile.createDimension('csiro_ssi', length)
            csiro_ssi_var = netfile.createVariable(
                'metadata/srf/csiro_ssi', 'f4',
                ('csiro_ssi',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            csiro_ssi_var[:] = csiro_ssi
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")


    if (hasattr(satobj, 'csiro_solar_wavelengths') and satobj.csiro_solar_wavelengths is not None):

        try:
            csiro_solar_wavelengths = satobj.csiro_solar_wavelengths
            length = len(satobj.csiro_solar_wavelengths)
            netfile.createDimension('csiro_solar_wavelengths', length)
            csiro_solar_wavelengths_var = netfile.createVariable(
                'metadata/srf/csiro_solar_wavelengths', 'f4',
                ('csiro_solar_wavelengths',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            csiro_solar_wavelengths_var[:] = csiro_solar_wavelengths
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")

    if (hasattr(satobj, 'csiro_binned_srfs') and satobj.csiro_binned_srfs is not None):

        try:
            csiro_binned_srfs = satobj.csiro_binned_srfs
            length = len(satobj.csiro_binned_srfs)
            csiro_binned_srfs_x = satobj.csiro_binned_srfs.shape[0]
            csiro_binned_srfs_y = satobj.csiro_binned_srfs.shape[1]
            netfile.createDimension('csiro_binned_srfs_x', csiro_binned_srfs_x)
            netfile.createDimension('csiro_binned_srfs_y', csiro_binned_srfs_y)
            csiro_binned_srfs_var = netfile.createVariable(
                'metadata/srf/csiro_binned_srfs', 'f4',
                ('csiro_binned_srfs_x', 'csiro_binned_srfs_y'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            csiro_binned_srfs_var[:] = csiro_binned_srfs
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")

    if (hasattr(satobj, 'csiro_effective_fwhm') and satobj.csiro_effective_fwhm is not None):

        try:
            csiro_effective_fwhm = satobj.csiro_effective_fwhm
            length = len(satobj.csiro_effective_fwhm)
            netfile.createDimension('csiro_effective_fwhm', length)
            csiro_effective_fwhm_var = netfile.createVariable(
                'metadata/srf/csiro_effective_fwhm', 'f4',
                ('csiro_effective_fwhm',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            csiro_effective_fwhm_var[:] = csiro_effective_fwhm
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")


    if (hasattr(satobj, 'csiro_esun') and satobj.csiro_esun is not None):

        try:
            csiro_esun = satobj.csiro_esun
            length = len(satobj.csiro_esun)
            netfile.createDimension('csiro_esun', length)
            csiro_esun_var = netfile.createVariable(
                'metadata/srf/csiro_esun', 'f4',
                ('csiro_esun',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            csiro_esun_var[:] = csiro_esun
        except Exception:
            logger.exception("Failed to write SRF metadata variable.")


    return None
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
            esun_var.units = "mW m-2 nm-1"
            esun_var.description = ("Band-averaged (SRF-weighted) solar irradiance F0 at 1 AU, from the "
                                    "TSIS-1 HSRS v2 (2022-11-30); the F0 used for this file's TOA reflectance")
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

    # Earth-Sun distance (AU) for the acquisition date, the same value
    # compute_reflectance used, so readers that compute their own
    # reflectance (rho = pi L d^2 / (F0 cos sza)) can apply it. Computed
    # here from iso_time rather than stored on satobj, so it is written
    # identically for in-session and file-loaded captures.
    if (getattr(satobj, 'esun', None) is not None and getattr(satobj, 'iso_time', None) is not None):
        from hypso.reflectance import earth_sun_distance
        srf_group.earth_sun_distance_au = earth_sun_distance(satobj.iso_time)

    # Per-band SRFs at full resolution, one Band_<i> variable with its own
    # wav_Band_<i> coordinate (non-zero part only), identical in content to
    # the <sensor>_srf.nc file PolymerAdapter.generate_srf_nc writes, so
    # Polymer can read its SRFs from the L1 file itself
    # (hypso.ac.ac_polymer_srf_getter_from_l1).
    if (getattr(satobj, 'srf', None) is not None and getattr(satobj, 'srf_ssi_wl', None) is not None):

        try:
            srf_wavelengths = np.asarray(satobj.srf_ssi_wl)
            for idx, wl in enumerate(satobj.wavelengths):
                bid = "Band_" + str(idx)
                srf = satobj.srf[idx, :].toarray().flatten()
                nonzero_mask = srf > 0
                if np.any(nonzero_mask):
                    srf_values, srf_wl = srf[nonzero_mask], srf_wavelengths[nonzero_mask]
                else:
                    srf_values, srf_wl = srf, srf_wavelengths

                dim = f"wav_{bid}"
                srf_group.createDimension(dim, len(srf_wl))
                wav_var = srf_group.createVariable(dim, srf_wl.dtype, (dim,),
                                                   compression=COMP_SCHEME, complevel=COMP_LEVEL,
                                                   shuffle=COMP_SHUFFLE)
                wav_var[:] = srf_wl
                wav_var.units = "nm"
                band_var = srf_group.createVariable(bid, srf_values.dtype, (dim,),
                                                    compression=COMP_SCHEME, complevel=COMP_LEVEL,
                                                    shuffle=COMP_SHUFFLE)
                band_var[:] = srf_values
                band_var.band_info = bid
                band_var.band_wavelength = float(wl)
                band_var.index = idx
                band_var.effective_fwhm = float(satobj.effective_fwhm[idx])
                band_var.center_fwhm = float(satobj.fwhm[idx])
        except Exception:
            logger.exception("Failed to write per-band SRF metadata variables.")

    return None
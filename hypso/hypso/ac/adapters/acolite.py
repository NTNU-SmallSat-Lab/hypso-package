"""ACOLITE adapter. run_correction runs ACOLITE in an ISOLATED SUBPROCESS (see
_acolite_driver.py's module docstring for the rationale - crash containment
and parallelism, plus consistency with Polymer's isolation, though ACOLITE has
no demonstrated version-conflict bug the way Polymer's v1/v2 split does) -
path/settings resolution stays here in the parent; everything past "import
ACOLITE itself" happens in _acolite_driver.py. Every other method is
HypsoCapture's corresponding ac_acolite_* method body relocated verbatim (see
base.py's ACAdapter docstring)."""
import logging
import sys
from pathlib import Path

import numpy as np

from hypso.load import load_acolite_l2r_nc, load_acolite_l2w_nc

from .base import ACAdapter, get_inferred_wavelength_band_map, run_subprocess_driver

logger = logging.getLogger(__name__)


class ACOLITEAdapter(ACAdapter):

    key = "acolite"

    def run_correction(self, satobj, settings_file: Path = None,
                       input_product_level: str = 'l1c',
                       EARTHDATA_u: str = None,
                       EARTHDATA_p: str = None,
                       python_path: str = None
                       ):
        """
        Runs ACOLITE in an isolated subprocess - see _acolite_driver.py's
        module docstring for why.

        EARTHDATA_u/EARTHDATA_p are passed to the subprocess via environment
        variables (not written into the JSON config file passed to the
        driver) to avoid putting credentials on disk even briefly.

        python_path: interpreter to run ACOLITE's subprocess under. Defaults
            to sys.executable (this same process's interpreter).

        Raises hypso.ac.adapters.base.ACRunError on failure (carries the
        subprocess's stdout/stderr and, if available, ACOLITE's own
        exception type/message/traceback).
        """
        acolite_path = Path(satobj.acolite_dir).absolute()

        logger.info("Running ACOLITE atmospheric correction installed in %s", acolite_path)

        # load() only applies a sensor's own config/defaults/<name>.txt (e.g.
        # HYPSO2.txt's dsf_wave_range=450,750) when explicitly given that
        # name - it does not auto-detect sensor from the input file. Passing
        # None would silently fall back to ACOLITE's fully generic
        # defaults.txt (dsf_wave_range=400,2500) instead, mirroring the fix
        # ac_runners.py's PACE runner already has via its explicit
        # load("PACE_OCI") call.
        settings_arg = str(settings_file) if settings_file is not None else satobj.platform.upper()

        if input_product_level.upper() == 'L1D':
            logger.info("Using L1d NetCDF as ACOLITE input.")
            inputfile = str(satobj.l1d_nc_file)  # L1d reflectance
        else:
            logger.info("Using L1c NetCDF as ACOLITE input.")
            inputfile = str(satobj.l1c_nc_file)  # default L1c (radiance)

        # capture_dir/acolite/, not capture_dir directly (2026-08-05) - see
        # the matching comment on satobj.acolite_l2r_output_nc_file
        # (hypso.io.dispatch.load_capture_file) for why (keeps ACOLITE's own
        # per-run log/settings .txt files out of the capture directory root).
        acolite_output_dir = Path(satobj.capture_dir, "acolite")
        acolite_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Writing ACOLITE output to %s", acolite_output_dir)

        settings_overrides = {
            "inputfile": inputfile,
            "output": str(acolite_output_dir),
            "polygon": None,
            "rgb_rhot": True,
            "rgb_rhos": True,
            "map_l2w": False,  # produces blank .pngs
            "l2w_mask": False,
            "l2w_mask_threshold": 0.2,
            "l2w_parameters": ['Rrs_*', 'spm_nechad2010', 'spm_nechad2016',
                               'chl_re_mishra', 'chl_oc2', 'chl_oc3',
                               'chl_re_moses3b', 'chl_re_moses3b740', 'fai',
                               'fai_rhot', 'fait', 'ndci'],
        }

        config = {
            "acolite_path": str(acolite_path),
            "settings_arg": settings_arg,
            "settings_overrides": settings_overrides,
        }

        extra_env = None
        if EARTHDATA_u is not None and EARTHDATA_p is not None:
            extra_env = {
                "HYPSO_ACOLITE_EARTHDATA_USERNAME": EARTHDATA_u,
                "HYPSO_ACOLITE_EARTHDATA_PASSWORD": EARTHDATA_p,
            }

        run_subprocess_driver(
            python_path=python_path or sys.executable,
            driver_module="hypso.ac.adapters._acolite_driver",
            config=config,
            tool_name="acolite",
            extra_env=extra_env,
        )

        logger.info("ACOLITE atmospheric correction complete.")

        return None

    def open_output(self, satobj, acolite_l2r_output_nc_file: Path = None, acolite_l2w_output_nc_file: Path = None):

        """
        Open and read ACOLITE atmospheric correction L2R and L2W NetCDF output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2a_cube' dictionary.

        :param acolite_l2r_output_nc_file: Path to the ACOLITE L2R NetCDF file (optional)
        :param acolite_l2w_output_nc_file: Path to the ACOLITE L2W NetCDF file (optional)

        :return: "datasets" Dictionary containing 2D and 3D datasets read from the NetCDFs and stored as xarray DataArrays.
        """


        if acolite_l2r_output_nc_file is not None:
            acolite_l2r_output_nc_file = Path(acolite_l2r_output_nc_file).absolute()
        else:
            acolite_l2r_output_nc_file = Path(satobj.acolite_l2r_output_nc_file).absolute()

        if acolite_l2w_output_nc_file is not None:
            acolite_l2w_output_nc_file = Path(acolite_l2w_output_nc_file).absolute()
        else:
            acolite_l2w_output_nc_file = Path(satobj.acolite_l2w_output_nc_file).absolute()




        if acolite_l2r_output_nc_file.is_file():
            logger.info("Opening ACOLITE L2R NetCDF output file %s", acolite_l2r_output_nc_file)
            l2r_datasets = load_acolite_l2r_nc(acolite_l2r_output_nc_file)

            try:
                key = "rhos"
                inferred_wavelengths = l2r_datasets[key].band.to_numpy()

                # Map inferred ACOLITE wavelengths to HYPSO wavelengths
                wl_band_map = get_inferred_wavelength_band_map(satobj, inferred_wavelengths=inferred_wavelengths)

                # Create empty cube with standard HYPSO cube dims
                shape = (satobj.spatial_dimensions[0], satobj.spatial_dimensions[1], satobj.bands)
                cube = np.full(shape=shape, fill_value=np.nan)
                cube[:,:,wl_band_map] = l2r_datasets[key]

                satobj.l2a_cubes["acolite_l2r"] = cube
                satobj.l2a_cubes["acolite_l2r"].attrs['l2_variable_name'] = key

            except Exception:
                logger.exception("Unable to load ACOLITE L2R dataset.")
                l2r_datasets = None

        else:
            logger.error("ACOLITE L2R NetCDF output file %s does not exist.", acolite_l2r_output_nc_file)
            l2r_datasets = None


        if acolite_l2w_output_nc_file.is_file():
            logger.info("Opening ACOLITE L2W NetCDF output file %s", acolite_l2w_output_nc_file)
            l2w_datasets = load_acolite_l2w_nc(acolite_l2w_output_nc_file)

            try:
                key = "Rrs"
                inferred_wavelengths = l2w_datasets[key].band.to_numpy()

                # Map inferred ACOLITE wavelengths to HYPSO wavelengths
                wl_band_map = get_inferred_wavelength_band_map(satobj, inferred_wavelengths=inferred_wavelengths)

                # Create empty cube with standard HYPSO cube dims
                shape = (satobj.spatial_dimensions[0], satobj.spatial_dimensions[1], satobj.bands)
                cube = np.full(shape=shape, fill_value=np.nan)
                cube[:,:,wl_band_map] = l2w_datasets[key]

                satobj.l2a_cubes["acolite_l2w"] = cube
                satobj.l2a_cubes["acolite_l2w"].attrs['l2_variable_name'] = key

            except Exception:
                logger.exception("Unable to load ACOLITE L2W dataset.")
                l2w_datasets = None

        else:
            logger.error("ACOLITE L2W NetCDF output file %s does not exist.", acolite_l2w_output_nc_file)
            l2w_datasets = None


        return l2r_datasets, l2w_datasets

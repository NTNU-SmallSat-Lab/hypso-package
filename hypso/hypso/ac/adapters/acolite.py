"""ACOLITE adapter. Method bodies are HypsoBase's ac_acolite_* methods relocated
verbatim (see base.py's ACAdapter docstring) - ACOLITE is imported as a package
from its installation directory (satobj.acolite_dir, appended to sys.path) and
run in-process via acolite_run(), writing L2R/L2W NetCDF output plus per-run
log/settings files into <capture_dir>/acolite/."""
import sys
from pathlib import Path

import numpy as np

from hypso.load import load_acolite_l2r_nc, load_acolite_l2w_nc

from .base import ACAdapter, get_inferred_wavelength_band_map


class ACOLITEAdapter(ACAdapter):

    key = "acolite"

    def run_correction(self, satobj, settings_file: Path = None,
                       input_product_level: str = 'l1c',
                       EARTHDATA_u: str = None,
                       EARTHDATA_p: str = None
                       ):

        acolite_path = Path(satobj.acolite_dir).absolute()

        print("[INFO] Running ACOLITE atmospheric correction installed in " + str(acolite_path))

        sys.path.append(str(acolite_path))
        #print(sys.path)

        import acolite as ac
        from acolite.acolite.settings import load
        from acolite.acolite import acolite_run

        # optional file with processing settings
        # if set to None defaults will be used

        # import settings
        #settings = ac.acolite.settings.load(settings_file)
        # load() only applies a sensor's own config/defaults/<name>.txt (e.g.
        # HYPSO2.txt's dsf_wave_range=450,750) when explicitly given that
        # name - it does not auto-detect sensor from the input file. Passing
        # None (as this always did before) silently fell back to ACOLITE's
        # fully generic defaults.txt (dsf_wave_range=400,2500) instead,
        # mirroring the fix ac_runners.py's PACE runner already has via its
        # explicit load("PACE_OCI") call.
        settings = load(settings_file if settings_file is not None else satobj.platform.upper())

        if EARTHDATA_u is not None and EARTHDATA_p is not None:
            settings['EARTHDATA_u'] = EARTHDATA_u
            settings['EARTHDATA_p'] = EARTHDATA_p
            settings['ancillary_data'] = True

        # set settings provided above

        if input_product_level.upper() == 'L1D':
            print("[INFO] Using L1d NetCDF as ACOLITE input.")
            settings['inputfile'] = str(satobj.l1d_nc_file) # L1d reflectance
        else:
            print("[INFO] Using L1c NetCDF as ACOLITE input.")
            settings['inputfile'] = str(satobj.l1c_nc_file) # default L1c (radiance)


        # capture_dir/acolite/, not capture_dir directly (2026-08-05) - see
        # the matching comment on satobj.acolite_l2r_output_nc_file
        # (hypso.io.dispatch.load_capture_file) for why (keeps ACOLITE's own
        # per-run log/settings .txt files out of the capture directory root).
        acolite_output_dir = Path(satobj.capture_dir, "acolite")
        acolite_output_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Writing ACOLITE output to " + str(acolite_output_dir))
        settings['output'] = str(acolite_output_dir)

        settings['polygon'] = None
        settings['rgb_rhot'] = True
        settings['rgb_rhos'] = True
        settings['map_l2w'] = False #produces blank .pngs
        settings['l2w_mask'] = False
        settings['l2w_mask_threshold'] = 0.2

        settings['l2w_parameters'] = ['Rrs_*', \
                                    'spm_nechad2010', \
                                    'spm_nechad2016', \
                                    'chl_re_mishra',\
                                    'chl_oc2', \
                                    'chl_oc3', \
                                    'chl_re_moses3b', \
                                    'chl_re_moses3b740', \
                                    'fai', \
                                    'fai_rhot', \
                                    'fait', \
                                    'ndci']


        processed = acolite_run(settings=settings)

        #acolite_l2_file = processed[0]['l2r'][0]

        print("[INFO] ACOLITE atmospheric correction complete.")

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
            print("[INFO] Opening ACOLITE L2R NetCDF output file " + str(acolite_l2r_output_nc_file))
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

                satobj.l2a_cube["acolite_l2r"] = cube
                satobj.l2a_cube["acolite_l2r"].attrs['l2_variable_name'] = key

            except Exception as ex:
                print("[ERROR] Unable to load ACOLITE L2R dataset.")
                l2r_datasets = None

        else:
            print("[ERROR] ACOLITE L2R NetCDF output file " + str(acolite_l2r_output_nc_file) + " does not exist.")
            l2r_datasets = None


        if acolite_l2w_output_nc_file.is_file():
            print("[INFO] Opening ACOLITE L2W NetCDF output file " + str(acolite_l2w_output_nc_file))
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

                satobj.l2a_cube["acolite_l2w"] = cube
                satobj.l2a_cube["acolite_l2w"].attrs['l2_variable_name'] = key

            except Exception as ex:
                print("[ERROR] Unable to load ACOLITE L2W dataset.")
                l2w_datasets = None

        else:
            print("[ERROR] ACOLITE L2W NetCDF output file " + str(acolite_l2w_output_nc_file) + " does not exist.")
            l2w_datasets = None


        return l2r_datasets, l2w_datasets

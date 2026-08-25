"""Polymer adapter. Method bodies are HypsoBase's ac_polymer_* methods relocated
verbatim (see base.py's ACAdapter docstring) - Polymer is imported from a
caller-supplied checkout (polymer_base_path and friends, inserted onto sys.path)
and run in-process via run_polymer(), with HYPSO's spectral response passed
through a generated per-capture SRF NetCDF (generate_srf_nc) and the
"hypso.ac.ac_polymer_srf_getter" hook - that dotted name is resolved *by string*
inside Polymer, so it must keep pointing at hypso/ac/ac_polymer.py's function
regardless of how this adapter is reorganized."""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.load import load_polymer_l2_v1_nc, load_polymer_l2_v2_nc

from .base import ACAdapter, get_inferred_wavelength_band_map


class PolymerAdapter(ACAdapter):

    key = "polymer"

    def get_id_sensor(self, satobj):

        sensor_version = "_" + str(satobj.coeff_type)

        # combine sensor name ("HYPSO-1" or "HYPSO-2") with coefficients version
        # Polymer expects format like "HYPSO-2_moved"
        id_sensor = str(satobj.sat_id) + sensor_version

        return id_sensor

    def get_srf_nc_path(self, satobj):

        id_sensor = self.get_id_sensor(satobj)

        srf_nc_file = id_sensor + "_srf.nc"
        srf_nc_path = Path(satobj.parent_dir, srf_nc_file )

        satobj.srf_nc_file = srf_nc_file
        satobj.srf_nc_path = srf_nc_path

        return srf_nc_file, srf_nc_path

    def get_ssi_nc_path(self, satobj):

        id_sensor = self.get_id_sensor(satobj)

        ssi_nc_file = id_sensor + "_ssi.nc"
        ssi_nc_path = Path(satobj.parent_dir, ssi_nc_file )

        satobj.ssi_nc_file = ssi_nc_file
        satobj.ssi_nc_path = ssi_nc_path

        return ssi_nc_file, ssi_nc_path

    def get_esun_nc_path(self, satobj):

        id_sensor = self.get_id_sensor(satobj)

        esun_nc_file = id_sensor + "_esun.nc"
        esun_nc_path = Path(satobj.parent_dir, esun_nc_file )

        # Fixed copy-paste bug (was satobj.ssi_nc_file/ssi_nc_path, clobbering
        # get_ssi_nc_path's attributes): harmless in practice - nothing reads
        # these cached attributes, the return values are what callers use -
        # but the cached names now match what they hold.
        satobj.esun_nc_file = esun_nc_file
        satobj.esun_nc_path = esun_nc_path

        return esun_nc_file, esun_nc_path

    def generate_srf_nc(self, satobj):

        id_sensor = self.get_id_sensor(satobj)

        _, srf_nc_path = self.get_srf_nc_path(satobj)


        ds = xr.Dataset()
        ds.attrs["desc"] = f'Spectral response functions for {id_sensor}'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = satobj.platform

        for idx, wl in enumerate(satobj.wavelengths):

            # Construct band ID
            bid = "Band_" + str(idx)

            # Read ith SRF and convert from CSR sparse array
            srf = satobj.srf[idx,:].toarray().flatten()
            srf_wavelengths = satobj.srf_ssi_wl

            # Find where SRF is non-zero
            nonzero_mask = srf > 0

            # Extract non-zero portion of SRF and SRF wavelength array
            if np.any(nonzero_mask):
                srf_nonzero = srf[nonzero_mask]
                srf_wavelengths_nonzero = srf_wavelengths[nonzero_mask]
            else:
                srf_nonzero = srf
                srf_wavelengths_nonzero = srf_wavelengths

            # Add band entry to dataset
            ds[bid] = xr.DataArray(
                srf_nonzero,
                coords={f"wav_{bid}": srf_wavelengths_nonzero},
                attrs={
                    "band_info": bid,
                    "band_wavelength": wl,
                    "index": idx,
                    "effective_fwhm": satobj.effective_fwhm[idx],
                    "center_fwhm": satobj.fwhm[idx]
                },
            )
            ds[f"wav_{bid}"].attrs["units"] = "nm"

        # Sort dataarrays within dataset based on index
        ds = ds[sorted(ds, key=lambda x: ds[x].attrs['index'])]



        ds.to_netcdf(srf_nc_path)

        return srf_nc_path

    def generate_ssi_nc(self, satobj):

        id_sensor = self.get_id_sensor(satobj)

        _, ssi_nc_path = self.get_ssi_nc_path(satobj)


        ds = xr.Dataset()
        ds.attrs["desc"] = f'TSIS-1 solar spectral irradiance for {id_sensor} (0.005 nm spectral resolution)'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = satobj.platform

        ds["ssi"] = xr.DataArray(
            satobj.srf_ssi,
            coords={f"wav_ssi": satobj.srf_ssi_wl},
            attrs={
                "units": "mW m-2 nm-1",
            },
        )
        ds[f"wav_ssi"].attrs["units"] = "nm"


        ds.to_netcdf(ssi_nc_path)

        return ssi_nc_path

    def generate_esun_nc(self, satobj):

        id_sensor = self.get_id_sensor(satobj)

        _, esun_nc_path = self.get_esun_nc_path(satobj)


        ds = xr.Dataset()
        ds.attrs["desc"] = f'ESUN for {id_sensor}'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = satobj.platform

        #ds.attrs["ssi"] = satobj.srf_ssi
        #ds.attrs["ssi_wavelengths"] = satobj.srf_ssi_wl
        #ds.attrs["ssi_units"] = "mW m-2 nm-1"

        #ds.attrs["esun"] = satobj.esun
        #ds.attrs["esun_wavlengths"] = satobj.esun_wl
        #ds.attrs["esun_units"] = "mW m-2 nm-1"


        ds["esun"] = xr.DataArray(
            satobj.esun,
            coords={f"wav_esun": satobj.esun_wl},
            attrs={
                "units": "mW m-2 nm-1",
            },
        )
        ds[f"wav_esun"].attrs["units"] = "nm"


        ds.to_netcdf(esun_nc_path)

        return esun_nc_path

    def run_correction(self, satobj,
                       polymer_base_path: str,
                       polymer_path: str = None,
                       eoread_path: str = None,
                       eotools_path: str = None,
                       core_path: str = None,
                       input_product_level: str = "l1c",
                       #coeff_type: str = None,
                       optional_output_datasets: list = ["SPM"],
                       if_exists: str = "overwrite",
                       polymer_version: str = "v1"):
        """
        polymer_version: which Polymer build polymer_path (etc.) point at -
            mirrors open_output's version parameter.
            - "v1": Polymer_HYPSO_SRF_Oct_2025 - run_polymer's output
              selection is driven by output_datasets, and it writes a
              linear-scale "chla"/"fb" directly.
            - "v2": the newer stock Polymer build - run_polymer no longer
              has an output_datasets parameter (silently ignored if passed -
              it lands in **kwargs and is never used for selection), so
              output selection is driven by outputs_names instead; the
              solver only exposes log-scale "logchl"/"logfb", not "chla"/"fb".
        """

        #polymer_path = Path(satobj.polymer_dir).absolute()

        if polymer_path is not None:
            polymer_path = str(Path(polymer_path).absolute())
            sys.path.insert(0, polymer_path)

        if eotools_path is not None:
            eotools_path = str(Path(eotools_path).absolute())
            sys.path.insert(0, eotools_path)

        if eoread_path is not None:
            eoread_path = str(Path(eoread_path).absolute())
            sys.path.insert(0, eoread_path)

        if core_path is not None:
            core_path = str(Path(core_path).absolute())
            sys.path.insert(0, core_path)

        sys.path.insert(0, polymer_base_path)



        # TODO
        srf_nc_path, srf_nc_path = self.get_srf_nc_path(satobj)

        run_polymer_kwargs = {"srf_getter": "hypso.ac.ac_polymer_srf_getter",
                                "srf_getter_arg": srf_nc_path}


        from eoread.hypso import Level1_HYPSO
        from polymer.main_v5 import run_polymer, run_polymer_dataset, default_output_datasets


        #if coeff_type is not None:
        #    coeff_type_str = "-" + str(coeff_type).lower()
        #else:
        #    coeff_type_str = ""

        # Output (not input) moved into a parent_dir/polymer/ subfolder
        # (2026-08-05, was parent_dir directly) - same reasoning as the
        # ACOLITE adapter's equivalent change (keeps per-run AC output out
        # of the capture directory root; matches the PACE-side Polymer
        # connector's existing convention).
        polymer_output_dir = Path(satobj.parent_dir, "polymer")
        polymer_output_dir.mkdir(parents=True, exist_ok=True)

        match input_product_level.lower():

            case "l1c":
                polymer_l1_input_nc_file = Path(satobj.parent_dir, satobj.l1c_nc_file)
                polymer_l2_output_nc_file = Path(polymer_output_dir, str(satobj.l1c_name) + ".polymer.nc")
            case "l1d":
                polymer_l1_input_nc_file = Path(satobj.parent_dir, satobj.l1d_nc_file)
                polymer_l2_output_nc_file = Path(polymer_output_dir, str(satobj.l1d_name) + ".polymer.nc")
            case _:
                return None



        #import os
        #cwd = os.getcwd()
        #os.chdir(polymer_path)

        # This is from the Feb 2026 version of Polymer
        #from polymer.level1 import Level1
        #from polymer.level2 import Level2
        #from eoread.hypso import Level1_HYPSO
        #from polymer.main_v5 import run_polymer, run_polymer_dataset
        #from core.files.fileutils import mdir
        #polymer_output_file = run_polymer(Level1_HYPSO(polymer_input_file), dir_out=mdir(polymer_output_dir), split_bands=False)

        match polymer_version:
            case "v1":
                output_selection_kwargs = {
                    "output_datasets": default_output_datasets + optional_output_datasets,
                }
            case "v2":
                output_selection_kwargs = {
                    "outputs": "named",
                    "outputs_names": [
                        "latitude", "longitude", "rho_w", "logchl", "logfb",
                        "Rgli", "Rnir", "flags",
                    ] + optional_output_datasets,
                }
            case _:
                raise ValueError(f"Unknown polymer_version: {polymer_version!r}")

        # Run Polymer
        if True:
            output_file = run_polymer(
                Level1_HYPSO(polymer_l1_input_nc_file),
                dir_out=str(polymer_output_dir),
                if_exists = if_exists,
                srf_getter = "hypso.ac.ac_polymer_srf_getter",
                srf_getter_arg = srf_nc_path,
                **output_selection_kwargs,

            )

        try:
            polymer_l2_output_nc_file = Path(output_file).rename(polymer_l2_output_nc_file)
        except FileNotFoundError:
            print("[WARNING] Polymer L2 NetCDF output file has already been renamed.")
            pass

        print(output_file)
        print(polymer_l2_output_nc_file)

        return Path(polymer_l2_output_nc_file)

    def open_output(self, satobj,
                    polymer_l2_output_nc_file: Path = None,
                    input_product_level="l1c",
                    version = "v1"
                    #coeff_type: str = None
                    ):

        #if coeff_type is not None:
        #    coeff_type_str = "-" + str(coeff_type).lower()
        #else:
        #    coeff_type_str = ""

        if polymer_l2_output_nc_file is not None:
            polymer_l2_output_nc_file = Path(polymer_l2_output_nc_file)
        else:
            match input_product_level.lower():

                case "l1c":
                    print("[INFO] Reading Polymer L2 NetCDF output file generated using L1c product.")
                    # parent_dir/polymer/, not parent_dir directly - see
                    # run_correction's matching change.
                    polymer_l2_output_nc_file = Path(satobj.parent_dir, "polymer", str(satobj.l1c_name)+ ".polymer.nc") #frohavet_2025-05-22T11-20-44Z-l1c.nc.polymer.nc

                case "l1d":
                    print("[INFO] Reading Polymer L2 NetCDF output file generated using L1d product.")
                    polymer_l2_output_nc_file = Path(satobj.parent_dir, "polymer", str(satobj.l1d_name) + ".polymer.nc") #frohavet_2025-05-22T11-20-44Z-l1d.nc.polymer.nc


        polymer_l2_output_nc_file = polymer_l2_output_nc_file.absolute()


        if polymer_l2_output_nc_file.is_file():

            if version == "v1":
                polymer_datasets = load_polymer_l2_v1_nc(polymer_l2_output_nc_file)

                try:
                    key = "rho_w"
                    inferred_wavelengths = polymer_datasets['bands'].data

                    # Map inferred Polymer wavelengths to HYPSO wavelengths
                    wl_band_map = get_inferred_wavelength_band_map(satobj, inferred_wavelengths=inferred_wavelengths)

                    # Create empty cube with standard HYPSO cube dims
                    shape = (satobj.spatial_dimensions[0], satobj.spatial_dimensions[1], satobj.bands)
                    cube = np.full(shape=shape, fill_value=np.nan)
                    cube[:,:,wl_band_map] = polymer_datasets[key]

                    satobj.l2a_cube["polymer"] = cube
                    satobj.l2a_cube["polymer"].attrs['l2_variable_name'] = key

                except Exception as ex:
                    print("[ERROR] Unable to load Polymer output dataset.")

            elif version == "v2":

                polymer_datasets = load_polymer_l2_v2_nc(polymer_l2_output_nc_file)

                try:
                    key = "rho_w"
                    inferred_wavelengths = polymer_datasets['bands'].data

                    # Map inferred Polymer wavelengths to HYPSO wavelengths
                    wl_band_map = get_inferred_wavelength_band_map(satobj, inferred_wavelengths=inferred_wavelengths)

                    # Create empty cube with standard HYPSO cube dims
                    shape = (satobj.spatial_dimensions[0], satobj.spatial_dimensions[1], satobj.bands)
                    cube = np.full(shape=shape, fill_value=np.nan)
                    cube[:,:,wl_band_map] = polymer_datasets[key]

                    satobj.l2a_cube["polymer"] = cube
                    satobj.l2a_cube["polymer"].attrs['l2_variable_name'] = key

                except Exception as ex:
                    print("[ERROR] Unable to load Polymer output dataset.")

        else:
            print("[ERROR] Polymer L2 NetCDF output file " + str(polymer_l2_output_nc_file) + " does not exist.")
            polymer_datasets = None


        return polymer_datasets

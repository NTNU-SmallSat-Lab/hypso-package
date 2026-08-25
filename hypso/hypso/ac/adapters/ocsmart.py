"""OC-SMART adapter. Method bodies are HypsoBase's ac_ocsmart_* methods
relocated verbatim (see base.py's ACAdapter docstring) - OC-SMART runs as a
'python3 OCSMART.py' subprocess inside its own installation directory
(satobj.ocsmart_dir), reading staged input from <ocsmart_dir>/L1B/ and writing
HDF5 output to <ocsmart_dir>/L2/."""
from pathlib import Path

import numpy as np

from hypso.load import load_ocsmart_h5

from .base import ACAdapter, get_inferred_wavelength_band_map


class OCSMARTAdapter(ACAdapter):

    key = "ocsmart"

    def stage_input(self, satobj):

        """
        Stages OC-SMART input file to the L1B directory located in the OC-SMART installation directory. The L1d file is copied and renamed to the L1B directory.

        :return: None
        """


        if satobj.ocsmart_dir is not None:
            try:

                dst_dir = Path(satobj.ocsmart_dir, "L1B/")
                dst_dir.mkdir(parents=True, exist_ok=True)

                src_file = satobj.l1d_nc_file
                dst_file = Path(dst_dir, satobj.ocsmart_l1d_input_nc_file.name)

                satobj.ocsmart_l1d_input_nc_file = dst_file

                import shutil
                shutil.copy2(src_file, dst_file)

                print("[INFO] Successfully staged OC-SMART input file to " + str(dst_file))

            except Exception as ex:
                print("[ERROR] Unable to stage OC-SMART input. An error occured.")
                print(ex)

        else:
            print("[ERROR] OC-SMART directory is not configured. The 'ocsmart_dir' attribute is empty.")

        return None

    def run_correction(self, satobj):
        """
        Execute 'OCSMART.py' as a subprocess.

        :return: None
        """

        print("[INFO] Running OC-SMART atmospheric correction as a subprocess.")

        import subprocess
        ocsmart_run_script = Path(satobj.ocsmart_dir, "OCSMART.py")
        subprocess.run(["python3", ocsmart_run_script], cwd=satobj.ocsmart_dir, check=True)

        print("[INFO] Removing staged OC-SMART input file " + str(satobj.ocsmart_l1d_input_nc_file))
        satobj.ocsmart_l1d_input_nc_file.unlink(missing_ok=True)

        print("[INFO] OC-SMART atmospheric correction complete.")

        return None

    def open_output(self, satobj, h5_file_path: Path = None):
        """
        Open and read OC-SMART atmospheric correction HDF5 output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2a_cube' dictionary.

        :param h5_file_path: Path to the OC-SMART HDF5 file (optional)

        :return: "datasets" Dictionary containing 2D and 3D datasets read from the HDF5 and stored as xarray DataArrays.
        """


        if h5_file_path is not None:
            h5_file_path = Path(h5_file_path).absolute()
        else:
            ocsmart_output_dir = Path(satobj.ocsmart_dir, "L2/")
            h5_file_path = Path(ocsmart_output_dir, satobj.ocsmart_l2a_output_h5_file.name)


        if h5_file_path.is_file():
            print("[INFO] Opening OC-SMART output file " + str(h5_file_path))
            datasets = load_ocsmart_h5(h5_file_path = h5_file_path)

        else:
            print("[ERROR] OC-SMART output file " + str(h5_file_path) + " does not exist.")
            return None

        try:
            key = "Rrs"
            inferred_wavelengths = datasets[key].band.to_numpy()

            # Map inferred OC-SMART wavelengths to HYPSO wavelengths
            wl_band_map = get_inferred_wavelength_band_map(satobj, inferred_wavelengths=inferred_wavelengths)

            # Create empty cube with standard HYPSO cube dims
            shape = (satobj.spatial_dimensions[0], satobj.spatial_dimensions[1], satobj.bands)
            cube = np.full(shape=shape, fill_value=np.nan)
            cube[:,:,wl_band_map] = datasets[key]

            satobj.l2a_cube["ocsmart"] = cube
            satobj.l2a_cube["ocsmart"].attrs['l2_variable_name'] = key

        except Exception as ex:
            print("[ERROR] Unable to load OC-SMART L2 Rrs dataset.")

        return datasets

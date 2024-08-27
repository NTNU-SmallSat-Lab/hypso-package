import numpy as np
from importlib.resources import files
from urllib.parse import urlparse, unquote
import urllib.request
import tarfile
from pathlib import Path
import netCDF4 as nc
import shutil

import hypso

#import sys
#print(sys.path)
#print(hypso.__path__)

from ..utils import MyProgressBar, find_file
#from ..utils.utils_file import MyProgressBar, find_file
#from hypso.utils import MyProgressBar, find_file




def get_acolite_repo() -> None:
    """
    Download the ACOLITE Repo as a tar archive

    :return: No return.
    """
    github_url = r"https://github.com/acolite/acolite/archive/refs/tags/20231023.0.tar.gz"
    filename = unquote(urlparse(github_url).path.split("/")[-1])

    output_filename = files(
        'hypso.atmospheric').joinpath(f'data/{filename}')

    try:
        urllib.request.urlretrieve(url=github_url,
                                   filename=str(output_filename),
                                   reporthook=MyProgressBar(filename))

    except Exception as err:
        print(f"Download Failed. {err}")
        print(f"Deleting {filename}")
        # If fail, delete
        output_filename.unlink(missing_ok=True)

    # Uncompress
    tar = tarfile.open(output_filename, 'r:gz')
    dst_path = files(
        'hypso.atmospheric').joinpath(f'data/')
    tar.extractall(dst_path)
    tar.close()


def reset_config_file(path: Path) -> None:
    """
    Reset the config file used for ACOLITE so that the user and password can be included if changed

    :param path: Absolute path of the config file

    :return: No return
    """
    # Reset config file for password and
    with open(path, 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # now change the 2nd line, note that you have to add a newline
    data[61] = "EARTHDATA_u=\n"
    data[62] = "EARTHDATA_p==\n"

    # and write everything back
    with open(path, 'w') as file:
        file.writelines(data)


def run_acolite(output_path: str, atmos_dict: dict, nc_file_acoliteready: Path) -> np.ndarray:
    """
    Run the ACOLITE correction model. Adjustments of the original files are made to ensure they work for HYPSO

    :param hypso_info: Dictionary containing the hypso capture information
    :param atmos_dict: Dictionary containing the information required for the atmospheric correction
    :param nc_file_acoliteready: Absolute path for the .nc file for ACOLITE (L1B)

    :return: Returns surface reflectanc corrected spectral image
    """
    # File and Dir Paths ---------------------------------------------------------------

    atmospheric_pkg_path = str(files(
        'hypso.atmospheric'))

    # Settings file
    settings_path = Path(
        atmospheric_pkg_path, "data", "acolite_settings.txt")

    # Dir to decompress tar file
    acolite_dir = Path(atmospheric_pkg_path, "acolite_main")

    # Path for acolite tar (for decompression)
    acolite_tar = Path(atmospheric_pkg_path, "data", "20231023.0.tar.gz")

    # Script that starts acolite
    launch_acolite_path = Path(atmospheric_pkg_path, "acolite_main", "launch_acolite.py")

    # Create Dir for the output
    acolite_output_dir = Path(output_path)
    acolite_output_dir.mkdir(parents=True, exist_ok=True)

    # If directory exists but empty, delete it, it happend on a pull from github
    if acolite_dir.is_dir():
        list_elements = []
        for ff in acolite_dir.iterdir():
            if "__pycache__" not in ff.stem:
                list_elements.append(ff)
        # If empty
        if len(list_elements) == 0:
            shutil.rmtree(str(acolite_dir))

    # Uncompress ACOLITE if not done and change a file to avoid creating a log
    if not acolite_dir.is_dir():
        # 1) Uncompress ---------------------------------------------
        tar = tarfile.open(acolite_tar, 'r:gz')
        dst_path = atmospheric_pkg_path
        tar.extractall(dst_path)
        tar.close()

        # Renamedst_path
        dst_path = Path(dst_path, "acolite-20231023.0")
        dst_path.rename(Path(acolite_dir))

        # 2) Add __init__ ---------------------------------------------
        Path(acolite_dir, "__init__.py").touch()

        # 3) Change path to comply with package output ---------------------------------------------
        file_to_change = Path(acolite_dir, "acolite", "acolite", "acolite_run.py")

        with open(file_to_change, 'r') as file:
            # read a list of lines into data
            data = file.readlines()

            # now change the 2nd line, note that you have to add a newline
            data[53] = "# " + data[53]
            data[297] = "# " + data[297]
        # and write everything back
        with open(file_to_change, 'w') as file:
            file.writelines(data)

        # Change ------------------------------------------------------------------
        file_to_change = Path(acolite_dir, "acolite", "__init__.py")

        with open(file_to_change, 'r') as file:
            # read a list of lines into data
            data = file.readlines()

            for idx, d in enumerate(data):
                data[idx] = d.replace("from acolite", "from hypso.atmospheric.acolite_main.acolite")

            # now change the 2nd line, note that you have to add a newline
            data[9] = "from pathlib import Path\n"
            data[43] = "from importlib.resources import files\n"
            data[
                54] = 'code_path = os.path.dirname(str(Path(files("hypso.atmospheric"),"acolite_main","acolite","__init__.py")))\n'
            data[67] = 'cfile = str(Path(files("hypso.atmospheric"),"acolite_main","config","config.txt"))\n'

        # and write everything back
        with open(file_to_change, 'w') as file:
            file.writelines(data)

    # If user and password for NASA earthdata provided, updated file ---------------
    user_pwd_file = Path(acolite_dir, "config", "config.txt")

    reset_config_file(user_pwd_file)

    # Read User and password if exists -----------------------------
    try:
        user_earthdata = atmos_dict["user"]
        pwd_earthdata = atmos_dict["password"]

        with open(user_pwd_file, 'r') as file:
            # read a list of lines into data
            data = file.readlines()

        # now change the 2nd line, note that you have to add a newline
        original_line61 = data[61]
        original_line62 = data[62]
        data[61] = data[61].replace("\n", "") + f"{user_earthdata}\n"
        data[62] = data[62].replace("\n", "") + f"{pwd_earthdata}\n"

        # and write everything back
        with open(user_pwd_file, 'w') as file:
            file.writelines(data)
    except Exception as err:
        print("No EARTH DATA user or pwd provided in the dictionary")
        print("Considering creating an account for better data")

    # Import changes ------------------------------------------------
    import_modfication_files = ["launch_acolite.py"]

    for f in import_modfication_files:
        file_to_change = Path(acolite_dir, f)
        with open(file_to_change, 'r') as file:
            # read a list of lines into data
            data = file.readlines()

            for idx, d in enumerate(data):
                data[idx] = d.replace("import acolite as", "import hypso.atmospheric.acolite_main.acolite as")

        # and write everything back
        with open(file_to_change, 'w') as file:
            file.writelines(data)

    # Recursive Changes ---------------------------------------------
    import_modfication_files = []
    recursive_path = Path(acolite_dir, "acolite")
    for subpath in recursive_path.rglob("*.py"):
        if subpath.is_file():
            import_modfication_files.append(subpath)

    for f in import_modfication_files:
        with open(f, 'r') as file:
            # read a list of lines into data
            data = file.readlines()

            for idx, d in enumerate(data):
                data[idx] = d.replace("import acolite as", "import hypso.atmospheric.acolite_main.acolite as")

        # and write everything back
        with open(f, 'w') as file:
            file.writelines(data)

    # Call ACOLITE Script -----------------------------------------------------------
    import subprocess
    try:
        res = subprocess.run(["python", launch_acolite_path, '--cli',
                              '--inputfile', nc_file_acoliteready,
                              '--output', acolite_output_dir,
                              '--settings', settings_path],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        print(res.stdout.decode())

    except Exception as e:
        raise Exception(e)

    # Read and return L2 .nc file generated by acolite
    acolite_l2_file = find_file(output_path, "L2R", ".nc")

    # Read .nc
    final_acolite_l2 = None

    with nc.Dataset(acolite_l2_file, format="NETCDF4") as f:
        group = f
        keys = [i for i in f.variables.keys()]

        toa_keys = [k for k in keys if 'rhos' not in k]
        surface_keys = [kk for kk in keys if 'rhot' not in kk]

        # Add Cube

        for i, k in enumerate(surface_keys):
            current_channel = np.array(group.variables[k][:])
            if final_acolite_l2 is None:
                final_acolite_l2 = np.empty(
                    (current_channel.shape[0], current_channel.shape[1], len(surface_keys)))

            final_acolite_l2[:, :, i] = current_channel

        # TODO: Confirm if zeros should be appended at the beginning or end
        # ACOLITE returns 118 bands
        # If number of bands less that 120, append zeros to the end
        delta = int(120 - final_acolite_l2.shape[2])
        if delta > 0:
            for _ in range(delta):
                zeros_arr = np.zeros((final_acolite_l2.shape[0], final_acolite_l2.shape[1]), dtype=float)
                final_acolite_l2 = np.dstack((final_acolite_l2, zeros_arr))

    # Recover config.txt file -----------------------------------------------
    reset_config_file(user_pwd_file)

    return final_acolite_l2

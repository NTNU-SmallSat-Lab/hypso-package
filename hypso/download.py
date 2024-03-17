from pathlib import Path
import urllib.parse
import urllib.request
from hypso.utils import MyProgressBar
from typing import Union


def download_nc_files(filename_list: list, download_dir: Union[str, None] = None) -> None:
    """
    Bulk download HYPSO-1_L1A data from server. NTNU VPN access is required.

    :param filename_list: list of filenames.
        Example: ["tibet_2022-09-29_0446Z.nc", "xaafuun_2023-09-11_0623Z-l1a.nc"]
    :param download_dir: Absolute path to directory to download files

    :return:
    """

    if download_dir is None:
        raise Exception("Please provide download directory path.")

    server_url = "http://web.hypso.ies.ntnu.no/data/HYPSO-1_L1A/"

    # Create Output Dir
    output_dir = Path(download_dir).absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Download Files
    for f in filename_list:
        dwnld_url = urllib.parse.urljoin(server_url, f)
        output_filename = Path(output_dir, f)
        try:
            urllib.request.urlretrieve(url=dwnld_url,
                                       filename=output_filename,
                                       reporthook=MyProgressBar(f))

        except Exception as err:
            print(f"Download Failed. {err}")
            print(f"Deleting {f}")
            # If fail, delete
            output_filename.unlink(missing_ok=True)

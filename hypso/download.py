from pathlib import Path
import urllib.parse
import urllib.request
from hypso.utils import MyProgressBar


def download_nc_files(filename_list: list, output_dir="",
                      server_url="http://web.hypso.ies.ntnu.no/data/HYPSO-1_L1A/"):
    # Create Output Dir
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Download Files
    for f in filename_list:
        dwnld_url = urllib.parse.urljoin(server_url, f)
        output_filename = Path(output_dir, f)
        try:
            urllib.request.urlretrieve(url=dwnld_url,
                                       filename=output_filename,
                                       reporthook= MyProgressBar(f))

        except Exception as err:
            print(f"Download Failed. {err}")
            print(f"Deleting {f}")
            # If fail, delete
            output_filename.unlink(missing_ok=True)

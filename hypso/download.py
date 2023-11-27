from pathlib import Path
import urllib.parse
import urllib.request
import progressbar

class MyProgressBar():
    def __init__(self,text_prefix):
        self.pbar = None
        self.text_prefix=text_prefix


    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size,
                                              widgets=[progressbar.Bar('=', f'Downloading: {self.text_prefix} [', ']',), ' ', progressbar.Percentage(),],)
            #self.pbar = progressbar.Percentage()
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def download_nc_files(filename_list: list, output_dir="",
                      server_url="http://web.hypso.ies.ntnu.no/data/HYPSO-1_L1A/"):
    # Create Output Dir
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Download Files
    for f in filename_list:
        dwnld_url = urllib.parse.urljoin(server_url, f)
        output_filename = Path(output_dir, f)
        #print(f"Downloading {f}")
        try:
            urllib.request.urlretrieve(url=dwnld_url,
                                       filename=output_filename,
                                       reporthook= MyProgressBar(f))

        except Exception as err:
            print(f"Download Failed. {err}")
            print(f"Deleting {f}")
            # If fail, delete
            output_filename.unlink(missing_ok=True)

'''

# Atmospheric correction functions



def set_acolite_path(self, path: str) -> None:

    self.acolite_path = Path(path).absolute()

    return None




def _run_atmospheric_correction(self, product_name: str) -> None:

    try:
        match product_name.lower():
            case "6sv1":
                self.l2a_cube = self._run_6sv1_atmospheric_correction()
            case "acolite":
                self.l2a_cube = self._run_acolite_atmospheric_correction()
            case "machi":
                self.l2a_cube = self._run_machi_atmospheric_correction() 
            case _:
                print("[ERROR] No such atmospheric correction product supported!")
                return None
    except:
        print("[ERROR] Unable to generate L2a datacube.")

    return None

def _run_6sv1_atmospheric_correction(self, **kwargs) -> xr.DataArray:

    # Py6S Atmospheric Correction
    # aot value: https://neo.gsfc.nasa.gov/view.php?datasetId=MYDAL2_D_AER_OD&date=2023-01-01
    # alternative: https://giovanni.gsfc.nasa.gov/giovanni/
    # atmos_dict = {
    #     'aot550': 0.01,
    #     # 'aeronet': r"C:\Users\alvar\Downloads\070101_151231_Autilla.dubovik"
    # }
    # AOT550 parameter gotten from: https://giovanni.gsfc.nasa.gov/giovanni/

    if self.VERBOSE: 
        print("[INFO] Running 6SV1 atmospheric correction")

    # TODO: which values should we use?
    if self.latitudes is None:
        latitudes = self.latitudes_original # fall back on geometry computed values
    else:
        latitudes = self.latitudes

    if self.longitudes is None:
        longitudes = self.longitudes_original # fall back on geometry computed values
    else:
        longitudes = self.longitudes

    py6s_dict = {
        'aot550': 0.0580000256
    }

    time_capture = parser.parse(self.iso_time)

    cube = self.l1b_cube.to_numpy()

    cube = run_py6s(wavelengths=self.wavelengths, 
                    hypercube_L1=cube, 
                    lat_2d_array=latitudes,
                    lon_2d_array=longitudes,
                    solar_azimuth_angles=self.solar_azimuth_angles,
                    solar_zenith_angles=self.solar_zenith_angles,
                    sat_azimuth_angles=self.sat_azimuth_angles,
                    sat_zenith_angles=self.sat_zenith_angles,
                    iso_time=self.iso_time,
                    py6s_dict=py6s_dict, 
                    time_capture=time_capture,
                    srf=self.srf)
    
    cube = self._format_l2a_dataarray(cube)
    cube.attrs['correction'] = "6sv1"

    return cube


def _run_acolite_atmospheric_correction(self) -> xr.DataArray:

    if hasattr(self, 'acolite_path'):

        nc_file_path = str(self.l1b_nc_file)
        acolite_path = str(self.acolite_path)

        cube = run_acolite(acolite_path=acolite_path, 
                            output_path=self.capture_dir, 
                            nc_file_path=nc_file_path)

        cube = self._format_l2a_dataarray(cube)
        cube.attrs['correction'] = "acolite"

        return cube
    else:
        print("[ERROR] Please set path to ACOLITE source code before generating ACOLITE L2a datacube using \"set_acolite_path()\"")
        print("[INFO] The ACOLITE source code can be downloaded from https://github.com/acolite/acolite")
        return None


# TODO
def _run_machi_atmospheric_correction(self) -> xr.DataArray:

    #print("[WARNING] Minimal Atmospheric Compensation for Hyperspectral Imagers (MACHI) atmospheric correction has not been enabled.")
    #return None




    if self.VERBOSE: 
        print("[INFO] Running MACHI atmospheric correction")

    # start working with ToA reflectance, so we don't have to worry about the solar spectrum
    cube = self.l1c_cube.to_numpy()
    
    T, S, objs = hypso.atmospheric.atm_correction(cube.reshape(-1,120), 
                                                    solar=np.ones(120), 
                                                    verbose=True,
                                                    tol=0.01, 
                                                    est_min_R=0.05)

    # normalize the whole cube
    cube_norm = (cube - S) /T
    cube_norm[self.cloud_mask] = np.nan

    cube = self._format_l2a_dataarray(cube_norm)
    cube.attrs['correction'] = "machi"

    return cube





'''
#!/usr/bin/env python3

import os
import sys
import numpy as np

from pathlib import Path

import sys
sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')


from hypso import Hypso
from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2_nc_file


def main(l1a_nc_path, lats_path=None, lons_path=None):
    # Check if the first file exists
    if not os.path.isfile(l1a_nc_path):
        print(f"Error: The file '{l1a_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l1a_nc_path}")

    nc_file = Path(l1a_nc_path)

    satobj = Hypso(path=nc_file, verbose=True)

    if satobj.l1d_cube is None:

        # Run indirect georeferencing
        if lats_path is not None and lons_path is not None:

            lats_file_path = Path(os.path.join(lats_path, "latitudes.dat"))
            lons_file_path = Path(os.path.join(lons_path, "longitudes.dat"))

            indirect_lats_file_path = Path(os.path.join(lats_path, "latitudes_indirectgeoref.dat"))
            indirect_lons_file_path = Path(os.path.join(lons_path, "longitudes_indirectgeoref.dat"))

            bool_indirect_lats = indirect_lats_file_path.is_file()
            bool_indirect_lons = indirect_lons_file_path.is_file()

            if bool_indirect_lats and bool_indirect_lons:
                lats_path = indirect_lats_file_path
                lons_path = indirect_lons_file_path

            else:
                lats_path = lats_file_path
                lons_path = lons_file_path


            try:

                with open(lats_path, mode='rb') as file:
                    file_content = file.read()
                
                lats = np.frombuffer(file_content, dtype=np.float32)

                lats = lats.reshape(satobj.spatial_dimensions)

                with open(lons_path, mode='rb') as file:
                    file_content = file.read()
                
                lons = np.frombuffer(file_content, dtype=np.float32)
    
                lons = lons.reshape(satobj.spatial_dimensions)


                # Directly provide the indirect lat/lons loaded from the files. This function will run the track geometry computations.
                satobj.run_georeferencing(latitudes=lats, longitudes=lons)

                satobj.generate_l1b_cube()
                satobj.generate_l1c_cube()
                satobj.generate_l1d_cube()

            except Exception as ex:
                print(ex)
                print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

                satobj.run_direct_georeferencing()
                satobj.generate_l1b_cube()
                satobj.generate_l1c_cube()
                satobj.generate_l1d_cube(use_direct_georef=True)

        else:
            satobj.run_direct_georeferencing()

            satobj.generate_l1b_cube()
            satobj.generate_l1c_cube()
            satobj.generate_l1d_cube(use_indirect_georef=False)
            
        write_l1b_nc_file(satobj, overwrite=True, datacube=False)
        write_l1c_nc_file(satobj, overwrite=True, datacube=False)
        write_l1d_nc_file(satobj, overwrite=True, datacube=False)

    else:
        pass

    EARTHDATA_u = "cpenne"
    EARTHDATA_p = "Dec1$!onJG0@1$LogoMen5un!"


    # Atmospheric correction

    TOGGLE_OCSMART = False
    TOGGLE_ACOLITE = False
    TOGGLE_6SV1 = True
    TOGGLE_SREM = True
    TOGGLE_POLYMER = True


    TOGGLE_RUN_AC = True
    TOGGLE_READ_AC = True

    if TOGGLE_OCSMART:
        satobj.ocsmart_dir = "/home/_shared/ARIEL/atmospheric_correction/OC-SMART/OC-SMART_with_HYPSO_9-29-25_release/"
        if TOGGLE_RUN_AC:
            satobj.ac_ocsmart_stage_input()
            satobj.ac_ocsmart_run_correction()
        if TOGGLE_READ_AC:
            satobj.ac_ocsmart_open_output()
            write_l2_nc_file(satobj=satobj, correction="ocsmart", overwrite=True, datacube=False)

    if TOGGLE_ACOLITE:
        satobj.acolite_dir = "/home/_shared/ARIEL/atmospheric_correction/acolite/"
        if TOGGLE_RUN_AC:
            satobj.ac_acolite_run_correction(input_product_level='L1D', EARTHDATA_u=EARTHDATA_u, EARTHDATA_p=EARTHDATA_p)
        if TOGGLE_READ_AC:
            satobj.ac_acolite_open_output()
            write_l2_nc_file(satobj=satobj, correction="acolite_l2r", overwrite=True, datacube=False)
            write_l2_nc_file(satobj=satobj, correction="acolite_l2w", overwrite=True, datacube=False)

    if TOGGLE_6SV1:
        from hypso.ac import run_6sv1_atmospheric_correction
        dem_path = Path("/home/cameron/Nedlastinger/GMTED2km.tif")

        luts_dir = "/home/cameron/Nedlastinger/6S_HYPSO_LUTS"

        cube = run_6sv1_atmospheric_correction(satobj, dem_path, use_luts=True, luts_dir=luts_dir)

        satobj.l2_cube['6sv1'] = cube

        write_l2_nc_file(satobj, correction='6sv1', datacube=False, overwrite=True)

'''
if __name__ == "__main__":

    
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python process_l1d_dir.py <nc_dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    

    #dir_path = "/home/cameron/Nedlastinger/aeronetvenice_2025-09-25T10-01-52Z"
    #dir_path = "/home/cameron/Nedlastinger/princewilliam_2025-12-11T21-13-38Z"
    dir_path = "/home/cameron/Nedlastinger/bankspeninsula_2025-11-27T22-30-10Z"

    base_path = dir_path.rstrip('/')

    folder_name = os.path.basename(base_path)
    l1a_nc_path = os.path.join(base_path, f"{folder_name}-l1a.nc")
    
    #lats_path = os.path.join(base_path, "processing-temp", "latitudes_indirectgeoref.dat")
    #lons_path = os.path.join(base_path, "processing-temp", "longitudes_indirectgeoref.dat") 
    #lats_path = os.path.join(base_path, "processing-temp", "latitudes.dat")
    #lons_path = os.path.join(base_path, "processing-temp", "longitudes.dat") 


    lats_path = os.path.join(base_path, "processing-temp/")
    lons_path = os.path.join(base_path, "processing-temp/") 


    main(l1a_nc_path, lats_path, lons_path)
'''

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python process_l1d_dir.py <nc_dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    base_path = dir_path.rstrip('/')

    folder_name = os.path.basename(base_path)
    l1a_nc_path = os.path.join(base_path, f"{folder_name}-l1a.nc")
    l1d_nc_path = os.path.join(base_path, f"{folder_name}-l1d.nc")

    lats_path = os.path.join(base_path, "processing-temp/")
    lons_path = os.path.join(base_path, "processing-temp/") 

    #print(base_path)
    #print(folder_name)
    #print(lat_file)
    #print(lon_file)
    #lats_path = sys.argv[2] if len(sys.argv) == 4 else None
    #lons_path = sys.argv[3] if len(sys.argv) == 4 else None

    main(l1a_nc_path, lats_path, lons_path)
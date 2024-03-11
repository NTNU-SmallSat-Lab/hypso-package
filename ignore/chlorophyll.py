from natsort import natsorted
# Numerical
import numpy as np
from scipy.interpolate import interp1d
from .chl_tpca_algorithms import calculate_chl_ocx
import os
from tqdm import tqdm
from scipy.interpolate import griddata


class Chlorophyll:
    method = ''
    algorithm_keys_numeric = [443.0, 490.0, 510.0, 555.0, 670.0]
    satellite = None

    def __init__(self, satellite):
        self.satellite = satellite

    def interpolate_spectra_lon_lat(self, sat_name, hypercube_to_interpolate):
        # Step 1: Destination Hypercube

        # This cube has the spatial Dimensions of Hypso (as it's the destination interpolation) but
        # number of bands of the other satellite (because we interpolate those bands)
        hypercube_interpolated = np.empty((self.satellite.spatialDim[0], self.satellite.spatialDim[1],
                                           hypercube_to_interpolate.shape[2]))

        # Step 2: Coordinates at which to interpolate (Hypso)
        xx = self.satellite.lat
        yy = self.satellite.lon

        # Step 3: Get source coordinates (Other satellites)
        x = self.satellite.other_sensors[sat_name].lat.flatten()
        y = self.satellite.other_sensors[sat_name].lon.flatten()

        # Experimental: To Speed Process --------------------------
        other_sat_lat = self.satellite.other_sensors[sat_name].lat
        other_sat_lon = self.satellite.other_sensors[sat_name].lon
        # Get Smaller Area where Two Satellites MAtch (Faster
        lat_condition = np.logical_and(other_sat_lat >= np.nanmin(self.satellite.lat),
                                       other_sat_lat <= np.nanmax(self.satellite.lat))
        lon_condition = np.logical_and(other_sat_lon >= np.nanmin(self.satellite.lon),
                                       other_sat_lon <= np.nanmax(self.satellite.lon))
        coord_condition = np.logical_and(lat_condition, lon_condition)

        # First Indices along each row
        first_on_row = np.argmax(coord_condition, axis=1)[:, None]
        min_col = np.min(np.ma.masked_equal(first_on_row, 0.0, copy=False))

        last_on_row = np.argmax(np.fliplr(coord_condition), axis=1)[:, None]
        max_col = coord_condition.shape[1] - \
            np.min(np.ma.masked_equal(last_on_row, 0.0, copy=False))

        first_on_col = np.argmax(coord_condition, axis=0)[:, None]
        min_row = np.min(np.ma.masked_equal(first_on_col, 0.0, copy=False))

        last_on_col = np.argmax(np.flipud(coord_condition), axis=0)[:, None]
        max_row = coord_condition.shape[0] - \
            np.min(np.ma.masked_equal(last_on_col, 0.0, copy=False))

        other_sat_lat_cropped = other_sat_lat[min_row:max_row +
                                              1, min_col:max_col + 1]
        other_sat_lon_cropped = other_sat_lon[min_row:max_row +
                                              1, min_col:max_col + 1]

        x = other_sat_lat_cropped.flatten()
        y = other_sat_lon_cropped.flatten()
        # End of Experimental --------------------------------------

        # Step 4: Interpolate 2D Array of Chlorophyll form Other Satellites
        # (Modis and Sentinel Interpolated to Hypso Coordinates)

        for i in tqdm(range(hypercube_to_interpolate.shape[2])):
            chanel_spectra = hypercube_to_interpolate[:, :, i].flatten()

            # Experimental
            chanel = hypercube_to_interpolate[min_row:max_row +
                                              1, min_col:max_col + 1, i]

            # Flattening and getting nan index
            chanel_raveled = chanel.flatten()
            # chanel_raveled_nan_index = np.where(chanel_raveled == np.nan) # this returns index
            chanel_raveled_nan_index = chanel_raveled == np.nan  # this returns bool mask

            # Selecting non np.nan
            chanel_raveled_filtered = chanel_raveled[~chanel_raveled_nan_index]
            x_filtered = x[~chanel_raveled_nan_index]
            y_filtered = y[~chanel_raveled_nan_index]

            # Interpolate Grid
            hypercube_interpolated[:, :, i] = griddata((x_filtered, y_filtered), chanel_raveled_filtered, (xx, yy),
                                                       method='linear')

        return hypercube_interpolated

    def ocx_estimation(self, satName=None, interpMethodUsed=None, special_extent=None,
                       overlapSatImg=False):
        chl_key = 'ocx'

        print("\n-------  Chlorophyll Estimator  ----------")
        print("Estimating Chlorophyll with OCX - Key: 'ocx'")
        # Get Satellite parameters
        satellite_wl, hypercube, spatial_dim, water_mask = self.get_data_for_sat(
            satName, interpMethodUsed)
        # if satName == 'hypso':
        #     satellite_wl = list(self.satellite.wavelengths)  # Wavelengths necessary, interpolation needed
        #     hypercube = self.satellite.hypercube['L1']
        #     spatial_dim = self.satellite.spatialDim
        #     water_mask = self.satellite.waterMask
        # elif satName == 'hypso_acolite' or satName == 'hypso_py6s':
        #     satellite_wl = list(self.satellite.wavelengths)  # Wavelengths necessary, interpolation needed
        #     hypercube = self.satellite.hypercube['L2']
        #     spatial_dim = self.satellite.spatialDim
        #     water_mask = self.satellite.waterMask
        # elif satName is not None:
        #     # Coming from Interpolation
        #     if interpMethodUsed is not None:
        #         # dic_name = interpMethodUsed + '_' + satName
        #         dic_name = satName
        #
        #         satellite_wl = list(self.satellite.other_sensors[
        #                                 dic_name].wavelengths)  # Wavelengths necessary, interpolation needed
        #         hypercube = self.satellite.other_sensors[
        #             dic_name].hypercube['L2']
        #         spatial_dim = self.satellite.other_sensors[
        #             dic_name].spatialDim
        #         water_mask = self.satellite.other_sensors[
        #             dic_name].waterMask
        #     # Chlorophyll Methods when Creating Modis or Aqua
        #     else:
        #         satellite_wl = list(self.satellite.wavelengths)  # Wavelengths necessary, interpolation needed
        #         hypercube = self.satellite.hypercube['L2']
        #         spatial_dim = self.satellite.spatialDim
        #         water_mask = self.satellite.waterMask
        #
        # else:
        #     raise Exception("Can't calculate Chlorophyll")

        # * -------------------------------------------------------------
        # *         CHLOROPHYLL ESTIMATION
        # * -------------------------------------------------------------

        # Keys to Interpolation
        algorithm_keys_to_interpolate = natsorted(
            list(set(self.algorithm_keys_numeric) - set(satellite_wl)))

        # Keys that exist on the data
        algorithm_keys_available = natsorted(
            list(set(self.algorithm_keys_numeric) - set(algorithm_keys_to_interpolate)))

        print("\nKeys to Interpolate: ", algorithm_keys_to_interpolate)
        print("Keys Available: ", algorithm_keys_available)

        # TODO: Get Date From Directory
        # date_string = ncdat.time_coverage_start
        # timestamp = datetime.strptime(
        #     date_string, '%Y-%m-%dT%H:%M:%S.%f%z')
        # pretty_date = datetime.strftime(timestamp, '%d/%b/%y')
        pretty_date = 'Pretty Date Missing'

        # Create Array to Store Rrs at Lambda for Chl Estimation
        Rrs_stacked_algorithm = np.empty(
            (spatial_dim[0] * spatial_dim[1], len(self.algorithm_keys_numeric)))
        fill_value_Rrs = 0.0

        # * Interpolation of Bands
        column_values = []
        for i, k in enumerate(self.algorithm_keys_numeric):
            if k in algorithm_keys_available:  # We have the data, no need to interpolate
                indexRrs = satellite_wl.index(k)
                Rrs_individual = hypercube[:, :, indexRrs]
                if np.ma.is_masked(Rrs_individual):
                    Rrs_individual = Rrs_individual.filled(np.nan)

                Rrs_individual = Rrs_individual.flatten()
                Rrs_individual[Rrs_individual < 0] = np.nan
                Rrs_stacked_algorithm[:, i] = Rrs_individual

            elif k in algorithm_keys_to_interpolate:

                # TODO: Replace After and Before band selection with Library that can do it by itself
                current_lambda_to_interpolate = k
                # Find position between satellite lamddas to interpolate there
                previous_idx = 0
                for idx in range(len(satellite_wl) - 1):
                    if (satellite_wl[idx] < current_lambda_to_interpolate) and (
                            current_lambda_to_interpolate < satellite_wl[idx + 1]):
                        previous_idx = idx
                        # continue

                lambda_previous = satellite_wl[previous_idx]
                Rrs_individual_previous = hypercube[:, :, previous_idx].flatten(
                )

                Rrs_individual_previous[Rrs_individual_previous < 0] = np.nan
                lambda_after = satellite_wl[previous_idx + 1]
                Rrs_individual_after = hypercube[:,
                                                 :, previous_idx + 1].flatten()
                Rrs_individual_after[Rrs_individual_after < 0] = np.nan

                x = np.array([lambda_previous, lambda_after])
                y = np.column_stack(
                    (Rrs_individual_previous, Rrs_individual_after))
                intf = interp1d(x, y, axis=1)
                Rrs_stacked_algorithm[:, i] = intf(
                    current_lambda_to_interpolate)
                # interpolated_Rrs = [current_lambda_to_interpolate]
                # interpolated_Rrs = np.interp(current_lambda_to_interpolate, [lambda_previous, lambda_after],)

            # TODO Remove the clipping based on feedback from GIS Stackexchange

            column_values.append(k)

        Rrs_stacked_algorithm[Rrs_stacked_algorithm < 0] = np.nan

        # Reduce Information based on dataTrim
        Rrs_stacked_algorithm = Rrs_stacked_algorithm

        # * Chlorophyll Estimation
        r443 = Rrs_stacked_algorithm[:, 0].reshape(spatial_dim)
        r490 = Rrs_stacked_algorithm[:, 1].reshape(spatial_dim)
        r510 = Rrs_stacked_algorithm[:, 2].reshape(spatial_dim)
        r555 = Rrs_stacked_algorithm[:, 3].reshape(spatial_dim)
        r670 = Rrs_stacked_algorithm[:, 4].reshape(spatial_dim)

        hypercube_to_save = np.dstack((r443, r490, r510, r555, r670))
        print(
            f"Value Hypercube Chl Estimation (Before)\nMin: {np.nanmin(hypercube_to_save)}\nMax: {np.nanmax(hypercube_to_save)}")
        # Interpolate Cube to Hypso Coordinates
        if ('hypso' not in satName) and (interpMethodUsed is not None):
            print("\nDoing Spatial Interpolation on OCX\n")
            hypercube_to_save = self.interpolate_spectra_lon_lat(
                satName, hypercube_to_save)
            r443 = hypercube_to_save[:, :, 0]
            r490 = hypercube_to_save[:, :, 1]
            r510 = hypercube_to_save[:, :, 2]
            r555 = hypercube_to_save[:, :, 3]
            r670 = hypercube_to_save[:, :, 4]

        print(
            f"Value Hypercube Chl Estimation (After)\nMin: {np.nanmin(hypercube_to_save)}\nMax: {np.nanmax(hypercube_to_save)}")
        divide_condition = np.logical_or(r555 == 0, r555 == np.nan)

        with np.errstate(divide='ignore'):

            div = np.where(divide_condition, np.nan, r443 / r555)
            # div2 = r490 / r555
            div2 = np.where(divide_condition, np.nan, r490 / r555)
            # div3 = r510 / r555
            div3 = np.where(divide_condition, np.nan, r510 / r555)
            # mbr=np.maximum(r443/r555,r490/r555,where=ocean_mask.flatten())
            mbr = np.maximum(div, div2)
            # mbr2=np.maximum(caca_div,caca_div2,where=ocean_mask)
            mbr2 = np.maximum(mbr, div3)  # Calculate max band ratio

            lmbr = np.where(np.logical_or(mbr2 == 0, mbr2 ==
                            np.nan), np.nan, np.log10(mbr2))

            ocx_poly = [0.3255, -2.7677, 2.4409, -1.1288, -0.4990]
            # Already Implementes 10** to Unlog
            chl_estimation = calculate_chl_ocx(ocx_poly, lmbr)

            # Mask with Wate Mask Determined Before
            chl_estimation[np.logical_not(water_mask)] = np.nan

            print(
                f"\nMin Chl: {np.nanmin(chl_estimation)}\nMax Chl: {np.nanmax(chl_estimation)}")

            plot_title = satName.upper() + ' Chl with '
            if satName == 'hypso':
                chl_external_key = 'hypso_' + chl_key
            elif satName == 'hypso_acolite' or satName == 'hypso_py6s':
                chl_external_key = satName + '_' + chl_key
            else:
                # when Interpolation done
                if interpMethodUsed is not None:
                    chl_external_key = interpMethodUsed + '_' + \
                        satName + '_' + chl_key  # 'interp_aqua for example

                    plot_title = satName.upper() + ' Chl (' + interpMethodUsed + ') with '
                # No interpolation done, when creating sat
                else:
                    chl_external_key = satName + '_' + chl_key  # 'interp_aqua for example

            # Saving File

            dict_to_save = {
                'wavelengths_algorithm': self.algorithm_keys_numeric,
                'hypercube_algorithm': hypercube_to_save,
                'chl_algorithm': chl_estimation,
            }
            # Save Interpolated Cube
            self.satellite.chl[chl_external_key] = dict_to_save

            # np.save(os.path.join(self.satellite.outputDir, chl_external_key + '_export.npy'),
            #         dict_to_save)

            # To Load data=np.load("file.npy")
            # data.item().get('wavelengths')
            # data.item().get('hypercube')

            # Plotting

            self.satellite.plotter.self_chl(plotTitle=plot_title + chl_key,
                                            chl_key=chl_key,
                                            satName=satName,
                                            interpMethodUsed=interpMethodUsed,
                                            special_extent=special_extent,
                                            overlapSatImg=overlapSatImg)

    def siver_TOA_chl(self):
        print("Estimating Chlorophyll with Sivert Inc. - Key: 'sivert_chl'")
        hypercube = self.satellite.L1_hypercube
        water_mask = self.satellite.waterMask
        satellite_wl = self.satellite.wavelengths
        spatial_dim = self.satellite.spatialDim

        # def is_water(spectra, wl, treshold=2.4):
        #     # Empirical super simple water classifier
        #     C1 = np.argmin(abs(wl - 460))
        #     C2 = np.argmin(abs(wl - 650))
        #     return spectra[C1] / spectra[C2] > treshold

        def oc6_pace(cube, wl, treshold=1):

            # From https://doi.org/10.1016/j.rse.2019.04.021

            result = np.zeros([cube.shape[0], cube.shape[1]])

            # Iitialize coefficients
            c = np.array([
                0.94297, -3.18493, 2.33682, -1.23923, 0.18697
            ])

            def get_chl_a(spectra, wl):
                # Find blue
                MBR_b = np.max([
                    spectra[np.argmin(abs(wl - 443))],
                    spectra[np.argmin(abs(wl - 490))],
                    spectra[np.argmin(abs(wl - 510))],
                ])

                # Find Green(red)
                MBR_g = np.mean([
                    spectra[np.argmin(abs(wl - 555))],
                    spectra[np.argmin(abs(wl - 678))],
                ])

                divide_condition = np.logical_or(MBR_g == 0, MBR_g == np.nan)

                X = []
                with np.errstate(divide='ignore'):
                    X = np.where(divide_condition, np.nan, MBR_b / MBR_g)

                    log_res = 0
                    for i in range(len(c)):
                        log_res += c[i] * X ** i

                    return 10 ** log_res

            # Iterate through cube
            for xx in range(spatial_dim[0]):
                for yy in range(spatial_dim[1]):
                    spectra = cube[xx, yy, :]

                    if water_mask[xx, yy]:
                        r = get_chl_a(spectra, wl)
                        if r > treshold:
                            result[xx, yy] = r
                        else:
                            result[xx, yy] = 0
                    else:
                        result[xx, yy] = 0

            return result

        chl_estimation = oc6_pace(hypercube, satellite_wl, treshold=0)

        # TODO: Implement filter of Water from CHl Calculation to Speed up Process
        chl_estimation[np.logical_not(water_mask)] = np.nan

        print(
            f"HYPSO (Sivert CHL)------\nMin Chl: {np.nanmin(chl_estimation)}\nMax Chl: {np.nanmax(chl_estimation)}")

        self.satellite.chl['sivert_chl'] = chl_estimation

    def get_data_for_sat(self, satName, interpMethodUsed):
        if satName == 'hypso':
            # Wavelengths necessary, interpolation needed
            satellite_wl = list(self.satellite.wavelengths)
            hypercube = self.satellite.hypercube['L1']
            spatial_dim = self.satellite.spatialDim
            water_mask = self.satellite.waterMask
        elif satName == 'hypso_acolite' or satName == 'hypso_py6s':
            # Wavelengths necessary, interpolation needed
            satellite_wl = list(self.satellite.wavelengths)
            hypercube = self.satellite.hypercube['L2']
            spatial_dim = self.satellite.spatialDim
            water_mask = self.satellite.waterMask
        elif satName is not None:
            # Coming from Interpolation
            if interpMethodUsed is not None:
                # dic_name = interpMethodUsed + '_' + satName
                dic_name = satName

                satellite_wl = list(self.satellite.other_sensors[
                    dic_name].wavelengths)  # Wavelengths necessary, interpolation needed
                hypercube = self.satellite.other_sensors[
                    dic_name].hypercube['L2']
                spatial_dim = self.satellite.other_sensors[
                    dic_name].spatialDim
                water_mask = self.satellite.other_sensors[
                    dic_name].waterMask
            # Chlorophyll Methods when Creating Modis or Aqua
            else:
                # Wavelengths necessary, interpolation needed
                satellite_wl = list(self.satellite.wavelengths)
                hypercube = self.satellite.hypercube['L2']
                spatial_dim = self.satellite.spatialDim
                water_mask = self.satellite.waterMask

        else:
            raise Exception("Can't calculate Chlorophyll")
        return satellite_wl, hypercube, spatial_dim, water_mask

    def oci_estimation(self, satName=None, interpMethodUsed=None, special_extent=None,
                       overlapSatImg=False):
        chl_key = 'ci'

        print("\n-------  Chlorophyll Estimator  ----------")
        print("Estimating Chlorophyll with CI - Key: 'ci'")
        # Get Satellite parameters
        satellite_wl, hypercube, spatial_dim, water_mask = self.get_data_for_sat(
            satName, interpMethodUsed)

        # * -------------------------------------------------------------
        # *         CHLOROPHYLL ESTIMATION
        # * -------------------------------------------------------------

        # Keys to Interpolation
        algorithm_keys_to_interpolate = natsorted(
            list(set(self.algorithm_keys_numeric) - set(satellite_wl)))

        # Keys that exist on the data
        algorithm_keys_available = natsorted(
            list(set(self.algorithm_keys_numeric) - set(algorithm_keys_to_interpolate)))

        print("\nKeys to Interpolate: ", algorithm_keys_to_interpolate)
        print("Keys Available: ", algorithm_keys_available)

        # TODO: Get Date From Directory
        # date_string = ncdat.time_coverage_start
        # timestamp = datetime.strptime(
        #     date_string, '%Y-%m-%dT%H:%M:%S.%f%z')
        # pretty_date = datetime.strftime(timestamp, '%d/%b/%y')
        pretty_date = 'Pretty Date Missing'

        # Create Array to Store Rrs at Lambda for Chl Estimation
        Rrs_stacked_algorithm = np.empty(
            (spatial_dim[0] * spatial_dim[1], len(self.algorithm_keys_numeric)))
        fill_value_Rrs = 0.0

        # * Interpolation of Bands
        column_values = []
        for i, k in enumerate(self.algorithm_keys_numeric):
            if k in algorithm_keys_available:  # We have the data, no need to interpolate
                indexRrs = satellite_wl.index(k)
                Rrs_individual = hypercube[:, :, indexRrs]
                if np.ma.is_masked(Rrs_individual):
                    Rrs_individual = Rrs_individual.filled(np.nan)

                Rrs_individual = Rrs_individual.flatten()
                Rrs_individual[Rrs_individual < 0] = np.nan
                Rrs_stacked_algorithm[:, i] = Rrs_individual

            elif k in algorithm_keys_to_interpolate:

                # TODO: Replace After and Before band selection with Library that can do it by itself
                current_lambda_to_interpolate = k
                # Find position between satellite lamddas to interpolate there
                previous_idx = 0
                for idx in range(len(satellite_wl) - 1):
                    if (satellite_wl[idx] < current_lambda_to_interpolate) and (
                            current_lambda_to_interpolate < satellite_wl[idx + 1]):
                        previous_idx = idx
                        # continue

                lambda_previous = satellite_wl[previous_idx]
                Rrs_individual_previous = hypercube[:, :, previous_idx].flatten(
                )

                Rrs_individual_previous[Rrs_individual_previous < 0] = np.nan
                lambda_after = satellite_wl[previous_idx + 1]
                Rrs_individual_after = hypercube[:,
                                                 :, previous_idx + 1].flatten()
                Rrs_individual_after[Rrs_individual_after < 0] = np.nan

                x = np.array([lambda_previous, lambda_after])
                y = np.column_stack(
                    (Rrs_individual_previous, Rrs_individual_after))
                intf = interp1d(x, y, axis=1)
                Rrs_stacked_algorithm[:, i] = intf(
                    current_lambda_to_interpolate)
                # interpolated_Rrs = [current_lambda_to_interpolate]
                # interpolated_Rrs = np.interp(current_lambda_to_interpolate, [lambda_previous, lambda_after],)

            # TODO Remove the clipping based on feedback from GIS Stackexchange

            column_values.append(k)

        Rrs_stacked_algorithm[Rrs_stacked_algorithm < 0] = np.nan

        # Reduce Information based on dataTrim
        Rrs_stacked_algorithm = Rrs_stacked_algorithm

        # * Chlorophyll Estimation
        r443 = Rrs_stacked_algorithm[:, 0].reshape(spatial_dim)
        r490 = Rrs_stacked_algorithm[:, 1].reshape(spatial_dim)
        r510 = Rrs_stacked_algorithm[:, 2].reshape(spatial_dim)
        r555 = Rrs_stacked_algorithm[:, 3].reshape(spatial_dim)
        r670 = Rrs_stacked_algorithm[:, 4].reshape(spatial_dim)

        hypercube_to_save = np.dstack((r443, r490, r510, r555, r670))
        print(
            f"Value Hypercube Chl Estimation (Before)\nMin: {np.nanmin(hypercube_to_save)}\nMax: {np.nanmax(hypercube_to_save)}")
        # Interpolate Cube to Hypso Coordinates
        if ('hypso' not in satName) and (interpMethodUsed is not None):
            print("\nDoing Spatial Interpolation on OCX\n")
            hypercube_to_save = self.interpolate_spectra_lon_lat(
                satName, hypercube_to_save)
            r443 = hypercube_to_save[:, :, 0]
            r490 = hypercube_to_save[:, :, 1]
            r510 = hypercube_to_save[:, :, 2]
            r555 = hypercube_to_save[:, :, 3]
            r670 = hypercube_to_save[:, :, 4]

        print(
            f"Value Hypercube Chl Estimation (After)\nMin: {np.nanmin(hypercube_to_save)}\nMax: {np.nanmax(hypercube_to_save)}")
        divide_condition = np.logical_or(r555 == 0, r555 == np.nan)

        with np.errstate(divide='ignore'):

            div = np.where(divide_condition, np.nan, r443 / r555)
            # div2 = r490 / r555
            div2 = np.where(divide_condition, np.nan, r490 / r555)
            # div3 = r510 / r555
            div3 = np.where(divide_condition, np.nan, r510 / r555)
            # mbr=np.maximum(r443/r555,r490/r555,where=ocean_mask.flatten())
            mbr = np.maximum(div, div2)
            # mbr2=np.maximum(caca_div,caca_div2,where=ocean_mask)
            mbr2 = np.maximum(mbr, div3)  # Calculate max band ratio

            lmbr = np.where(np.logical_or(mbr2 == 0, mbr2 ==
                            np.nan), np.nan, np.log10(mbr2))

            ocx_poly = [0.3255, -2.7677, 2.4409, -1.1288, -0.4990]
            # Already Implementes 10** to Unlog
            chl_estimation = calculate_chl_ocx(ocx_poly, lmbr)

            # Mask with Wate Mask Determined Before
            chl_estimation[np.logical_not(water_mask)] = np.nan

            print(
                f"\nMin Chl: {np.nanmin(chl_estimation)}\nMax Chl: {np.nanmax(chl_estimation)}")

            plot_title = satName.upper() + ' Chl with '
            if satName == 'hypso':
                chl_external_key = 'hypso_' + chl_key
            elif satName == 'hypso_acolite' or satName == 'hypso_py6s':
                chl_external_key = satName + '_' + chl_key
            else:
                # when Interpolation done
                if interpMethodUsed is not None:
                    chl_external_key = interpMethodUsed + '_' + \
                        satName + '_' + chl_key  # 'interp_aqua for example

                    plot_title = satName.upper() + ' Chl (' + interpMethodUsed + ') with '
                # No interpolation done, when creating sat
                else:
                    chl_external_key = satName + '_' + chl_key  # 'interp_aqua for example

            # Saving File

            dict_to_save = {
                'wavelengths_algorithm': self.algorithm_keys_numeric,
                'hypercube_algorithm': hypercube_to_save,
                'chl_algorithm': chl_estimation,
            }
            # Save Interpolated Cube
            self.satellite.chl[chl_external_key] = dict_to_save

            # np.save(os.path.join(self.satellite.outputDir, chl_external_key + '_export.npy'),
            #         dict_to_save)

            # To Load data=np.load("file.npy")
            # data.item().get('wavelengths')
            # data.item().get('hypercube')

            # Plotting

            self.satellite.plotter.self_chl(plotTitle=plot_title + chl_key,
                                            chl_key=chl_key,
                                            satName=satName,
                                            interpMethodUsed=interpMethodUsed,
                                            special_extent=special_extent,
                                            overlapSatImg=overlapSatImg)

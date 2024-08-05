import numpy as np
import csv
import pyproj as prj
import pandas as pd
from typing import Tuple, Literal
import skimage
from pathlib import Path
# GIS
import cartopy.crs as ccrs


def start_coordinate_correction(points_path: Path, satinfo: dict, proj_metadata: dict,
                                correction_type: Literal["affine", "homography",
                                "polynomial", "lstsq"] = "affine") -> Tuple[np.ndarray, np.ndarray]:
    """
    Start coordinate correction process

    :param points_path: Path of the .points file generated with QGIS software
    :param satinfo: Dictionary containing capture information
    :param proj_metadata: Dictionary containing projection metadata of the capture
    :param correction_type: String with the method out of "affine", "homography", polynomial" and "lstsq"

    :return: Corrected lat and lon coordinate 2D arrays
    """
    # point_file = find_file(top_folder_name, ".points")
    if points_path is None:
        print("Points File Was Not Found. No Correction done.")
        return satinfo["lat_original"], satinfo["lon_original"]
    else:
        print("Doing manual coordinate correction with .points file")
        lat, lon = coordinate_correction(
            points_path, proj_metadata,
            satinfo["lat_original"], satinfo["lon_original"], correction_type)

        return lat, lon


def coordinate_correction_matrix(filename: Path, projection_metadata: dict, correction_type: Literal["affine", "homography",
                                "polynomial", "lstsq"]) -> dict:
    """
    Generate the coordinate correction matrix

    :param filename: Absolute path for the .points file generated with QGIS
    :param projection_metadata: Dictionary containing projection metadata of hypso capture
    :param correction_type: String indicating the correction type. Options are "affine", "homography", polynomial"
        and "lstsq"

    :return: Dictionary with the correction matrix
    """
    # Hypso CRS
    inproj = projection_metadata['inproj']
    # Convert WKT projection information into a cartopy projection
    projcs = inproj.GetAuthorityCode('PROJCS')
    projection_hypso = ccrs.epsg(projcs)
    projection_hypso_transformer = prj.Proj(projection_hypso)
    # Create projection transformer to go from QGIS to Hypso

    # Load GCP File
    gcps = None
    with open(filename, encoding="utf8", errors='ignore') as csv_file:
        # Open CSV file
        reader = csv.reader(csv_file)

        next(reader, None)  # Skip row with CRS data
        # mapX, mapY, sourceX, sourceY, enable, dX, dY, residual
        column_names = list(next(reader, None))

        # Get CRS
        qgis_crs = 'epsg:3857'  # QGIS Also Uses as a Defaultb EPSG:4326 or EPSG:32617
        hypso_crs = projection_metadata["crs"]  # 'epsg:32632'

        # Transformer from dataset crs to destination crs
        transformer_src_dest = prj.Transformer.from_crs(
            qgis_crs, hypso_crs, always_xy=True)

        # Iterate through rows and
        for gcp in reader:

            if gcps is None:
                gcps = gcp
            else:
                gcps = np.row_stack((gcps, gcp))

        # Convert data to Dataframe

        gcps = pd.DataFrame(gcps, columns=column_names)
        gcps.rename(columns={"mapX": "mapLon",
                             "mapY": "mapLat",
                             "sourceX": "uncorrectedHypsoLon",
                             "sourceY": "uncorrectedHypsoLat"}, inplace=True)

        # Add Columns for Map Results Convert from MapCRS to Hypso CRS

        gcps['transformed_mapLon'] = pd.Series(dtype='float')  # X is Longitude
        gcps['transformed_mapLat'] = pd.Series(dtype='float')  # Y is latitude

        # Iterate through each gcps array
        for index, row in gcps.iterrows():
            # Rows are copies, not references any more
            row = row.copy()
            # Transform Coordinates to Hypso CRS
            # inverse = False -> From QGIS to Hypso
            # inverse = True -> From Hypso to QGIS

            # Option 1: Assume that everything is already in coordinates
            transformed_lon, transformed_lat = projection_hypso_transformer(
                row['mapLon'], row['mapLat'], inverse=True)

            # Option 2: Assume that we need to move from one system to another
            # transformed_lon, transformed_lat = transformer_src_dest.transform(row['mapLon'], row['mapLat'])

            gcps.loc[index, 'transformed_mapLon'] = transformed_lon
            gcps.loc[index, 'transformed_mapLat'] = transformed_lat

            transformed_lon, transformed_lat = projection_hypso_transformer(row['uncorrectedHypsoLon'],
                                                                            row['uncorrectedHypsoLat'], inverse=True)
            gcps.loc[index, 'uncorrectedHypsoLon'] = transformed_lon
            gcps.loc[index, 'uncorrectedHypsoLat'] = transformed_lat

    # Estimate transform
    hypso_src = gcps[["uncorrectedHypsoLon",
                      "uncorrectedHypsoLat"]].to_numpy().astype(np.float32)
    hypso_dst = gcps[["transformed_mapLon",
                      "transformed_mapLat"]].to_numpy().astype(np.float32)

    print("Hypso Source\n",
          gcps[["uncorrectedHypsoLon", "uncorrectedHypsoLat"]])
    print("Hypso Destination\n",
          gcps[["transformed_mapLon", "transformed_mapLat"]])
    print('--------------------------------')

    M = None

    if correction_type == "polynomial":
        # 2nd Order skimage
        M = skimage.transform.PolynomialTransform()
        M.estimate(hypso_src, hypso_dst, order=2)


    elif correction_type == "homography":
        M = skimage.transform.ProjectiveTransform()
        M.estimate(hypso_src, hypso_dst)

    elif correction_type == "affine":
        # [[a0  a1  a2]
        # [b0  b1  b2]
        # [0   0    1]]
        M = skimage.transform.AffineTransform()
        M.estimate(hypso_src, hypso_dst)

    elif correction_type == "lstsq":
        # Using Numpy Least Squares ( Good Results!)
        M = []
        x = hypso_src[:, 0]
        y = hypso_dst[:, 0]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        M.append([m, c])
        x = hypso_src[:, 1]
        y = hypso_dst[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        M.append([m, c])

    return {correction_type: M}  # lat_coeff, lon_coeff


def coordinate_correction(point_file: Path, projection_metadata: dict, originalLat: np.ndarray, originalLon: np.ndarray,
                          correction_type: Literal["affine", "homography", "polynomial", "lstsq"]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct the coordinates of the latitude and longitude 2D arrays

    :param point_file: Absolute path to the .points file generated with QGIS
    :param projection_metadata: Dictionary containing the projection information of the capture
    :param originalLat: 2D latitude array
    :param originalLon: 2D longitude array
    :param correction_type: String indicate the correction type. Options are "affine", "homography", polynomial"
        and "lstsq"

    :return: The corrected latitude 2D array and the corrected longitude 2D array
    """
    # Use Utility File to Extract M
    M_dict = coordinate_correction_matrix(point_file, projection_metadata, correction_type)
    M_type = list(M_dict.keys())[0]
    M = M_dict[M_type]

    finalLat = originalLat.copy()
    finalLon = originalLon.copy()

    for i in range(originalLat.shape[0]):
        for j in range(originalLat.shape[1]):
            X = originalLon[i, j]
            Y = originalLat[i, j]

            modifiedLat = None
            modifiedLon = None

            # Second Degree Polynomial (Scikit)
            if M_type == "polynomial":
                # SK Image polynomial -----------------------------------------------

                lon_coeff = M.params[0]
                lat_coeff = M.params[1]

                deg = None
                if len(lon_coeff) == 10:
                    deg = 3
                elif len(lon_coeff) == 6:
                    deg = 2

                def apply_poly(X, Y, coeff, deg):
                    if deg == 2:
                        return coeff[0] + coeff[1] * X + coeff[2] * Y + coeff[3] * X ** 2 + coeff[4] * X * Y + coeff[
                            5] * Y ** 2
                    elif deg == 3:
                        return coeff[0] + coeff[1] * X + coeff[2] * Y + coeff[3] * X ** 2 + coeff[4] * X * Y + coeff[
                            5] * Y ** 2 + coeff[6] * X ** 3 + coeff[7] * (X ** 2) * (Y) + coeff[8] * (X) * (Y ** 2) + \
                            coeff[9] * Y ** 3

                modifiedLat = apply_poly(X, Y, lat_coeff, deg=deg)
                modifiedLon = apply_poly(X, Y, lon_coeff, deg=deg)

            elif M_type == "affine" or M_type == "homography":
                result = M.params @ np.array([[X], [Y], [1]]).ravel()
                modifiedLon = result[0]
                modifiedLat = result[1]

            # Np lin alg
            elif M_type == "lstsq":
                LonM = M[0]
                modifiedLon = LonM[0] * X + LonM[1]

                LatM = M[1]
                modifiedLat = LatM[0] * Y + LatM[1]

            finalLat[i, j] = modifiedLat
            finalLon[i, j] = modifiedLon

    return finalLat, finalLon

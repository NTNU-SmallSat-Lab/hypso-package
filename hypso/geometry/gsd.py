
import numpy as np
import pyproj as prj

from .bbox import compute_bbox

def compute_gsd(frame_count, 
                image_height, 
                latitudes, 
                longitudes,
                verbose=False) -> None:

    #frame_count = self.frame_count
    #image_height = self.image_height

    #latitudes = self.latitudes
    #longitudes = self.longitudes

    #try:
    #    bbox = self.bbox
    #except:
    #    self._compute_bbox()

    bbox = compute_bbox(latitudes=latitudes, longitudes=longitudes)

    aoi = prj.aoi.AreaOfInterest(west_lon_degree=bbox[0],
                                    south_lat_degree=bbox[1],
                                    east_lon_degree=bbox[2],
                                    north_lat_degree=bbox[3], 
                                )

    utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi)


    #bbox_geodetic = [np.min(latitudes), 
    #                 np.max(latitudes), 
    #                 np.min(longitudes), 
    #                 np.max(longitudes)]

    #utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84",
    #                                                area_of_interest=prj.aoi.AreaOfInterest(
    #                                                west_lon_degree=bbox_geodetic[2],
    #                                                south_lat_degree=bbox_geodetic[0],
    #                                                east_lon_degree=bbox_geodetic[3],
    #                                                north_lat_degree=bbox_geodetic[1], )
    #                                            )
    
    if verbose:
        print(f'[INFO] Using UTM map: ' + utm_crs_list[0].name, 'EPSG:', utm_crs_list[0].code)

    # crs_25832 = prj.CRS.from_epsg(25832) # UTM32N
    # crs_32717 = prj.CRS.from_epsg(32717) # UTM17S
    crs_4326 = prj.CRS.from_epsg(4326)  # Unprojected [(lat,lon), probably]
    source_crs = crs_4326
    destination_epsg = int(utm_crs_list[0].code)
    destination_crs = prj.CRS.from_epsg(destination_epsg)
    latlon_to_proj = prj.Transformer.from_crs(source_crs, destination_crs)


    pixel_coords_map = np.zeros([frame_count, image_height, 2])

    for i in range(frame_count):
        for j in range(image_height):
            pixel_coords_map[i, j, :] = latlon_to_proj.transform(latitudes[i, j], 
                                                                    longitudes[i, j])

    # time line x and y differences
    a = np.diff(pixel_coords_map[:, image_height // 2, 0])
    b = np.diff(pixel_coords_map[:, image_height // 2, 1])
    along_track_gsd = np.sqrt(a * a + b * b)
    #along_track_mean_gsd = np.mean(along_track_gsd)

    # detector line x and y differences
    a = np.diff(pixel_coords_map[frame_count // 2, :, 0])
    b = np.diff(pixel_coords_map[frame_count // 2, :, 1])
    across_track_gsd = np.sqrt(a * a + b * b)
    #across_track_mean_gsd = np.mean(across_track_gsd)


    #self.along_track_gsd = along_track_gsd
    #self.across_track_gsd = across_track_gsd

    #self.along_track_mean_gsd = along_track_mean_gsd
    #self.across_track_mean_gsd = across_track_mean_gsd

    return along_track_gsd, across_track_gsd







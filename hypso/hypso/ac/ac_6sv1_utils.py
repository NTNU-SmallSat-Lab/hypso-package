
import numpy as np

def get_lat_lon(satobj):

    try:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
    except Exception as ex:
        print(ex)
        print("[WARNING] 6SV1 attempting to use direct georeferencing.")
        latitudes = satobj.latitudes_direct
        longitudes = satobj.longitudes_direct

    return latitudes, longitudes
    
    
def get_image_extent_lat_lon(satobj):

    lat, lon = get_lat_lon(satobj)

    min_lat = np.nanmin(lat)
    max_lat = np.nanmax(lat)

    min_lon = np.nanmin(lon)
    max_lon = np.nanmax(lon)

    return min_lat, max_lat, min_lon, max_lon


def get_image_center_lat_lon(satobj, VERBOSE: bool = True):

    min_lat, max_lat, min_lon, max_lon = get_image_extent_lat_lon(satobj)

    if VERBOSE:
        print(f"[INFO] ROI:\nMax Lat: {max_lat}  Min Lat: {min_lat}\nMax Lon: {max_lon}  Min Lon: {min_lon}")

    ImageCenterLon = np.mean([min_lon, max_lon])
    ImageCenterLat = np.mean([min_lat, max_lat])

    return ImageCenterLat, ImageCenterLon

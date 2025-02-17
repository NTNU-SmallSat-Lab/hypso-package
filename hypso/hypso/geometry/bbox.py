
def compute_bbox(latitudes, 
                 longitudes) -> None:

    lon_min = longitudes.min()
    lon_max = longitudes.max()
    lat_min = latitudes.min()
    lat_max = latitudes.max()

    lon_min = float(lon_min)
    lon_max = float(lon_max)
    lat_min = float(lat_min)
    lat_max = float(lat_max)

    bbox = (lon_min,lat_min,lon_max,lat_max)
    
    return bbox




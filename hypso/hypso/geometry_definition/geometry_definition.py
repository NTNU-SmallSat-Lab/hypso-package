
from pyproj import Proj
from pyresample import geometry
from pyresample.geometry import AreaDefinition
from pyresample.geometry import SwathDefinition



def generate_area_def(area_id: str,
                      proj_id: str,
                      description: str,
                      bbox: tuple[float, float, float, float],
                      height: int = None,
                      width: int = None
                      ) -> AreaDefinition:

    lon_0 = (bbox[2] - bbox[0])/2 + bbox[0]
    lat_ts = (bbox[3] - bbox[1])/2 + bbox[1]

    p = Proj(proj="stere", ellps="bessel", lat_0=90.0, lon_0=lon_0, lat_ts=lat_ts)

    projection = {"proj": "stere", 
                "ellps": "bessel", 
                "lat_0": 90.0, 
                "lon_0": lon_0, 
                "lat_ts": lat_ts, 
                "units": "m"}

    lower_left_x, lower_left_y = p(bbox[0], bbox[1])
    upper_right_x, upper_right_y = p(bbox[2], bbox[3])
    area_extent = (lower_left_x, lower_left_y, upper_right_x, upper_right_y)


    if (height is None) or (width is None):

        res = 300 #meters

        width = (area_extent[2] - area_extent[0]) / res
        height = (area_extent[3] - area_extent[1]) / res


    area_def = geometry.AreaDefinition(area_id, proj_id, description, projection,  width, height, area_extent)

    return area_def


# TODO
def generate_swath_def() -> SwathDefinition:

    return None
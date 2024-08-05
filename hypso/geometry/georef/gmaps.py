#!/usr/bin/python
# GoogleMapDownloader.py
# Created by Hayden Eskriett [http://eskriett.com]
# Edited by Nima Farhadi
# Edited by Dennis Langer
#
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/

import PIL

from PIL.Image import Image
import math, shutil, requests, os
import io
from typing import Tuple
import numpy as np


def pixel_to_latlon(px_x, px_y, tl_x, tl_y, zoom_level) -> Tuple[float, float]:
    """
    Pixel to Lat Lon

    :param px_x: Pixel coordinates of google maps image (can be float)
    :param px_y: Pixel coordinates of google maps image (can be float)
    :param tl_x: tile indices of top left (noth-western) corner (must be int)
    :param tl_y: tile indices of top left (noth-western) corner (must be int)
    :param zoom_level: zoom level of the image (also must be int)

    :return: lat,lon
    """

    # background:
    # https://developers.google.com/maps/documentation/javascript/coordinates
    # https://en.wikipedia.org/wiki/Web_Mercator_projection

    tile_size = 256

    world_coord_x = (tile_size * tl_x + px_x) / (1 << int(zoom_level))
    world_coord_y = (tile_size * tl_y + px_y) / (1 << int(zoom_level))

    lon = (2.0 * math.pi * world_coord_x / 256.0 - math.pi) * 180.0 / math.pi
    lat = (2.0 * math.atan(
        math.e ** math.pi * math.e ** (- 2.0 * math.pi * world_coord_y / 256.0)) - math.pi / 2.0) * 180.0 / math.pi

    return lat, lon


class GoogleMapsLayers:
    """
    Google Maps Layers Class
    """
    ROADMAP = "v"
    TERRAIN = "p"
    ALTERED_ROADMAP = "r"
    SATELLITE = "s"
    TERRAIN_ONLY = "t"
    HYBRID = "y"


class GoogleMapDownloader:
    """
    A class which generates high resolution google maps images given a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12, layer=GoogleMapsLayers.SATELLITE):
        """
        GoogleMapDownloader Constructor

        :param lat: The latitude of the location required
        :param lng: The longitude of the location required
        :param zoom: The zoom level of the location required, ranges from 0 - 23. Defaults to 12
        :param layer: Default to GoogleMapsLayers.Satellite
        """

        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self._layer = layer

    def latlon_to_tileXY(self) -> Tuple[float, float]:
        """
        Generates an X,Y tile coordinate based on the latitude, longitude and zoom level

        :return: An X,Y tile coordinate
        """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
                tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs) -> Tuple[Image,tuple]:
        """
        Generates an image by stitching a number of google map tiles together.

        :param kwargs:
            center_tile_x:  The center tile x coordinate
            center_tile_y:  The center tile y coordinate
            tile_count_x:   The number of tiles wide the image should be
            tile_count_y:   The number of tiles high the image should be

        :return: A high-resolution Goole Map image, and the tile coordinates of the top left tile in the image.
        """

        center_tile_x = kwargs.get('center_tile_x', None)
        center_tile_y = kwargs.get('center_tile_y', None)
        tile_count_x = kwargs.get('tile_count_x', 16)
        tile_count_y = kwargs.get('tile_count_y', 16)

        # Check that we have x and y tile coordinates
        if center_tile_x == None or center_tile_y == None:
            center_tile_x, center_tile_y = self.latlon_to_tileXY()

        # Determine the size of the image
        width, height = 256 * tile_count_x, 256 * tile_count_y
        # Create a new image of the size require
        map_img = PIL.Image.new('RGB', (width, height))

        j = 1
        for x in range((-tile_count_x) // 2, tile_count_x // 2):
            for y in range((-tile_count_y) // 2, tile_count_y // 2):
                tile_coord_x = center_tile_x + x
                tile_coord_y = center_tile_y + y

                # https://mt0.google.com/vt?lyrs=s&x=280&y=54&z=9
                url = f'https://mt0.google.com/vt?lyrs={self._layer}&x={tile_coord_x}&y={tile_coord_y}&z={self._zoom}'

                if tile_coord_x >= 0 and tile_coord_y >= 0:
                    response = requests.get(url, stream=True)

                    data = response.raw.read()
                    if data[
                        0] == 60:  # 60 is ascii code for '<', meaning an html response was given, meaning no image data
                        continue

                    current_tile_pseudofile = io.BytesIO(data)
                    im = PIL.Image.open(current_tile_pseudofile, formats=('JPEG',))
                    map_img.paste(im,
                                  ((x + math.ceil(tile_count_x / 2)) * 256, (y + math.ceil(tile_count_y / 2)) * 256))
        tile_coords_corner = center_tile_x - tile_count_x // 2, center_tile_y - tile_count_y // 2
        return map_img, (tile_coords_corner)


def main() -> None:
    """
    Main entry point

    :return:
    """
    # usage example

    lat = 63.4163724
    lon = 10.404902
    zoom_level = 9

    # Create a new instance of GoogleMap Downloader
    gmd = GoogleMapDownloader(lat, lon, zoom_level, GoogleMapsLayers.SATELLITE)

    print(f"The center tile coorindates are {gmd.latlon_to_tileXY()}")
    print(f"Zoom level: {zoom_level}")

    tiles_eastwest = 6
    tiles_northsouth = 10

    try:
        # Get the high resolution image
        print('Downloading google maps ...')
        img, tile_coords_corner = gmd.generateImage(tile_count_x=tiles_eastwest, tile_count_y=tiles_northsouth)
    except IOError:
        print("Could not generate the image - try adjusting the zoom level and checking your coordinates")
    else:
        print('Image size (pix): ', img.size)
        print(f'Tile coordinate top left (north-west) corner: {tile_coords_corner[0]},{tile_coords_corner[1]}')
        # Save the image to disk
        img.save(f"google_maps_{tile_coords_corner[0]}_{tile_coords_corner[1]}_zl{zoom_level}.png")
        print("The map has successfully been created")


if __name__ == '__main__':
    main()

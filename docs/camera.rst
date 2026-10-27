RGB camera
==========

HYPSO-2 carries an RGB camera next to the hyperspectral imager. Each capture
directory holds its image as ``<capture id>-rgb/<target>_<time>.bmp``: one 8-bit
raw RGGB Bayer mosaic, 3840 x 2748 pixels (about 30 m), taken about 2 s before
the HSI capture. It has no georeferencing of its own.

Reading
-------

.. code-block:: python

   from hypso.rgb import find_camera_image, read_camera_image

   camera = read_camera_image(find_camera_image(capture_dir))   # (y, x, channel) R, G, B

The mosaic is demosaiced (bilinear, RGGB) and rotated 180 degrees, as the
ground processing's ``rgb-camera-image-debayered.png``. Reading never
registers the image.

Registration
------------

Registration is an explicit step that takes the geolocation from the HSI
capture:

.. code-block:: python

   from hypso.rgb import register_camera, georeference_camera

   l1d = hypso.open(l1d_path, wavelengths=[640, 550, 460])   # three bands suffice
   reg = register_camera(camera, l1d)
   reg.ok, reg.inliers, reg.rms_m, reg.coverage
   geo = georeference_camera(camera, reg, l1d)      # R, G, B, latitude, longitude, in_hsi_swath

How it works: the HSI bands near red, green and blue are regridded onto a UTM
map (150 m); SIFT keypoints are matched between that map and the camera image;
RANSAC fits a homography from camera pixels to map coordinates (a frame camera
looking at nearly flat ground maps onto the ground plane by a homography). The
fit must have at least 30 inliers, cover at least a quarter of the camera
image (a fit anchored in one corner is unreliable elsewhere) and have a
residual RMS of at most 500 m; otherwise ``reg.ok`` is false with the reason
and the image is not georeferenced. On the test captures the residual is about
190-220 m, well below the HSI's 876 x 108 m pixels, in about 40 s.

``hsi_to_camera_pixels`` and ``camera_to_hsi_pixels`` map between the two
images in their own pixel coordinates. Camera pixels outside the HSI swath are
georeferenced by extrapolating the fit (``in_hsi_swath`` is 0 there).

Maps
----

.. code-block:: python

   from hypso.rgb import regrid_camera, plot_rgb_map

   on_hsi_grid = regrid_camera(geo, l1d.regrid(resolution=300).area_definition())
   own_grid = regrid_camera(geo)                     # UTM, the camera's ~30 m
   plot_rgb_map(own_grid, graticule=True, scale_bar=True, north_arrow=True, coastlines=True)

``regrid_camera`` averages the camera pixels in each cell when cells are at
least twice the camera's pixel size and takes the nearest pixel otherwise.
``plot_rgb_map`` draws any gridded RGB image (a regridded camera image or a
regridded hypso.Product) with optional latitude/longitude graticule, scale bar,
north arrow, coastlines (needs cartopy and its Natural Earth data) and title.
Both need matplotlib, which the package does not depend on.

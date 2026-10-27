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

Camera products
---------------

The camera is not a ``hypso.Product`` (no wavelengths, calibration levels or
SRFs); a camera product is an xarray Dataset in the same CF/ACDD conventions:

.. code-block:: python

   from hypso.rgb import camera_product, write_camera, open_camera

   camera_l1a = camera_product(camera, reg, l1d)    # reg may be None or failed
   path = write_camera(camera_l1a, out_dir=".")     # HYPSO2_RGB.20250404T022723.ariake.L1A.nc
   camera_l1a = open_camera(path)

``R``, ``G`` and ``B`` are the demosaiced counts in the camera's own frame,
stored as uint8 (the bilinear demosaic rounded). ``L1A`` means uncalibrated
counts. A registered product adds float32 ``latitude``/``longitude`` per pixel,
``in_hsi_swath`` (when the HSI product is given) and the registration
attributes; an unregistered one records why, without geolocation
(``georeferenced = "false"``). The file name uses the ``RGB`` instrument token
(:doc:`products`); a :func:`~hypso.rgb.regrid_camera` result gets the grid
token too (``...L1A.utm52n_100m.nc``). About 22 MB for a registered
3840 x 2748 image.

Satpy
-----

The ``hypso_rgb_l1a`` reader matches ``_RGB`` names only (the HSI readers match
``_HSI``) and has a ``true_color`` composite from R, G and B:

.. code-block:: python

   from satpy import Scene

   scene = Scene(reader="hypso_rgb_l1a", filenames=[str(path)])
   scene.load(["true_color"])

A native-frame product is a swath (per-pixel latitude/longitude), a regridded
one is read on its map grid. Unregistered products cannot be placed by Satpy.

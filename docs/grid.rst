Map grids
=========

Products from L1C on are in sensor (swath) geometry: ``(lines, samples)``
pixels, each with its own latitude and longitude. ``hypso.grid`` puts them on
map grids.

Regridding
----------

``Product.regrid()`` returns the same product, at the same level, on a map
grid:

.. code-block:: python

   import hypso

   l1d = hypso.open("HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc")
   g = l1d.regrid()                      # capture's UTM zone, its own resolution
   g.grid                                # "utm33n_876m"
   g.write()                             # ....L1D.utm33n_876m.nc

   l1d.regrid(resolution=300)            # UTM, 300 m
   l1d.regrid("EPSG:4326", resolution=0.005)
   other.regrid(g)                       # onto g's grid: pixel-by-pixel comparable
   l1d.regrid(area_def)                  # any pyresample AreaDefinition

Grids are snapped to whole multiples of the pixel size, so two captures
regridded to the same CRS and resolution share pixel boundaries.

``target``
   ``"utm"`` (default; the WGS84 UTM zone of the scene centre), any CRS
   pyproj accepts, a pyresample ``AreaDefinition``, or a gridded ``Product``.
``resolution``
   Pixel size in the CRS's units. The default is the product's resolution
   (the coarser of its along- and across-track pixel spacing), so nothing is
   oversampled. HYPSO pixels are far from square (the wide capture used in
   the tests: about 876 m along-track, 108 m across-track), so a finer grid
   is a deliberate choice. ``max_gb`` (default 4) refuses grids whose cube
   would be larger.
``method``
   ``"nearest"`` (default): every grid pixel holds one measured spectrum,
   unmixed. ``"bilinear"`` blends neighbouring pixels' spectra, on request.

Per variable: the product cube and float variables use ``method``; azimuth
angles too, but bilinear works on their sine and cosine to avoid 0/360
artefacts; flags and integer variables are always nearest neighbour.
``pixel_flags`` gains an ``outside_swath`` flag for grid pixels the capture
does not cover (their spectra are NaN). ``*_direct`` geolocation and angles
are dropped. The method, CRS, resolution and coverage are recorded in
``metadata/processing/regrid``.

Any level with geolocation can be regridded: L1C, L1D and L2A. Per-pixel
steps still work on a gridded product (a gridded L1C can be turned into L1D
with ``toa_reflectance()``). ``correct_atmosphere()`` cannot: the AC tools
read sensor geometry, so correct first, then regrid.

File layout
-----------

A gridded product has dimensions ``(y, x)``, 1D ``x``/``y`` projection
coordinates, and a CF grid-mapping variable ``crs`` (with ``crs_wkt``) that
every band and variable references through ``grid_mapping``. ESA SNAP and GIS
tools read it as a map-projected raster. The 2D ``latitude`` and
``longitude`` stay, as auxiliary coordinates. The grid's name is a field of
the file name, between the AC method and the label.

Resampling to anything else
---------------------------

``Product.resample(target, method="nearest")`` returns a bare
``xr.Dataset`` on any pyresample target, including a ``SwathDefinition``
such as another sensor's swath for matchups. It holds the cube, the derived
variables, each flag as a boolean ``flag_<name>``, and the target's latitude
and longitude.

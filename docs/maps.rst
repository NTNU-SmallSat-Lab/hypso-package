Footprints and maps
===================

``hypso.maps`` makes a capture's ground footprint (as a polygon, a shapefile or
GeoJSON), RGB images and maps of any product variable. It needs shapely for
footprints, pyshp for shapefiles, matplotlib for figures and cartopy (with its
Natural Earth data) for coastlines; none is a dependency of the package.

Footprints
----------

.. code-block:: python

   import hypso
   from hypso import maps

   l1d = hypso.open("HYPSO2_HSI.20261006T022612.koreanCalValUB03.L1D.nc", wavelengths=[460, 550, 640])
   l1d.footprint()                          # shapely Polygon, WGS84 lon/lat: the swath outline
   l1d.footprint(mask="data")               # only the pixels with data (blocks of 8 x 8)
   l1d.write_footprint("footprint.shp")     # .shp .shx .dbf .prj .cpg, EPSG:4326
   maps.write_footprints([a, b, c], "all.geojson")   # one feature per capture

The swath outline traces the outer ring of the pixel grid (a vertex every 10
pixels). Across the antimeridian, longitudes are given in 0-360. Each feature
carries the capture's metadata (field names of at most 10 characters for the
shapefile format):

==========  ========================================================
capture     ``<target>_<YYYY-MM-DDTHH-MM-SSZ>``, as on the HYPSO servers
satellite   HYPSO-1 or HYPSO-2
target      the capture's target name
start_utc   acquisition start
level       L1C, L1D, L2A, L2B
product     e.g. ``L1D``, ``L2A.RRS.ocsmart``, ``L2B.BGC.ocsmart``
georef      ``indirect`` (GCP-corrected), ``direct`` or ``unknown``
calib       radiometric calibration file
ac_method   atmospheric correction (L2)
lines       swath size
samples
area_km2    footprint area (geodesic)
centre_lat  mean pixel position
centre_lon
source      file name, when read from a file
==========  ========================================================

RGB images
----------

``maps.save_rgb(product, "rgb.png")`` writes an RGB PNG (bands nearest 640,
550 and 460 nm) in sensor geometry, transparent where there is no data, in
the HYPSO servers' quicklook orientation (``orientation="product"`` for the
arrays as stored; ``bin_samples=3`` like the servers' ``-bin3`` images).
L1 products are stretched per channel (2-98 percentiles, one gain for all
channels renders TOA land pink) with a display gamma of 2.2. L2A surface
reflectance (Rrs, rhos) gets one linear gain for all channels, to the 99.5th
percentile: true colour, deep blue for clear ocean and green for turbid lakes
(a gamma or a per-channel stretch washes it out to flat cyan).

Maps
----

.. code-block:: python

   maps.plot_map(l1d)                                   # RGB
   maps.plot_map(hypso.open(rrs_file, wavelengths=[560]), "Rrs", 560)
   maps.plot_map(hypso.open(bgc_file), "chlor_a_oci")   # log scale for concentrations and IOPs
   maps.plot_overview([p1, p2, p3])                     # footprints of several captures

``plot_map`` reduces the product to what it plots, regrids it
(``Product.regrid``, nearest neighbour, UTM, the product's resolution) and
draws it over light-grey land and light-blue sea and lakes (Natural Earth 10 m
ocean and lakes; ``water=False`` to switch off), with the footprint outline,
coastlines and lake shores, a latitude/longitude graticule, scale bar, north
arrow and colour bar. ``plot_overview`` draws footprint outlines in distinct
colours with the legend beside the map.

For processed capture folders, ``demo/make_maps.py`` writes all of these at
once::

   demo/make_maps.py ~/HYPSO_DATA/koreanCalVal* --overview ~/HYPSO_DATA/overview --group korea=koreanCalVal

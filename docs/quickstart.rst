Quick start
===========

A runnable version of this page, with figures, is the demo in the
repository: ``demo/demo_processing.py`` (or ``demo_processing.ipynb``).

Open and process
----------------

``hypso.open`` reads a product file of any level, in the current or an older
file layout. Every processing method returns a new, immutable
:class:`~hypso.api.Product`; the one it was called on is unchanged.

.. code-block:: python

   import hypso

   l1a = hypso.open("HYPSO2_HSI.20250304T103805.aeronetvenice.L1A.nc")
   l1b = l1a.calibrate()                    # radiometric "abs", V3 coefficients
   l1c = l1b.georeference(latitudes="latitudes_indirectgeoref.dat",
                          longitudes="longitudes_indirectgeoref.dat")
   l1d = l1c.toa_reflectance()

   # or: run only the missing steps
   l1d = l1a.to("L1D", georeference={"latitudes": "latitudes_indirectgeoref.dat",
                                     "longitudes": "longitudes_indirectgeoref.dat"})

``to("L2A")`` continues through atmospheric correction with MACHI, the default
method; ``to("L2A", correct_atmosphere={"method": "polymer", "config": "ac_tools.yaml"})``
picks another.

Calibration and the other steps log what they do through Python's
``logging``; ``logging.basicConfig(level=logging.INFO)`` shows it.

Write
-----

.. code-block:: python

   l1d.write()                         # ./HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc
   l1d.write(out_dir="out", label="test")   # ...L1D.test.nc, for experiments
   l1d.provenance                      # how it was made, one entry per step

Look inside
-----------

.. code-block:: python

   l1d.cube                  # xarray DataArray (lines, samples, bands)
   l1d.wavelengths           # band centres, nm
   l1d.geometry              # latitude, longitude, angles
   l1d.flags                 # {name: boolean mask} from pixel_flags
   l1d.spectrum(45.3139, 12.5083)            # nearest pixel's spectrum
   l1d.window(45.3139, 12.5083, size=5)      # 5x5 window and its statistics
   l1d.tree                  # the whole product as an xarray DataTree

Masks and derived products
--------------------------

.. code-block:: python

   from hypso.masks import sea_land_cloud_masks

   # the ground processing's classification, processing-temp/sea-land-cloud.labels
   slc = sea_land_cloud_masks("sea-land-cloud.labels", l1d.cube.shape[:2])
   masked = l1d.with_masks(land=slc["land"], cloud=slc["cloud"])   # or any boolean arrays
   with_chl = masked.with_variables(chla=chla_array)      # written alongside the cube

Atmospheric correction
----------------------

.. code-block:: python

   l2a = l1d.correct_atmosphere()                    # MACHI, the default
   l2a = l1d.correct_atmosphere("dps")
   l2a = l1d.correct_atmosphere("polymer", config="ac_tools.yaml", masked=True)
   l2a.variables             # the tool's own 2D products and flags

Map grids
---------

.. code-block:: python

   g = l1d.regrid()                                  # capture's UTM zone, nearest neighbour
   coarse = l1d.regrid(resolution=3000, method="average")
   other.regrid(g)                                   # onto the same grid

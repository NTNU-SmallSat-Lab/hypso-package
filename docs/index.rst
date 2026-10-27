hypso
=====

``hypso`` processes images from the HYPSO-1 and HYPSO-2 hyperspectral
satellites (NTNU SmallSat Lab), from raw counts to atmospherically corrected
reflectance, and reads and writes them as CF-compliant NetCDF files that open
in ESA SNAP.

.. code-block:: python

   import hypso

   l1d = (hypso.open("HYPSO2_HSI.20250304T103805.aeronetvenice.L1A.nc")
              .calibrate()
              .georeference(latitudes="latitudes_indirectgeoref.dat",
                            longitudes="longitudes_indirectgeoref.dat")
              .toa_reflectance())
   l1d.write()                          # HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc

   l2a = l1d.correct_atmosphere()        # MACHI; or "polymer", config="ac_tools.yaml"
   l2a.window(45.3139, 12.5083, size=3)  # matchup window at AAOT

=========  ===============================================  =================
Level      Contents                                         Made by
=========  ===============================================  =================
L1A        raw counts (``dn``)                              ground station
L1B        top-of-atmosphere radiance (``Lt``)              ``calibrate()``
L1C        L1B with latitude, longitude and angles          ``georeference()``
L1D        top-of-atmosphere reflectance (``rhot``)         ``toa_reflectance()``
L2A        atmospherically corrected reflectance            ``correct_atmosphere()``
=========  ===============================================  =================

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: User guide

   products
   calibration
   coefficient_history
   atmospheric_correction
   grid
   windows
   camera

.. toctree::
   :maxdepth: 1
   :caption: Development

   architecture
   versioning
   documentation

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/hypso/index

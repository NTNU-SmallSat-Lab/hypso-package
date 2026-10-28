Architecture
============

This page explains how the package is organised and why. Exact signatures are
in the API reference generated from the docstrings. Usage is covered in
:doc:`calibration`, :doc:`atmospheric_correction` and :doc:`versioning`.

Overview
--------

A product is one capture at one processing level, from L1A (raw counts) to
L2A (atmospherically corrected). In memory it is an ``xarray.DataTree``; on
disk it is one NetCDF file. Processing steps are functions that take one
product and return the next level's product.

.. code-block:: text

   hypso.Product  (hypso/api.py)                      what users call
        |
   hypso.processing                                   one function per step
     l1a_to_l1b   calibrate.py      radiometric, smile, destriping, spectral
     l1b_to_l1c   georeference.py   coordinates and viewing/solar angles
     l1c_to_l1d   toa.py            TOA reflectance, metadata/srf
     correct_atmosphere  atmospheric.py   MACHI (default), DPS, Polymer, ACOLITE, OC-SMART
     select_bands bands.py          band subsets (from L1B on)
   hypso.grid                                         regrid / resample onto map grids
   hypso.window                                       pixel windows for matchups
        |
   hypso.io                                           the file format
     product.py      read_product / write_product (every level, every layout)
     naming.py       file names
     provenance.py   metadata/processing and history
     cf.py, schema.py  CF attributes, per-level titles and units
        |
   building blocks
     hypso.calibration (incl. the optional keystone correction), hypso.geometry,
     hypso.srf, hypso.masks (flags, sea/land/cloud labels),
     hypso.ac (MACHI, tool drivers, output readers, configuration),
     hypso.rgb (HSI2RGB rendering, RGB camera images), hypso.sensors,
     hypso1_calibration / hypso2_calibration (coefficient files and catalogs)

Earlier versions centred on one large capture object (``HypsoCapture``) that
held about a hundred attributes, ran every step in place, and was copied
between per-level classes. It has been removed. Its responsibilities are now
split as shown above, and its state is the product tree.

The product tree
----------------

``hypso.io.product`` defines the in-memory model that every processing
function takes and returns:

.. code-block:: text

   /                          root Dataset
       <prefix>               product cube (lines, samples, bands), or
                              (y, x, bands) when regridded: dn (L1A), Lt (L1B/L1C),
                              rhot (L1D), the method's variable at L2A
                              (rhos or Rrs; rho_w in older POLYMER files)
       wavelength             band coordinate: nominal centre, 0.1 nm
       radiation_wavelength   band coordinate: calibrated centre
       fwhm                   band coordinate
       band_index             band coordinate: index in the full band set
                              (after select_bands, and on products read from files)
       x, y, crs              regridded products: projection coordinates, grid mapping
       latitude, longitude, crs_wgs84, <angles>, <angles>_direct
                              from L1C on
       pixel_flags            optional: masks as CF flags
       <other>                optional: derived (lines, samples) products
       attrs                  global attributes; processing_level is required
   /metadata/<group>          capture_config, timing, adcs, corrections,
                              gcp, srf, processing (and machi after MACHI)

The product is one 3D array in memory, which makes band arithmetic and
``.sel()`` straightforward. It is split into one variable per band only when
written (below).

A processing function never changes its input tree. It copies the tree,
replaces what its step changes, and adds a ``metadata/processing/<step>``
node. ``tree.encoding["source"]`` records the file a tree was read from; it
is not written, and steps use it as their recorded input file.

``hypso.Product``
-----------------

``hypso.open(path)`` reads any product file and returns a ``Product``: an
immutable wrapper around one tree.

.. code-block:: python

   import hypso

   l1d = (hypso.open("…L1A.nc")
             .calibrate()                                    # abs / V3
             .georeference(latitudes=lat_file, longitudes=lon_file)
             .toa_reflectance())
   l2a = l1d.correct_atmosphere()                    # MACHI; or "polymer", config=...
   l2a.write()

Every method returns a new ``Product``; setting an attribute raises. That
makes it safe to branch from one product (two AC methods from the same L1D)
and lets pipelines treat each stage as file in, function, file out. Calling a
step on the wrong level raises an error that says which level it needs and
suggests ``.to(level)``, which runs only the missing steps.

Besides the processing steps (and ``.to(level)``, up to L2A), ``Product``
offers ``.cube``, ``.wavelengths``, ``.geometry``, ``.flags``, ``.variables``,
``.provenance``, ``.spectrum()``, ``.window()``, ``.pixel()``,
``.select_bands()``, ``.resolution``, ``.swath_definition()``, ``.resample()``,
``.regrid()``, ``.area_definition()``, ``.grid``, ``.to_satpy()``,
``.with_masks()``, ``.with_variables()``, ``.filename()`` and ``.write()``.
``.tree`` returns a shallow copy of the underlying tree for anything else
(arrays are shared: replace them, never modify them in place).

The processing functions in ``hypso.processing`` stay public for callers that
prefer working on trees directly.

Processing steps
----------------

========================  =================  =========================================
Function                  Levels             Records in ``metadata/processing``
========================  =================  =========================================
``l1a_to_l1b``            L1A -> L1B         coefficient release, per-correction
                                             versions, files and SHA-256, corrections
                                             applied (keystone only on request),
                                             capture type
``l1b_to_l1c``            L1B -> L1C         indirect or direct georeferencing,
                                             coordinate files
``l1c_to_l1d``            L1C -> L1D         solar irradiance source, SRF grid,
                                             Earth-Sun distance
``correct_atmosphere``    L1C or L1D -> L2A  tool, version, installation commit,
                                             effective and user settings
``select_bands``          same level         selection, bands kept
``hypso.grid.regrid``     same level         grid, CRS, method, coverage
========================  =================  =========================================

The science code these steps call (``hypso.calibration``, ``hypso.geometry``,
``hypso.srf``) is unchanged from the capture-object version. Each step
was checked to give identical output to the old path before that path was
removed, and the whole chain reproduces a golden baseline recorded from the
original package (``tests/test_regression_real_capture.py``).

``hypso.processing.capture_params(tree)`` derives what the steps need from
``metadata/capture_config`` (crop window, bin factor, background, exposure,
image size, capture type, acquisition time), the same way the old loader did.

File format
-----------

``write_product`` is the only writer and ``read_product`` the only reader; the
format itself is described for users in :doc:`products`. Design choices
behind it: product and geometry variables sit at the root group because CF's
``coordinates`` and ``grid_mapping`` references resolve only upwards through
parent groups, never sideways between siblings (with the old ``products`` and
``geometry`` sibling groups a band's ``coordinates`` could never resolve), and
because SNAP's NetCDF-CF reader dates from NetCDF 3 without groups. One 2D
variable per band is the layout SNAP's spectrum view expects. All CF
attributes come from ``hypso.io.cf``, so a file read in any supported layout
is written back in the current one. Float bands keep 14 significant mantissa
bits by default (netCDF BitRound, at most 3.1e-5 relative error, far below the
sensor noise; about 40% smaller L1D files); BitRound rather than
GranularBitRound because the latter turns NaN into -0.0 in netCDF-C 4.9.3.

Sensors and calibration
-----------------------

Sensor profiles (``hypso.sensors``)
   Instrument data (FWHM lookup table, imaging modes, how to find calibration
   files) is in a ``SensorProfile``, registered at import time in
   ``hypso/sensors/hypso1.py`` and ``hypso2.py``. Imaging modes (nominal, wide,
   moon, ...) and how each is recognised are YAML (``hypso1_modes.yaml``,
   ``hypso2_modes.yaml``). Adding a sensor means adding one profile module and
   calling ``register_sensor``.

Coefficient catalogs
   Each calibration package lists its coefficient files, versions and releases
   in ``data/coefficients.yaml`` (HYPSO-2: V3, V2, V1; HYPSO-1: V2, V1), read by
   ``_catalog.py``, identical in both packages. ``coeff_type`` picks the
   radiometric variant, and ``version`` picks the release or sets versions per
   correction. A file entry can name a different file for one capture type
   (V3 ``abs`` uses a pre-binned wide matrix for wide captures), and
   ``crop: as_is`` marks a file that must not be cropped and binned. Models
   outside the releases (HYPSO-2's keystone model, shipped disabled for
   review) are under ``optional``. See :doc:`calibration`.

Your own coefficient files
   Without editing the calibration packages, either register a named set once
   per process:

   .. code-block:: python

      from hypso.calibration import register_calibration_coeffs

      register_calibration_coeffs(sat_id="HYPSO-1", name="my_lab_calibration",
                                  files={"radiometric": "/path/to/radiometric.npz",
                                         "spectral": "/path/to/spectral.npz"})
      l1b = l1a.calibrate(coeff_type="my_lab_calibration")

   or pass the files for one run with ``l1a.calibrate(coeff_files={...})``.
   Corrections without a file are skipped. Explicit files win over a registered
   name, and a registered name wins over the sensor's built-in sets.

Each calibration run logs a summary through Python's ``logging`` module, and
records the same details in the product.

Spectral response and TOA reflectance
-------------------------------------

``hypso.srf`` holds the spectral response of the bands. ``gaussian_srfs`` builds
one Gaussian SRF per unbinned band on the solar spectrum's wavelength grid
(TSIS-1 HSRS, or Thuillier 2002, from ``srf/solar.py``); ``bin_srfs`` bins them
like the image; ``band_irradiance`` gives each band's solar irradiance F0
and ``effective_fwhm`` its width. ``compute_spectral_response``
returns all of it as one ``SpectralResponse``, and ``srf_dataset`` turns that
into a product's ``metadata/srf`` group: ``F0``, effective FWHM,
``earth_sun_distance_au``, and one ``Band_<i>`` variable per band (``i`` its
``band_index``) on its own ``wav_Band_<i>`` grid. Polymer reads its SRFs from this group (its
``srf_getter`` is ``hypso.ac.ac_polymer_srf_getter_from_l1``), and readers that
convert radiance themselves can take F0 and the Earth-Sun distance from it.

``hypso.processing.toa`` is the L1C -> L1D step: ``toa_reflectance`` computes
``pi L d^2 / (F0 cos(sza))``, with the Earth-Sun distance ``d`` from
``hypso.srf.earth_sun_distance``; ``spectral_response(tree)`` and
``srf_node(tree)`` build the above for a product (atmospheric correction uses
``srf_node`` to give Polymer SRFs for an L1C input).

Atmospheric correction
----------------------

``hypso.processing.correct_atmosphere`` runs MACHI (the default; vendored in
``hypso.ac.machi``) and DPS in-process, and Polymer, ACOLITE and OC-SMART as
separate processes:

- Polymer and ACOLITE run through ``python -m
  hypso.ac.adapters._polymer_driver`` / ``_acolite_driver`` started by
  ``run_subprocess_driver``. Polymer's v1 and v2 builds ship incompatible
  versions of the same top-level packages, and Python caches whichever it
  imports first for the rest of the process, so each run gets a fresh process.
  This also contains crashes in the tools' native code. Settings go in a JSON
  file; credentials only through environment variables.
- OC-SMART has no Python API. It runs as ``OCSMART.py`` in its installation
  directory and reads its paths from a shared ``OCSMART_Input.txt`` there,
  which is written and restored under a file lock shared with
  hypso-processing-pipeline.

Where the tools are installed comes only from configuration
(``hypso.ac.config``, a YAML file). Their outputs are read by
``hypso/ac/*_output.py`` and their bands mapped onto the input product's
bands. See :doc:`atmospheric_correction`.

Satpy
-----

``Product.to_satpy()`` builds a ``satpy.Scene`` from a product. Separately,
``hypso_l1c``, ``hypso_l1d`` and ``hypso_l2a`` are registered Satpy reader plugins, so files
open through Satpy's own interface without this package's API::

    from satpy import Scene

    scene = Scene(reader="hypso_l1c", filenames=["HYPSO2_HSI.….L1C.nc"])
    scene.load(["Lt_550", "latitude", "longitude"])

The readers share ``HypsoL1FileHandler`` (``hypso/satpy/hypso_handler.py``).
Bands are discovered per file with ``hypso.io.product.list_band_variables``,
because the band set depends on binning and calibration (and, for L2A, on
the quantity: ``Rrs_<nm>`` or ``rhos_<nm>``; ``rho_w_<nm>`` in older POLYMER files). Regridded files
are read on their map grid (``get_area_def``), others as swaths. The readers match
current and older file names. They are registered through a ``satpy.readers``
entry point, so the package must be installed (an editable install is
enough).

Resampling and regridding
-------------------------

``hypso.grid`` holds everything spatial beyond the product's own
geolocation: swath and grid definitions, ``resample_product`` and
``regrid``. ``Product.resample(target)`` returns a bare Dataset on any
pyresample target, including another sensor's swath; ``Product.regrid()``
returns a product of the same level on a map grid. See :doc:`grid`.

Extending the package
---------------------

A new sensor
   A profile module in ``hypso/sensors`` plus its modes YAML; a calibration
   package with its files.

New calibration coefficients
   Files in the calibration package's ``data/`` and entries in its catalog;
   then update the checksum manifest and tag a release (:doc:`versioning`).

A new atmospheric-correction method
   A function in ``processing/atmospheric.py`` that runs the method, maps its
   bands onto the input and returns ``_l2a_tree(...)``, plus its entry in
   ``_METHODS`` and ``_INPUT_LEVELS``: ``_<method>(tree, masked, mask_flags,
   **options)`` for an in-process method, with ``work_dir`` and ``config``
   added (and an entry in ``_EXTERNAL``) for an external tool, which runs in a
   subprocess.

A new product level
   A ``LevelSchema`` in ``io/schema.py`` (title, product prefix and units,
   whether it has geometry), a processing function that returns a tree with
   that ``processing_level``, and a ``Product`` method calling it.

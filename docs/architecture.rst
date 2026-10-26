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
     correct_atmosphere  atmospheric.py   DPS, Polymer, ACOLITE, OC-SMART
        |
   hypso.io                                           the file format
     product.py      read_product / write_product (every level, every layout)
     naming.py       file names
     provenance.py   metadata/processing and history
     cf.py, schema.py  CF attributes, per-level titles and units
        |
   building blocks
     hypso.calibration, hypso.geometry, hypso.srf, hypso.masks,
     hypso.ac (tool drivers, output readers, configuration),
     hypso.sensors, hypso1_calibration / hypso2_calibration

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
       <prefix>               product cube (lines, samples, bands):
                              dn (L1A), Lt (L1B/L1C), rhot (L1D),
                              the method's variable at L2A (Rrs, rho_w, rhos)
       wavelength             band coordinate: nominal centre, 0.1 nm
       radiation_wavelength   band coordinate: calibrated centre
       fwhm                   band coordinate
       latitude, longitude, crs_wgs84, <angles>, <angles>_direct
                              from L1C on
       pixel_flags            optional: masks as CF flags
       <other>                optional: derived (lines, samples) products
       attrs                  global attributes; processing_level is required
   /metadata/<group>          capture_config, timing, adcs, corrections,
                              gcp, srf, processing

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
   l2a = l1d.correct_atmosphere("polymer", config="ac_tools.yaml")
   l2a.write()

Every method returns a new ``Product``; setting an attribute raises. That
makes it safe to branch from one product (two AC methods from the same L1D)
and lets pipelines treat each stage as file in, function, file out. Calling a
step on the wrong level raises an error that says which level it needs and
suggests ``.to(level)``, which runs only the missing steps.

Besides the processing steps, ``Product`` offers ``.cube``, ``.geometry``,
``.flags``, ``.variables``, ``.provenance``, ``.spectrum()``, ``.pixel()``,
``.resolution``, ``.swath_definition()``, ``.resample()``, ``.to_satpy()``,
``.with_masks()``, ``.with_variables()``, ``.filename()`` and ``.write()``.
``.tree`` returns a copy of the underlying tree for anything else.

The processing functions in ``hypso.processing`` stay public for callers that
prefer working on trees directly.

Processing steps
----------------

========================  =================  =========================================
Function                  Levels             Records in ``metadata/processing``
========================  =================  =========================================
``l1a_to_l1b``            L1A -> L1B         coefficient release, per-correction
                                             versions, files and SHA-256, corrections
                                             applied, capture type
``l1b_to_l1c``            L1B -> L1C         indirect or direct georeferencing,
                                             coordinate files
``l1c_to_l1d``            L1C -> L1D         solar irradiance source, SRF grid,
                                             Earth-Sun distance
``correct_atmosphere``    L1C or L1D -> L2A  tool, version, installation commit,
                                             effective and user settings
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

``write_product`` is the only writer. ``read_product`` reads current files,
older grouped files (``/products`` and ``/geometry`` groups, written by the
pre-refactor package) and ground-station L1A files (one 3D ``/products/dn``
array). Reading an old file and writing it again converts it;
``convert_product(src)`` does that and also gives the file its current name.

Layout
   Product and geometry variables are at the **root** group; only
   ``metadata/*`` is nested. CF's ``coordinates`` and ``grid_mapping``
   references resolve by walking *up* to parent groups, never sideways between
   sibling groups, so with ``products`` and ``geometry`` as siblings (the old
   layout) a band's ``coordinates`` could never resolve. The flat layout also
   suits ESA SNAP's NetCDF-CF reader, which dates back to NetCDF 3 and its lack
   of groups. Metadata groups define the dimensions they use themselves.

Bands
   One 2D variable per band (``Lt_378``, ``rhot_378``, ...), the layout SNAP's
   spectrum view expects. Each band carries ``wavelength``,
   ``radiation_wavelength`` with ``radiation_wavelength_unit`` (what SNAP reads
   as the band's wavelength), ``fwhm``, ``band`` (its index) and ``parameter``
   (its own name). ``datacube=True`` writes a single 3D variable instead.

Masks
   Products are written unmasked. Land, cloud and custom masks are bits of one
   ``pixel_flags`` variable with CF ``flag_masks`` and ``flag_meanings``, which
   SNAP shows in its Mask Manager. ``hypso.masks.flags`` encodes and decodes
   them; atmospheric correction applies them to its input when asked
   (``masked=True``) instead of reading a separately written masked copy.

Encoding
   zlib level 4 with shuffle, each array in one chunk, no ``_FillValue``.
   L1A counts stay ``uint16``.

Global attributes
   CF and ACDD (``Conventions = "CF-1.10, ACDD-1.3"``), ``title``,
   ``processing_level``, ``product_format_version`` (``"2.0"``; files without
   it are format 1), ``date_created``, ``geospatial_lat_min`` and the other
   ACDD bounds, and ``history`` with one line per processing step.

Provenance
   ``hypso.io.provenance.add_step`` adds a ``metadata/processing/<step>`` group
   for each step. Groups are kept when a product is read and written again, so
   a file records its whole lineage. Software versions come from
   ``hypso.versioning`` (:doc:`versioning`).

File names
   ``{SAT}_HSI.{YYYYMMDDTHHMMSS}.{target}.{LEVEL}[.{method}][.{label}].nc``,
   for example ``HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.polymer.nc``
   (``hypso.io.naming``). The satellite comes first, so a listing groups by
   satellite and then sorts by time. ``.`` separates fields only, so fields can
   contain ``_`` and ``-``. The calibration set and masking are not in the name;
   they are recorded in the file. The optional label is for experiments. Older
   names (``{target}_{time}-{label}-{level}.nc``) still parse.

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
   HYPSO-2's coefficient files, versions (V3, V2, V1) and releases are listed in
   ``hypso2_calibration/data/coefficients.yaml``. ``coeff_type`` picks the
   radiometric variant, and ``version`` picks the release or sets versions per
   correction. A file entry can name a different file for one capture type
   (V3 ``abs`` uses a pre-binned wide matrix for wide captures), and
   ``crop: as_is`` marks a file that must not be cropped and binned. See
   :doc:`calibration`.

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
like the image; ``band_irradiance`` gives each band's solar irradiance (F0,
``esun``) and ``effective_fwhm`` its width. ``compute_spectral_response``
returns all of it as one ``SpectralResponse``, and ``srf_dataset`` turns that
into a product's ``metadata/srf`` group: ``esun``, effective FWHM,
``earth_sun_distance_au``, and one ``Band_<i>`` variable per band on its own
``wav_Band_<i>`` grid. Polymer reads its SRFs from this group (its
``srf_getter`` is ``hypso.ac.ac_polymer_srf_getter_from_l1``), and readers that
convert radiance themselves can take F0 and the Earth-Sun distance from it.

``hypso.processing.toa`` is the L1C -> L1D step: ``toa_reflectance`` computes
``pi L d^2 / (F0 cos(sza))``, with the Earth-Sun distance ``d`` from
``hypso.srf.earth_sun_distance``; ``spectral_response(tree)`` and
``srf_node(tree)`` build the above for a product (atmospheric correction uses
``srf_node`` to give Polymer SRFs for an L1C input).

Atmospheric correction
----------------------

``hypso.processing.correct_atmosphere`` runs DPS in-process, and Polymer,
ACOLITE and OC-SMART as separate processes:

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
``hypso_l1c`` and ``hypso_l1d`` are registered Satpy reader plugins, so files
open through Satpy's own interface without this package's API::

    from satpy import Scene

    scene = Scene(reader="hypso_l1c", filenames=["HYPSO2_HSI.….L1C.nc"])
    scene.load(["Lt_550", "latitude", "longitude"])

Both readers share ``HypsoL1FileHandler`` (``hypso/satpy/hypso_handler.py``).
Bands are discovered per file with ``hypso.io.product.list_band_variables``,
because the band set depends on binning and calibration. The readers match
current and older file names. They are registered through a ``satpy.readers``
entry point, so the package must be installed (an editable install is
enough).

Resampling
----------

``Product.resample(area_def)`` resamples the cube, derived variables and flags
onto a pyresample area by nearest neighbour, within the product's ground
resolution by default, and returns a Dataset on the target grid with its
latitude and longitude. It is the starting point for regridding to common or
other sensors' grids (planned).

Extending the package
---------------------

A new sensor
   A profile module in ``hypso/sensors`` plus its modes YAML; a calibration
   package with its files.

New calibration coefficients
   Files in the calibration package's ``data/`` and entries in its catalog;
   then update the checksum manifest and tag a release (:doc:`versioning`).

A new atmospheric-correction method
   A ``_<method>(tree, masked, mask_flags, work_dir, config, **options)``
   function in ``processing/atmospheric.py`` that runs the method, maps its
   bands onto the input and returns ``_l2a_tree(...)``, plus its entry in
   ``_METHODS`` and ``_INPUT_LEVELS``. Run external tools in a subprocess.

A new product level
   A ``LevelSchema`` in ``io/schema.py`` (title, product prefix and units,
   whether it has geometry), a processing function that returns a tree with
   that ``processing_level``, and a ``Product`` method calling it.

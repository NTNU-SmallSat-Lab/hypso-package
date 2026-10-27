Products and files
==================

Levels
------

=========  ==============  ===============================================  ===================
Level      Cube            Contents                                         Units (as written)
=========  ==============  ===============================================  ===================
L1A        ``dn``          raw counts, as downlinked                        1 (counts)
L1B        ``Lt``          top-of-atmosphere radiance                       W m-2 um-1 sr-1
L1C        ``Lt``          L1B with latitude, longitude and angles          W m-2 um-1 sr-1
L1D        ``rhot``        top-of-atmosphere reflectance                    1
L2A        ``Rrs``,        atmospherically corrected reflectance; the       sr-1 (``Rrs``),
           ``rho_w``,      variable depends on the method                   1 (``rho_w``,
           ``rhos``        (:doc:`atmospheric_correction`)                  ``rhos``)
=========  ==============  ===============================================  ===================

``hypso.open`` reads any level; ``product.level`` says which it is.

File names
----------

.. code-block:: text

   {SAT}_{INSTRUMENT}.{YYYYMMDDTHHMMSS}.{target}.{LEVEL}[.{method}][.{grid}][.{label}].nc

   HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc
   HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.polymer.nc
   HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.utm33n_876m.nc      (regridded)
   HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.test.nc             (label)

The first field is the satellite and the instrument, as in NASA's
``{MISSION}_{INSTRUMENT}`` names (``PACE_OCI``): ``HYPSO2_HSI`` is the
Hyperspectral Imager (HSI) on HYPSO-2. The instrument field is there for
other instruments on board (an RGB camera, future optical instruments): each
gets its own token in ``hypso.io.naming.INSTRUMENTS`` and its own Satpy
readers, which match only that token. Then comes the capture time (UTC), so a
listing groups by satellite and sorts by time. ``.`` only separates fields. L2A always names its
atmospheric correction method, a regridded product its grid (:doc:`grid`);
the optional label is for experiments. The calibration set and masking are
not part of the name: they are recorded in the file. Older names
(``{target}_{YYYY-MM-DDTHH-MM-SSZ}-{label}-{level}.nc``) are still read.
``hypso.io.naming`` builds and parses names.

What is in a file
-----------------

Root group
   One 2D variable per band, named ``{cube}_{wavelength}`` (``Lt_378``,
   ``rhot_413``, ...), each with ``wavelength``, ``radiation_wavelength``
   (what ESA SNAP reads as the band's wavelength), ``fwhm`` and ``band`` (its
   index). Then ``latitude``, ``longitude`` and the sun and sensor angles from
   L1C on, ``pixel_flags`` if the product has masks, and any derived 2D
   products (an AC tool's chlorophyll, for example).

``metadata/``
   Capture configuration, timing, attitude (ADCS), calibration coefficients
   used, ground control points, spectral response functions (L1D and L2A,
   ``metadata/srf``) and the processing record (``metadata/processing``).

Global attributes
   CF-1.10 and ACDD-1.3: ``title``, ``processing_level``,
   ``product_format_version`` (``2.0``; files without it are format 1),
   ``date_created``, geographic bounds, and ``history`` with one line per
   processing step.

Data are compressed (zlib level 4) and stored unmasked; L1A counts stay
``uint16``. ``product.write(datacube=True)`` writes the cube as one 3D
variable instead of one variable per band.

Band subsets
------------

``select_bands`` keeps some bands and drops the rest, from L1B on (calibration
needs every band, because the smile correction resamples each spectrum across
bands):

.. code-block:: python

   vis = l1b.select_bands(wavelength_range=(450, 700))   # bands centred in 450-700 nm
   rgb = l1d.select_bands(wavelengths=[640, 550, 460])   # the nearest band to each
   few = l1d.select_bands(indices=[20, 40, 60])          # by index in the full band set

Each kept band keeps its index in the sensor's full band set (the
``band_index`` coordinate; ``band`` attribute in files), so processing a subset
gives the same values as processing all bands and selecting afterwards: TOA
reflectance and ``metadata/srf`` look F0 and the SRFs up by it. The selection is
recorded in the processing record. ``metadata/corrections`` stays complete; in a
subset its full-band variables use the dimension ``bands_full``. Atmospheric
correction on subsets works for the in-process methods; the external tools have
not been tried on them.

Masks
-----

Products are written unmasked. Land, cloud and custom masks are bits of one
``pixel_flags`` variable with CF ``flag_masks`` and ``flag_meanings``:

.. code-block:: python

   flagged = l1d.with_masks(land=land, cloud=cloud,
                            descriptions={"cloud": "my cloud mask v2"})
   flagged.flags["cloud"]                     # boolean DataArray
   l2a = flagged.correct_atmosphere("dps", masked=True)   # NaN input where flagged

Processing record
-----------------

Every step adds a ``metadata/processing/<step>`` group: software version and
git commit, input file, parameters, and for calibration the coefficient
files with their checksums, for atmospheric correction the tool version and
its full settings (credentials removed). The groups are kept when a product
is read and written again, so a file records its whole lineage.

.. code-block:: python

   for step in l2a.provenance:
       print(step["step"], step["date_utc"], step["software_version"])

Older files
-----------

``hypso.open`` also reads files written before format 2 (``/products`` and
``/geometry`` groups) and ground-station L1A files. Writing them again
converts them; ``hypso.io.product.convert_product(path)`` does that and gives
the file its current name.

ESA SNAP
--------

Files open in SNAP with File > Open Product. Bands appear under their names
with their wavelengths, so the spectrum view works; masks in ``pixel_flags``
show in the Mask Manager; products from L1C on are geocoded through
``latitude`` and ``longitude``, and regridded products as map-projected
rasters.

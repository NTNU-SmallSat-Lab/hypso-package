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
L2A        ``rhos`` or     atmospherically corrected: surface reflectance   1 (``rhos``),
           ``Rrs``         ``rhos`` (MACHI, the default; DPS; ACOLITE L2R)  sr-1 (``Rrs``)
                           or remote-sensing reflectance ``Rrs``
L2B        none            geophysical products derived from an L2A, one    per variable
                           group per file (``BGC``: chlorophyll, TSM;
                           ``IOP``: IOPs, Kd): 2D
                           variables only (``product.variables``)
=========  ==============  ===============================================  ===================

``hypso.open`` reads any level; ``product.level`` says which it is. The L2A methods
are described in :doc:`atmospheric_correction`.

File names
----------

.. code-block:: text

   {SAT}_{INSTRUMENT}.{YYYYMMDDTHHMMSS}.{target}.{LEVEL}[.{QUANTITY}.{method}][.{grid}][.{label}].nc

   HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc
   HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.RHOS.machi.nc
   HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.RRS.polymer.nc
   HYPSO2_HSI.20250304T103805.aeronetvenice.L2B.BGC.oc4.nc
   HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.utm33n_876m.nc      (regridded)
   HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.test.nc             (label)

The first field is the satellite and the instrument, as in NASA's
``{MISSION}_{INSTRUMENT}`` names (``PACE_OCI``): ``HYPSO2_HSI`` is the
Hyperspectral Imager (HSI) on HYPSO-2. The instrument field is there for
other instruments on board (an RGB camera, future optical instruments): each
gets its own token in ``hypso.io.naming.INSTRUMENTS`` and its own Satpy
readers, which match only that token. Then comes the capture time (UTC), so a
listing groups by satellite and sorts by time. ``.`` only separates fields. L2A always names
its quantity, ``RRS`` (remote-sensing reflectance ``Rrs``, sr-1) or ``RHOS``
(surface reflectance ``rhos``), and the atmospheric correction method that
produced it (ACOLITE's L2W and L2R are both ``acolite``; the quantity tells
them apart). A third token, ``AUX``, names the file of the AC tool's other
per-band outputs next to an L2A (`AC tool intermediates (L2A.AUX)`_). The
quantity is also the ``l2a_quantity`` global attribute. L2A
names from before the quantity field (``L2A.<method>``) are still read; older
POLYMER files hold ``rho_w`` (= pi Rrs) and are named ``RHOW`` when converted.
L2B names its product group (``hypso.io.naming.L2B_GROUPS``, as PACE OCI's
suites: ``BGC`` chlorophyll-a and suspended matter, ``IOP`` absorption,
scattering, backscattering and Kd) and its source, the algorithm or the AC
tool whose own product it is. L2B files written before 2026-11-06 had one
group per product (``CHL``, ``TSM``, ``KD``, ``IOP``, ``AOT``); they are
still read. A regridded product names its grid (:doc:`grid`),
at any level: a single capture on a map stays L1D, L2A or L2B (L3 is kept for
future multi-capture composites);
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
``uint16``. The float bands of L1B-L2A keep 14 significant mantissa bits by
default (netCDF BitRound: at most 3.1e-5 relative error, far below the sensor
noise, about 40% smaller files; values stay float32 and NaN is kept; the
``band_quantization`` attribute says so); ``write(significant_bits=None)``
writes them lossless. ``product.write(datacube=True)`` writes the cube as one 3D
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

Reading L2 files without the package
------------------------------------

``demo/hypso_l2_reader.py`` is a standalone script (numpy and xarray only,
no ``hypso`` import) for users of the L2 files. It opens one method's
L2A.RRS (or RHOS), L2B.BGC and L2B.IOP files of a capture directory as one
``xarray.Dataset``, with the geometry and flags once and each per-band
quantity stacked into a spectral variable (``Rrs`` on ``wavelength``,
``aph`` ... ``bp`` on ``wavelength_iop``, ``aod`` on ``wavelength_aod``):

.. code-block:: python

   from hypso_l2_reader import open_l2, flag_mask
   ds = open_l2("aeronetvenice_2025-03-04T10-38-05Z")      # method= if there are several
   ds["chlor_a_oci"].where(~flag_mask(ds, "land", "cloud"))
   ds["Rrs"].isel(lines=300, samples=500)                   # one spectrum

``aux=True`` adds the L2A.AUX file, ``lazy=True`` reads with dask.
``python hypso_l2_reader.py DIR`` lists the contents and ``--merge out.nc``
writes them to one netCDF file.


L2B products
------------

An L2B product holds geophysical products derived from one L2A product, one
group per file. It has no band cube: every product is a 2D variable with its
own units, and per-wavelength quantities are separate variables (``aph_443``,
``bbp_443``), as in NASA's OC_IOP files. It keeps the L2A's geometry,
``pixel_flags``, grid and metadata, so ``regrid``, ``window`` (statistics per
variable over a ``products`` dimension), ``to_satpy`` and the ``hypso_l2b``
Satpy reader work as for the other levels; ``product.cube`` and band methods
do not apply. ``hypso.processing.make_l2b`` builds one from an L2A and the
derived variables. The AC tools' own products
(Polymer's ``chla``, ACOLITE's ``chl_*``, OC-SMART's ``chlor_a_*``, ...) are
L2B products: ``l2a.tool_products()``, written by ``l2a.write()`` as
``L2B.<GROUP>.<tool>`` files (:doc:`atmospheric_correction`). The package's
own algorithms run on an L2A, mostly on ``Rrs`` (an ``L2A.RRS`` product):

.. code-block:: python

   chl = l2a.derive("BGC")                 # ...L2B.BGC.oci.nc: chlor_a, chl_oc4, chl_ci
   tsm = l2a.derive("BGC", "nechad2010")   # ...L2B.BGC.nechad2010.nc: spm_nechad2010_665
   ndci = l2a.derive("BGC", "ndci")        # red-edge, turbid/eutrophic waters; also from rhos
   iop = l2a.derive("IOP")                 # ...L2B.IOP.qaa.nc: a_, bbp_, Kd_<nm> at 443 ... 665 nm

=====  ==============  ==================================================================
Group  Algorithm       Products and reference
=====  ==============  ==================================================================
BGC    ``oci``         ``chlor_a``: NASA's blend of the colour index (Hu et al. 2012,
       (default)       coefficients 2019) below 0.25 mg m-3 and OC4 above 0.35;
                       also ``chl_oc4``, ``chl_ci``
BGC    ``oc4``         ``chl_oc4``: OC4 band ratio, OLCI coefficients (O'Reilly and
                       Werdell 2019)
BGC    ``ci``          ``chl_ci``: colour index only
BGC    ``ndci``        ``chl_ndci``, ``ndci``: Mishra and Mishra 2012 (turbid and
                       eutrophic waters only; no validity mask)
BGC    ``nechad2010``  ``spm_nechad2010_665`` (g m-3): Nechad et al. 2010 at 665 nm,
                       NaN where rhow >= 0.5 C
IOP    ``qaa``         ``a_<nm>``, ``bbp_<nm>`` (m-1) at 443, 490, 560, 665 nm: Lee's QAA
       (default)       v6, v5 where Rrs665 < 0.0015, as implemented in ACOLITE;
                       ``Kd_<nm>`` (m-1) from a and bbp, Lee et al. 2005 (30 degree sun)
=====  ==============  ==================================================================

Rrs at each algorithm's wavelength is interpolated between the two nearest
bands. Coefficients follow the tables in ACOLITE (``data/Shared/algorithms``)
and the cited papers. QAA does not yet split absorption into phytoplankton
and CDOM/detritus. Negative Rrs (e.g. in the red over clear water) gives
negative TSM, absorption and Kd; there is no validity flag for it yet.

AC tool intermediates (L2A.AUX)
-------------------------------

Some AC tools write more per-band quantities than the reflectance: OC-SMART
writes ``Lt``, ``Lr``, ``Lrc``, ``Lwp`` (its whitecap radiance, not
water-leaving radiance), ``tg_sol`` and ``tg_sen`` (its ``l2_prod``) for every
band it returns. They are kept, but not in the L2A:
``l2a.intermediates()`` is an L2A product with ``l2a_quantity`` ``AUX`` and no
cube, one 2D variable per quantity and band (``Lt_434``, ``tg_sol_434``,
attributes ``wavelength`` and ``source_variable``), with the L2A's geometry
and flags, and ``l2a.write()`` writes it next to the L2A:

.. code-block:: text

   HYPSO2_HSI.20250528T101551.mjosa.L2A.RRS.ocsmart.nc     Rrs, aod_<nm>
   HYPSO2_HSI.20250528T101551.mjosa.L2B.BGC.ocsmart.nc     chlor_a_oci, chlor_a_yoc, tsm_yoc
   HYPSO2_HSI.20250528T101551.mjosa.L2B.IOP.ocsmart.nc     aph_, adg_, bbp_, apg_, bp_<nm> (m-1)
   HYPSO2_HSI.20250528T101551.mjosa.L2A.AUX.ocsmart.nc     Lt_434 ... tg_sen_718 (6 x 83)

OC-SMART's per-band geophysical products are not in AUX: the IOPs go to
L2B.IOP and the aerosol optical depth stays in the L2A, named as NASA's
OC IOP products (its user guide, 3.6.1): ``aph`` phytoplankton absorption,
``adg`` detritus + CDOM absorption, ``bbp`` particulate backscattering,
``apg`` (its ``at``) particulate + CDOM absorption without water, ``bp`` (its
``bt``) particulate scattering, ``aod`` (its ``AOD``) aerosol optical depth;
``source_variable`` keeps OC-SMART's name. The default ``l2_prod``
(``hypso.processing.atmospheric.OCSMART_L2_PROD``) requests all of them and
``tsm``; set ``l2_prod`` in the AC tool configuration for fewer (keep ``rrs``).

For HYPSO, OC-SMART's input is ``rhot cos(solar zenith) / pi``, the radiance
divided by the solar irradiance F0, so its ``Lt``, ``Lr`` and ``Lrc`` are in
sr-1 (``Lt`` equals that input); ``tg_*`` are transmittances. Together the
three files hold every variable of the tool's output (geometry as the
product's own). ``write(intermediates=False)`` skips the AUX file, and
``regrid`` drops the intermediates (write the L2A first). The ``hypso_l2a``
Satpy reader reads an AUX file's variables like an L2B file's. Polymer and
ACOLITE return no intermediates (ACOLITE L2R's ``rhot`` is its copy of the
input).

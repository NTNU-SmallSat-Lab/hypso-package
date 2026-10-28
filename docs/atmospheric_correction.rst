Atmospheric correction
======================

``Product.correct_atmosphere(method, ...)`` turns an L1C or L1D product into an
L2A product. Without a method it uses MACHI, which needs nothing installed:

.. code-block:: python

   import hypso

   l1d = hypso.open("....L1D.nc")
   l2a = l1d.correct_atmosphere()                    # MACHI (default)
   l2a = l1d.correct_atmosphere("polymer", config="ac_tools.yaml")
   l2a.write()      # HYPSO2_HSI.<time>.<target>.L2A.RRS.polymer.nc

==============  ===========  ==============================  ============================
Method          Input        Product                         Runs
==============  ===========  ==============================  ============================
``machi``       L1D          ``rhos``                        in this process (default)
``dps``         L1D          ``rhos``                        in this process
``polymer``     L1C or L1D   ``Rrs`` (Polymer's ``rho_w``    Polymer, in a subprocess
                             divided by pi)
``acolite``     L1C or L1D   ``Rrs`` (``output="l2w"``,      ACOLITE, in a subprocess
                             default) or ``rhos``
                             (``output="l2r"``)
``ocsmart``     L1D          ``Rrs``                         OC-SMART (``OCSMART.py``)
==============  ===========  ==============================  ============================

The L2A product keeps the input's geometry, ``pixel_flags`` and metadata. Its
cube is on the input's bands; bands the method does not return are NaN. The
method, its version, the installation's git commit and the settings used are
recorded in ``metadata/processing/l2a_<method>``.

Every method produces one of two quantities, named in the file name
(:doc:`products`): ``Rrs`` (``L2A.RRS.<method>``), remote-sensing reflectance
in sr-1 (CF standard name
``surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air``),
or ``rhos`` (``L2A.RHOS.<method>``), surface reflectance, dimensionless.
Polymer's water-leaving reflectance ``rho_w`` (= pi Rrs) is divided by pi, so
all water products are ``Rrs``. ``dps`` subtracts each band's darkest value
from the TOA reflectance: only the path signal is removed (no transmittance
or glint correction), so the result is ``rhos``.

Each method's 2D outputs are kept as variables next to the cube (in
``product.variables``), with the tool's own flags as an integer flag variable:

==============  ==================================================================
Method          Variables
==============  ==================================================================
``dps``         none
``polymer``     ``chla`` (10**logchl, mg m-3), ``logfb``, ``SPM``, ``Rgli``,
                ``Rnir``, ``polymer_flags`` (with Polymer's flag table)
``acolite``     L2W: every 2D output except lat/lon/Rrs (what ``l2w_parameters``
                asks for, e.g. ``chl_re_mishra``, ``SPM_Nechad2010_664``,
                ``ndci``) and ``acolite_l2_flags``; L2R: none
``ocsmart``     ``chlor_a_oci``, ``chlor_a_yoc``, ``tsm_yoc`` (when written) and
                ``ocsmart_l2_flags``
==============  ==================================================================

Tool standard names that are not CF names are kept as ``source_standard_name``;
ACOLITE's chlorophyll and SPM names are mapped to the CF ``..._in_sea_water``
names.

MACHI
-----

MACHI (Minimal Atmospheric Compensation for Hyperspectral Imagers, NTNU
SmallSat Lab, `github.com/NTNU-SmallSat-Lab/MACHI
<https://github.com/NTNU-SmallSat-Lab/MACHI>`_, MIT licence) is included in the
package (``hypso.ac.machi``, from upstream commit ``ebbc3e72``, unchanged). It
models the surface reflectance as ``rhos = (rhot - A) / T``, with one scattering
spectrum ``A`` and one transmission spectrum ``T`` for the whole image, fitted so
that the corrected spectra are as smooth as possible: surface spectra are
smooth, gas absorption is not. ``A`` starts from the darkest pixel per band.

.. code-block:: python

   l2a = l1d.correct_atmosphere("machi", est_min_R=0.05, tol=0.01)
   l2a.tree["metadata/machi"]        # the fitted T and A, and the objective per iteration

Options: ``est_min_R`` (expected minimum surface reflectance, default 0.05),
``tol`` (relative change of the objective at which the fit stops, default
0.01), ``approach_rate`` (default 1), and ``fit_exclude_flags`` (pixel_flags
left out of the fit, e.g. ``["cloud"]``; by default every finite pixel is used,
as upstream), and ``fit_subsample`` (fit on every Nth pixel; default 1). The
correction is applied to every pixel.

``fit_subsample`` trades a little accuracy for speed and memory, e.g. for many
captures in parallel. The fit always keeps each band's darkest pixel, because
MACHI starts the scattering spectrum from the per-band minimum; without it the
result over dark water shifts by several percent. On the reference capture:

=====================  =======  ===========  ====================================
``fit_subsample``      Fit      Peak memory  Change in ``rhos``
=====================  =======  ===========  ====================================
1 (default)            34 s     3.3 GB       (reference)
16                     2.7 s    0.8 GB       about 7e-5 (0.3% of dark water)
=====================  =======  ===========  ====================================

Bands without signal
(HYPSO's smile correction leaves the bands below 400 nm at zero) are not fitted
and come out NaN.

It is fast and needs no tools, which makes it the default, but treat it as a
first look rather than a validated water product (upstream's own list): little
in situ validation yet; a simple light-propagation model; one correction for the
whole image although aerosols, glint and water vapour vary across it; no sun
glint correction; Rayleigh and aerosol scattering are hard to separate from
smooth surface spectra; bands with nearly complete absorption are a problem; and
it needs the expected minimum reflectance. Its output is surface reflectance
``rhos``, not water-leaving ``Rrs``. On the reference capture (598 x 1092 pixels,
120 bands) it takes about 30 s and 4 GB of memory.

Masked input
------------

Products are written unmasked. To give a method masked input, pass
``masked=True``: pixels where any flag in ``pixel_flags`` is set (or only the
flags in ``mask_flags``) become NaN in the input the method sees.

.. code-block:: python

   l1d.correct_atmosphere("dps", masked=True, mask_flags=["land", "cloud"])

Tool configuration
------------------

The package does not assume where Polymer, ACOLITE or OC-SMART are installed.
Give their locations in a YAML file, passed as ``config=`` or named by the
``HYPSO_AC_CONFIG`` environment variable. Only the sections for the tools you
use are needed:

.. code-block:: yaml

   polymer:
     python: /path/to/envs/ac/bin/python    # optional; default: this Python
     base_path: /path/to/Polymer
     polymer_path: /path/to/Polymer/polymer
     eoread_path: /path/to/Polymer/eoread
     eotools_path: /path/to/Polymer/eotools
     core_path: /path/to/Polymer/core
     version: v2                            # Polymer build: v1 or v2
     ancillary_source: nasa                 # nasa or era5
     settings:                              # Polymer Params overrides
       bands_corr: [448, 459, 470]

   acolite:
     path: /path/to/acolite
     settings_file: /path/to/settings.txt   # optional, over the sensor defaults
     settings:
       dsf_wave_range: "450,800"

   ocsmart:
     dir: /path/to/OCSMART_Linux_v2.6.3
     python: /path/to/envs/ocsmart/bin/python
     solz_limit: 70.0

Any key can be overridden for one call, for example
``correct_atmosphere("polymer", settings={"band_cloudmask": 790})``;
``settings`` dictionaries are merged key by key.

ACOLITE's EARTHDATA credentials (for ancillary data) are never read from the
configuration file. Set ``HYPSO_ACOLITE_EARTHDATA_USERNAME`` and
``HYPSO_ACOLITE_EARTHDATA_PASSWORD`` in the environment instead.

Working files
-------------

External tools read files, so the input is written to a working directory
together with the tool's raw output. By default this is a temporary directory
that is removed afterwards. Pass ``work_dir=`` to keep it. With a kept
``work_dir``, ``reuse_output=True`` reads ACOLITE's earlier output instead of
running it again, for example to get L2R after L2W from one run.

Notes per tool
--------------

* **Polymer** reads the per-band spectral response functions from the input
  file. They are added when the input has none (an L1C made directly by
  ``georeference``).
* **OC-SMART** reads its paths from ``OCSMART_Input.txt`` in its installation
  directory. The package writes that file for the run and restores it
  afterwards, under the same lock file hypso-processing-pipeline uses, so runs
  on one installation never overlap. Because ``OCSMART.py`` rewrites any ``.``
  in these paths, the working directory's path must not contain a dot.

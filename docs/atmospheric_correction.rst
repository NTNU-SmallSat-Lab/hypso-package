Atmospheric correction
======================

``Product.correct_atmosphere(method, ...)`` turns an L1C or L1D product into an
L2A product:

.. code-block:: python

   import hypso

   l1d = hypso.open("....L1D.nc")
   l2a = l1d.correct_atmosphere("polymer", config="ac_tools.yaml")
   l2a.write()      # HYPSO2_HSI.<time>.<target>.L2A.polymer.nc

==============  ===========  ==============================  ============================
Method          Input        Product                         Runs
==============  ===========  ==============================  ============================
``dps``         L1D          ``Rrs``                         in this process
``polymer``     L1C or L1D   ``rho_w``                       Polymer, in a subprocess
``acolite``     L1C or L1D   ``Rrs`` (``output="l2w"``,      ACOLITE, in a subprocess
                             default) or ``rhos``
                             (``output="l2r"``)
``ocsmart``     L1D          ``Rrs``                         OC-SMART (``OCSMART.py``)
==============  ===========  ==============================  ============================

The L2A product keeps the input's geometry, ``pixel_flags`` and metadata. Its
cube is on the input's bands; bands the method does not return are NaN. The
method, its version, the installation's git commit and the settings used are
recorded in ``metadata/processing/l2a_<method>``.

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

Installation
============

.. code-block:: bash

   pip install hypso

Python 3.10 or newer. The calibration coefficients come in two packages
installed with it, ``hypso1_calibration`` and ``hypso2_calibration``.

From a clone, for development:

.. code-block:: bash

   git clone https://github.com/NTNU-SmallSat-Lab/hypso-package.git
   cd hypso-package
   pip install -e hypso -e hypso1_calibration -e hypso2_calibration

Install from a git clone (not a source archive) so the version can be read from
the git tags (:doc:`versioning`). An editable install also registers the Satpy
readers for the product files.

Atmospheric correction tools
----------------------------

Dark pixel subtraction (``dps``) needs nothing more. Polymer, ACOLITE and
OC-SMART are separate installations that the package runs in their own
processes; tell it where they are in a YAML file
(:doc:`atmospheric_correction`).

GDAL
----

If installing a dependency fails on GDAL, install the system library first:

.. code-block:: bash

   sudo apt-get install gdal-bin libgdal-dev

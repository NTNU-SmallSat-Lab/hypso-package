# HYPSO Python Package
"hypso" processes the hyperspectral images taken by the HYPSO-1 and HYPSO-2 satellites developed by the
Norwegian University of Science and Technology (NTNU), from raw data to atmospherically corrected
reflectance, and writes CF-compliant NetCDF files that open in ESA SNAP.

Product levels:
- L1A: raw data (digital numbers)
- L1B: top-of-atmosphere radiance
- L1C: top-of-atmosphere radiance with georeferencing
- L1D: top-of-atmosphere reflectance with georeferencing
- L2A: atmospherically corrected reflectance (MACHI by default; dark pixel subtraction, Polymer, ACOLITE or OC-SMART)

## Quick start
```python
import hypso

l1a = hypso.open("aeronetvenice_2025-06-12T09-58-02Z-l1a.nc")   # any level, any HYPSO file layout
l1d = (l1a.calibrate()                                          # HYPSO-2: abs / V3 by default
          .georeference(latitudes="latitudes_indirectgeoref.dat",
                        longitudes="longitudes_indirectgeoref.dat")
          .toa_reflectance())
l1d.write()          # HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.nc

l2a = l1d.correct_atmosphere()                                  # MACHI; or "polymer", config="ac_tools.yaml"
l2a.spectrum(latitude=45.314, longitude=12.508)                 # spectrum at a location
```
Every step returns a new product and records itself in the file (`metadata/processing`, `history`).

## Links
- Documentation: [https://ntnu-smallsat-lab.github.io/hypso-package/](https://ntnu-smallsat-lab.github.io/hypso-package/)
- Source Code: [https://github.com/NTNU-SmallSat-Lab/hypso-package](https://github.com/NTNU-SmallSat-Lab/hypso-package)
- Issues: [https://github.com/NTNU-SmallSat-Lab/hypso-package/issues](https://github.com/NTNU-SmallSat-Lab/hypso-package/issues)
- PyPI URL: [https://pypi.org/project/hypso/](https://pypi.org/project/hypso/)

## Installation
The HYPSO package can be installed using the Python package manager `pip`:
```
pip install hypso
```

## Calibration Libraries
Calibration files for HYPSO-1 and HYPSO-2 are distributed in two separate Python packages, installed
as dependencies of `hypso`:
- hypso1_calibration: [https://pypi.org/project/hypso1-calibration/](https://pypi.org/project/hypso1-calibration/)
- hypso2_calibration: [https://pypi.org/project/hypso2-calibration/](https://pypi.org/project/hypso2-calibration/)

## Development
- [Packaging projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- Create an account at [PyPI.org](https:/pypi.org) and request access to the hypso project (contact Cameron or Aria, updated 2025-02-17)
- Add your [PyPI.org](https:/pypi.org) login credentials and token to `~/.pypirc`
- Versions come from git tags (calendar versioning, `vYY.M.N`; pre-releases `vYY.M.Na1`, `b1`, `rc1`),
  shared by `hypso`, `hypso1_calibration` and `hypso2_calibration`: tag the release commit, then build.
  See `docs/versioning.rst`.
- Documentation: Sphinx in `docs/` (`pip install -r docs/requirements.txt`, then
  `sphinx-build -W -b html docs docs/_build/html`); CI builds it on every push and publishes `main`.
- Build the package with `python3 -m build`
- Upload the newly built package to PyPI: `python3 -m twine upload --repository pypi dist/*`
- View the project at [pypi.org/project/hypso/](https://pypi.org/project/hypso/)
- Important Considerations:
    1. Read package data files through `importlib.resources`, e.g.
       `files('hypso.srf').joinpath('data/Solar_irradiance_Thuillier_2002.csv')`.
    2. Any non-python file that wants to be included to be uploaded needs to be added in the `MANIFEST.in` file
    3. Use lower bounds (>=) rather than exact versions (==) for dependencies in `pyproject.toml`.

## Authors
- Maintainers: Cameron Penne (@CameronLP)
- Calibration: Marie Henriksen, Joe Garett, Aria Alinejad
- Georeferencing: Sivert Bakken, Dennis Langer

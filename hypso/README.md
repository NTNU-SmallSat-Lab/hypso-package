# HYPSO Python Package
"hypso" is a simple, fast, processing and visualization tool for the hyperspectral
images taken by the HYPSO-1 and HYPSO-2 satellites developed by the Norwegian University of Science and
Technology (NTNU).

The HYPSO package can process the following data products for HYPSO-1 and HYPSO-2 hyperspectral captures:
- L1a (raw data)
- L1b (top-of-atmosphere radiance)
- L1c (top-of-atmosphere radiance with georeferencing)
- L1d (top-of-atmosphere reflectance with georeferencing)

## Links
- Documentation: [https://ntnu-smallsat-lab.github.io/hypso-package/](https://ntnu-smallsat-lab.github.io/hypso-package/) (NB: out of date)
- Source Code: [https://github.com/NTNU-SmallSat-Lab/hypso-package](https://github.com/NTNU-SmallSat-Lab/hypso-package)
- Issues: [https://github.com/NTNU-SmallSat-Lab/hypso-package/issues](https://github.com/NTNU-SmallSat-Lab/hypso-package/issues)
- PyPI URL: [https://pypi.org/project/hypso/](https://pypi.org/project/hypso/)

## Installation
The HYPSO package can be installed using the Python package manager `pip`:
```
pip install hypso
```
If you encounter an error about gdal, try the following commands:
```
sudo apt-get install gdal-bin libgdal-dev
pip install gdal==3.8.4
pip install hypso
```

## Calibration Libraries
Radiometric calibration files for HYPSO-1 and HYPSO-2 are distributed in two separate Python packages which can also be installed using `pip`:
- hypso1_calibration: [https://pypi.org/project/hypso1-calibration/](https://pypi.org/project/hypso1-calibration/)
- hypso2_calibration: [https://pypi.org/project/hypso2-calibration/](https://pypi.org/project/hypso2-calibration/)

It is highly recommended to install these packages alongside the HYPSO package using the following commands:
```
pip install hypso1-calibration
pip install hypso2-calibration
```

## Development 
- [Packaging projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- Create an account at [PyPI.org](https:/pypi.org) and request access to the hypso project (contact Cameron or Aria, updated 2025-02-17)
- Add your [PyPI.org](https:/pypi.org) login credentials and token to `~/.pypirc`
- Install the `setuptools` build system and `twine` using pip 
- Update the version number in `pyproject.toml`
    - Use the version number format "vX.Y.Z" for normal releases 
    - Use the version number format "vX.Y.Z.a1" for alpha releases 
    - Use the version number format "vX.Y.Z.b1" for beta releases 
- Build the package with `python3 -m build`
- Upload the newly built package to PyPI: `python3 -m twine upload --repository pypi dist/*`
- View the project at [pypi.org/project/hypso/](https://pypi.org/project/hypso/)
- Important Considerations:
    1. Importing files needs to be done using package file imports like the following line of code.

    ```
    full_rad_coeff_file = files('hypso.calibration').joinpath(
                    f'data/{"radiometric_calibration_matrix_HYPSO-1_full_v1.csv"}')
    ```
        
    2. Any non-python file that wants to be included to be uploaded needs to be added in the `MANIFEST.in` file
    3. Packages names and version in both the `pyproject.toml` and `meta.yaml` are case and space sensitive, be carefull with the spacing. Avoid using specific versions (==) and try to use higher than (>=) as it makes it easier for future compatibility.

## Authors
- Maintainers: Cameron Penne (@CameronLP)
- Calibration: Marie Henriksen, Joe Garett, Aria Alinejad
- Georeferencing: Sivert Bakken, Dennis Langer
- Package: Alvaro Romero


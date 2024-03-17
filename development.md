# DEVELOPMENT OF CONDA-FORGE PACKAGE

There are two main repos:
    - Package: https://github.com/NTNU-SmallSat-Lab/hypso-package
    - Conda-Forge Feedstock: https://github.com/conda-forge/hypso-feedstock

You need:
- Create a Fork of the hypso-package to implement new functionality
- Create Fork of hypso-feedstock to trigger a conda-forge upversion

<span style="color: red; font-weight:bold">IMPORTANT: Only modify your forks and create a PR to merge. Never modify directly. </span>.


# To Update conda-forge Package


1. Modify your fork of hypso

# To Generate SHA 256 for

Use the follosing code and change the version relesed (for linux and mac):

    curl -sL https://github.com/NTNU-SmallSat-Lab/hypso-package/archive/v1.9.3.tar.gz | openssl sha256

1. 

    https://github.com/conda-forge/hypso-feedstock

2. On your fork, modify the following file

    recipe/meta.yaml

Specifically the lines:

    {% set version = "1.9.2" %}

    sha256: 6ef940b60d97e373753371824c348839d04b1ea06f3d8a6719e3f4f9f9b4460d

If new packages are added they should be included in the "run section"

    requirements:
        host:
            - python >=3.9
            - pip
            - setuptools >=61.0
            - setuptools_scm  >=3.4
        run:
            - newpackage

Dont forget to include it in the .toml file in the original package
    
    pyproject.toml


# Important Considerations

1. Importing files needs to be done like this
    full_rad_coeff_file = files('hypso.calibration').joinpath(
                f'data/{"radiometric_calibration_matrix_HYPSO-1_full_v1.csv"}')
2. Any non-python file that wants to be included to be uploaded needs to be added in the "MANIFEST.in" file

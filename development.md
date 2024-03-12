
There are two main repos:
    - Package: https://github.com/NTNU-SmallSat-Lab/hypso-package
    - Conda-Forge Feedstock: https://github.com/conda-forge/hypso-feedstock

# To Generate SHA 256 for

Use the follosing code and change the version relesed (for linux and mac):

    curl -sL https://github.com/NTNU-SmallSat-Lab/hypso-package/archive/v1.9.3.tar.gz | openssl sha256


1. Create Fork of conda-forge package feedstock

    https://github.com/conda-forge/hypso-feedstock

2. On your fork, modify

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

# HYPSO
"hypso" is a simple, fast, processing and visualization tool for the hyperspectral
images taken by the HYPSO mission from the Norwegian University of Science and
Technology (NTNU) for Python >3.10

- Documentation: https://ntnu-smallsat-lab.github.io/hypso-package/ (NB: out of date)
  
- Development: https://github.com/NTNU-SmallSat-Lab/hypso-package
  
- PyPI URL: https://pypi.org/project/hypso/
  
- Anaconda URL: https://anaconda.org/conda-forge/hypso

- Anaconda Github Feedstock: https://github.com/conda-forge/hypso-feedstock

## Installation

### Pip 
```
pip install hypso
```

### Anaconda
It is recommended to use mamba as it is less prone to errors in dependency management than the default conda terminal (see https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

if conda installed:
```
conda install -c conda-forge conda-libmamba-solver 
conda config --set solver libmamba
conda create -n hypsoenv python=3.9
conda activate hypsoenv
conda install -c conda-forge hypso
```

if mamba installed:
```
mamba create -n hypsoenv python=3.9
mamba activate hypsoenv
mamba install -c conda-forge hypso
```

To update to the most recent version it is suggested to run the following code (change "mamba" for "conda" if needed):
```
mamba search -c conda-forge hypso
mamba update hypso
```

## Authors

- Package: @DevAlvaroF, @CameronLP
- Correction Coefficients: Marie Henriksen, Joe Garett
- Georeferencing: Sivert Bakken, Dennis Langer


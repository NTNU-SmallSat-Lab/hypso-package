"""Spectra of single pixels of a hypso.Product (see Product.spectrum)."""
from pathlib import Path

import matplotlib.pyplot as plt


def get_spectrum(product, latitude=None, longitude=None, line: int = None, sample: int = None):
    """The cube's spectrum at the pixel nearest to latitude/longitude, or at
    line/sample, as a DataArray over bands with wavelength coords."""
    return product.spectrum(latitude=latitude, longitude=longitude, line=line, sample=sample)


def plot_spectrum(product, latitude=None, longitude=None, line: int = None, sample: int = None,
                  path=None) -> Path:
    """Plot that spectrum against wavelength; saved to `path` if given,
    shown otherwise."""
    if latitude is not None and longitude is not None:
        line, sample = product.pixel(latitude, longitude)
    spectrum = product.spectrum(line=line, sample=sample)
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(spectrum["radiation_wavelength"].values, spectrum.values)
    axes.set_xlabel("Wavelength (nm)")
    axes.set_ylabel(f"{spectrum.name} ({product.tree.dataset[spectrum.name].attrs.get('units', '') or 'see product'})")
    location = f"({latitude}, {longitude}) -> " if latitude is not None else ""
    axes.set_title(f"{product.level} {spectrum.name}, {location}(line {line}, sample {sample})")
    axes.grid(True)
    if path is not None:
        figure.savefig(path)
        plt.close(figure)
        return Path(path)
    plt.show()
    return None

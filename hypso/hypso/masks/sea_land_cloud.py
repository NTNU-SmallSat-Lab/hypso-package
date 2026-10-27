"""Reader for the ground processing's sea/land/cloud classification.

The ground processing writes ``processing-temp/sea-land-cloud.labels``: one
uint8 label per pixel (0 water, 1 land, 2 cloud), in the capture's
``(lines, samples)`` order, from a CNN classifier.

Examples
--------
>>> masks = sea_land_cloud_masks("processing-temp/sea-land-cloud.labels", l1d.cube.shape[:2])
>>> flagged = l1d.with_masks(land=masks["land"], cloud=masks["cloud"])
"""
from pathlib import Path

import numpy as np

#: Label value of each class.
LABELS = {"water": 0, "land": 1, "cloud": 2}


def read_sea_land_cloud(path, shape) -> np.ndarray:
    """Read a sea/land/cloud label file.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``sea-land-cloud.labels`` file.
    shape : tuple of int
        ``(lines, samples)`` of the capture.

    Returns
    -------
    numpy.ndarray
        uint8 labels, see :data:`LABELS`.

    Raises
    ------
    ValueError
        If the file's size does not match ``shape``.
    """
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path} holds {data.size} labels, a {shape[0]} x {shape[1]} capture needs "
                         f"{int(np.prod(shape))}")
    return data.reshape(shape)


def sea_land_cloud_masks(path, shape) -> dict:
    """Boolean water, land and cloud masks from a sea/land/cloud label file.

    Parameters
    ----------
    path, shape
        As in :func:`read_sea_land_cloud`.

    Returns
    -------
    dict
        ``{"water", "land", "cloud": boolean array}``, ready for
        :meth:`hypso.Product.with_masks`.
    """
    labels = read_sea_land_cloud(path, shape)
    return {name: labels == value for name, value in LABELS.items()}

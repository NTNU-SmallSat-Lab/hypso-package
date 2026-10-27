"""Processing steps as functions from one product tree to the next level's.

The trees are :mod:`hypso.io.product`'s in-memory model; :class:`hypso.Product`
calls these functions. Each step records itself in ``metadata/processing``
(:mod:`hypso.io.provenance`) and leaves its input tree unchanged.
"""
from .capture import CaptureParams, capture_params, classify_capture_type
from .calibrate import l1a_to_l1b
from .georeference import l1b_to_l1c
from .toa import l1c_to_l1d, toa_reflectance
from .atmospheric import correct_atmosphere
from .bands import select_bands

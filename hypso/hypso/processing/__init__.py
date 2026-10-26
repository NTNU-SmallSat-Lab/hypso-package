"""Processing steps as functions from one product tree (hypso.io.product's
in-memory model) to the next level's tree (REFACTOR_PROGRESS.md x37 step 2).
Each step records itself in metadata/processing (hypso.io.provenance) and
leaves its input tree unchanged."""
from .capture import CaptureParams, capture_params
from .calibrate import l1a_to_l1b
from .georeference import l1b_to_l1c
from .toa import l1c_to_l1d, srf_dataset
from .atmospheric import correct_atmosphere

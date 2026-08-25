# Compatibility shim: this package moved to hypso.masks (these classifier models
# exist to produce sea/land/cloud masks - the mask-oriented name says what they
# are for, not how they work; see REFACTOR_PROGRESS.md). Import from hypso.masks
# in new code; this shim keeps the confirmed external import path used by
# hypso-processing-pipeline (from hypso.classification import decode_jon_cnn_*)
# working unchanged.
from hypso.masks import *  # noqa: F401,F403

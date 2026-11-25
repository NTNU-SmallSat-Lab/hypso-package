"""Atmospheric correction support: the external tools' subprocess drivers
(ac/adapters), their output readers, Polymer's SRF hook and the tool
configuration. The atmospheric-correction step itself is
hypso.processing.correct_atmosphere / hypso.Product.correct_atmosphere."""
from .ac_polymer import ac_polymer_srf_getter, SRF_GETTER_PATH, \
                         ac_polymer_srf_getter_from_l1, SRF_GETTER_FROM_L1_PATH
from .acolite_output import load_acolite_l2r_nc, load_acolite_l2w_nc
from .polymer_output import load_polymer_l2_v1_nc, load_polymer_l2_v2_nc
from .ocsmart_output import load_ocsmart_h5
from .adapters.base import ACRunError, run_subprocess_driver

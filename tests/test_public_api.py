"""The public API (REFACTOR_PROGRESS.md ×50): what users import, and that the
removed capture-object API stays removed. No real data needed."""
import importlib

import pytest


def test_top_level_api():
    import hypso
    from hypso import Product, open, read_product, write_product  # noqa: F401, A004

    assert callable(hypso.open) and isinstance(Product, type)
    assert hypso.__version__


def test_processing_functions():
    from hypso.processing import (  # noqa: F401
        capture_params, correct_atmosphere, l1a_to_l1b, l1b_to_l1c, l1c_to_l1d,
    )


def test_io_modules():
    from hypso.io.naming import parse_product_name, product_name  # noqa: F401
    from hypso.io.product import convert_product, list_band_variables, read_product, write_product  # noqa: F401
    from hypso.io.provenance import add_step, PRODUCT_FORMAT_VERSION  # noqa: F401


def test_masks_and_ac():
    from hypso.masks import decode_jon_cnn_cloud_mask, decode_jon_cnn_land_mask  # noqa: F401
    from hypso.masks.flags import apply_flags, decode_flags, encode_flags  # noqa: F401
    from hypso.ac import (  # noqa: F401
        load_acolite_l2r_nc, load_acolite_l2w_nc, load_ocsmart_h5, load_polymer_l2_v1_nc, load_polymer_l2_v2_nc,
    )
    from hypso.ac.config import load_ac_config, tool_config  # noqa: F401


def test_polymer_srf_getter_hook_paths():
    # Polymer resolves these by dotted name (srf_getter=<path>) inside its own
    # process, so they must resolve exactly as written.
    mod = importlib.import_module("hypso.ac.ac_polymer")
    for path, func in ((mod.SRF_GETTER_PATH, mod.ac_polymer_srf_getter),
                       (mod.SRF_GETTER_FROM_L1_PATH, mod.ac_polymer_srf_getter_from_l1)):
        module_path, attr_name = path.rsplit(".", 1)
        assert getattr(importlib.import_module(module_path), attr_name) is func


@pytest.mark.parametrize("name", ["Hypso", "Hypso1", "Hypso2"])
def test_capture_object_api_removed(name):
    import hypso
    assert not hasattr(hypso, name)


@pytest.mark.parametrize("module", ["hypso.HypsoCapture", "hypso.capture_types", "hypso.containers",
                                    "hypso.io.dispatch", "hypso.io.reader", "hypso.io.writer", "hypso.load",
                                    "hypso.write", "hypso.calibration.pipeline", "hypso.ac.adapters.polymer"])
def test_old_modules_removed(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)

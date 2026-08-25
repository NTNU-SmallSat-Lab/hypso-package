"""Plan §Verification item 4: every externally-depended-on name still imports
from where external code imports it, and the class hierarchy/export surface is
intact. No real data needed."""
import importlib

import pytest


def test_top_level_exports():
    from hypso import Hypso, Hypso1, Hypso2  # noqa: F401
    from hypso.HypsoBase import HypsoBase

    assert issubclass(Hypso1, HypsoBase)
    assert issubclass(Hypso2, HypsoBase)


def test_confirmed_external_writer_names():
    # The five names hypso-processing-pipeline's process_capture.py imports
    # directly - the hard-frozen surface of this refactor.
    from hypso.write import (  # noqa: F401
        write_l1b_nc_file,
        write_l1c_nc_file,
        write_l1d_nc_file,
        write_l2a_nc_file,
        write_products_nc_file,
    )


def test_load_names():
    from hypso.load import (  # noqa: F401
        load_l1a_nc,
        load_l1b_nc,
        load_l1c_nc,
        load_l1d_nc,
        load_l2a_nc,
        load_ocsmart_h5,
        load_acolite_l2r_nc,
        load_acolite_l2w_nc,
        load_polymer_l2_v1_nc,
        load_polymer_l2_v2_nc,
    )


def test_composition_modules_import():
    # The extracted HypsoBase slices (self.geo / self.calibration / self.io /
    # self.ac composition) - each must import standalone.
    for mod in ("hypso.geo", "hypso.calibration.pipeline", "hypso.io.dispatch",
                "hypso.ac.adapters"):
        importlib.import_module(mod)


def test_import_order_independence():
    # Regression guard for the circular-import class of bug hit twice during
    # the refactor (hypso.io.writer <-> hypso.write, hypso.io.reader <->
    # hypso.load): each subpackage must be importable first, in a fresh
    # interpreter. Approximated here in-process: these must all already be
    # importable regardless of what conftest/other tests imported earlier.
    for mod in ("hypso.io", "hypso.load", "hypso.write", "hypso.ac", "hypso"):
        importlib.import_module(mod)


def test_polymer_srf_getter_hook_path():
    # Polymer resolves this BY DOTTED-STRING NAME ("hypso.ac.ac_polymer_srf_getter"
    # passed as srf_getter=...) - the import path is frozen API even though
    # nothing in this package calls it by name.
    mod = importlib.import_module("hypso.ac")
    assert callable(mod.ac_polymer_srf_getter)


def test_hypso_base_ac_wrapper_surface():
    # Every public ac_* method name confirmed used by hypso-processing-pipeline
    # must still exist on HypsoBase (as delegating wrappers post-extraction).
    from hypso.HypsoBase import HypsoBase

    expected = [
        "ac_ocsmart_stage_input", "ac_ocsmart_run_correction", "ac_ocsmart_open_output",
        "ac_acolite_run_correction", "ac_acolite_open_output",
        "ac_polymer_run_correction", "ac_polymer_open_output",
        "ac_polymer_generate_srf_nc", "ac_polymer_generate_ssi_nc", "ac_polymer_generate_esun_nc",
        "ac_dark_pixel_subtraction",
    ]
    for name in expected:
        assert callable(getattr(HypsoBase, name)), name


def test_hypso_base_geo_wrapper_surface():
    # run_georeferencing (hypso-processing-pipeline) and
    # run_direct_georeferencing (hypso/ac/loading_acolite_output.py) stayed as
    # methods when their bodies moved to hypso.geo.
    from hypso.HypsoBase import HypsoBase

    assert callable(HypsoBase.run_georeferencing)
    assert callable(HypsoBase.run_direct_georeferencing)

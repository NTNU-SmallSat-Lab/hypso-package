"""Plan §Verification item 4: every externally-depended-on name still imports
from where external code imports it, and the class hierarchy/export surface is
intact. No real data needed."""
import importlib

import pytest


def test_top_level_exports():
    from hypso import Hypso, Hypso1, Hypso2  # noqa: F401
    from hypso.HypsoCapture import HypsoCapture

    assert issubclass(Hypso1, HypsoCapture)
    assert issubclass(Hypso2, HypsoCapture)


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
    # The extracted HypsoCapture slices (self.geo / self.calibration / self.io /
    # self.ac composition) - each must import standalone.
    for mod in ("hypso.georeferencing.geo", "hypso.calibration.pipeline", "hypso.io.dispatch",
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
    # Polymer resolves this BY DOTTED-STRING NAME (srf_getter=<dotted path>,
    # resolved via importlib.import_module + getattr inside Polymer's own
    # code - see ac_polymer.py's SRF_GETTER_PATH docstring). SRF_GETTER_PATH
    # is derived from the function object itself (module + qualname), not
    # hand-typed - deliberately NOT asserted against a literal string here,
    # since hardcoding the expected value would reintroduce the exact
    # drift risk this constant exists to eliminate. Reproduces Polymer's
    # own resolution (import_module + getattr) to confirm it actually
    # works, same as test_ac_subprocess.py's
    # test_srf_getter_path_actually_resolves.
    mod = importlib.import_module("hypso.ac")
    assert callable(mod.ac_polymer_srf_getter)

    module_path, attr_name = mod.SRF_GETTER_PATH.rsplit(".", 1)
    resolved_mod = importlib.import_module(module_path)
    assert getattr(resolved_mod, attr_name) is mod.ac_polymer_srf_getter


def test_hypso_base_ac_wrapper_surface():
    # Every public ac_* method name confirmed used by hypso-processing-pipeline
    # must still exist on HypsoCapture (as delegating wrappers post-extraction).
    # ac_ocsmart_run_correction was removed (HypsoCapture cleanup): pipeline
    # confirmed NOT to call it or its old stage_input sibling - it replaced
    # both with its own ac_runners_hypso.run_ocsmart_correction - and its only
    # other caller was a permanently-dead branch in ac/loading_acolite_output.py
    # (TOGGLE_OCSMART = False), removed alongside it. Also removed:
    # ac_polymer_get_id_sensor/get_srf_nc_path/get_ssi_nc_path/get_esun_nc_path
    # (zero external callers; superseded by hypso.ac.adapters.PolymerAdapter +
    # SpectralResponse) - never part of this expected list.
    from hypso.HypsoCapture import HypsoCapture

    expected = [
        "ac_ocsmart_open_output",
        "ac_acolite_run_correction", "ac_acolite_open_output",
        "ac_polymer_run_correction", "ac_polymer_open_output",
        "ac_polymer_generate_srf_nc", "ac_polymer_generate_ssi_nc", "ac_polymer_generate_esun_nc",
        "ac_dark_pixel_subtraction",
    ]
    for name in expected:
        assert callable(getattr(HypsoCapture, name)), name

    for removed in ("ac_ocsmart_run_correction", "ac_polymer_get_id_sensor",
                    "ac_polymer_get_srf_nc_path", "ac_polymer_get_ssi_nc_path",
                    "ac_polymer_get_esun_nc_path"):
        assert not hasattr(HypsoCapture, removed), removed


def test_hypso_base_geo_wrapper_surface():
    # run_georeferencing (hypso-processing-pipeline) and
    # run_direct_georeferencing (hypso/ac/loading_acolite_output.py) stayed as
    # methods when their bodies moved to hypso.geo.
    from hypso.HypsoCapture import HypsoCapture

    assert callable(HypsoCapture.run_georeferencing)
    assert callable(HypsoCapture.run_direct_georeferencing)


def _bare_capture():
    """A HypsoCapture instance with no real capture data loaded - enough for
    capture_types.spawn_as() (needs _custom_masks/_l2a_cubes) and the AC
    open_output _impl dispatch (mocked below, never actually touches file
    I/O), without needing the real reference capture fixture."""
    from hypso.HypsoCapture import HypsoCapture
    from hypso.containers import DatasetDict

    obj = object.__new__(HypsoCapture)
    obj._custom_masks = DatasetDict(dim_names=('y', 'x'), num_dims=2)
    obj._l2a_cubes = DatasetDict(num_dims=3, key_attribute='correction')
    return obj


@pytest.mark.parametrize("correction", ["polymer", "acolite", "ocsmart"])
def test_to_l2a_spawns_new_object_and_dispatches_through_adapter_registry(monkeypatch, correction):
    # to_l2a() must leave self untouched (mirrors to_l1b/to_l1c/to_l1d) and
    # dispatch through the same hypso.ac.adapters registry self.ac uses
    # (get_ac_adapter), not a hardcoded per-tool if/elif.
    from hypso.ac.adapters import get_ac_adapter

    satobj = _bare_capture()
    calls = []
    monkeypatch.setattr(type(get_ac_adapter(correction)), "open_output",
                        lambda self, satobj, **kwargs: calls.append((satobj, kwargs)))

    new_obj = satobj.to_l2a(correction, some_kwarg="x")

    assert new_obj is not satobj
    assert len(calls) == 1
    called_satobj, called_kwargs = calls[0]
    assert called_satobj is new_obj
    assert called_kwargs == {"some_kwarg": "x"}


def test_to_l2a_unknown_correction_raises():
    satobj = _bare_capture()
    with pytest.raises(KeyError, match="No AC adapter registered"):
        satobj.to_l2a("6sv1")


@pytest.mark.parametrize("public_name,impl_name", [
    ("ac_polymer_open_output", "_ac_polymer_open_output_impl"),
    ("ac_acolite_open_output", "_ac_acolite_open_output_impl"),
    ("ac_ocsmart_open_output", "_ac_ocsmart_open_output_impl"),
])
def test_deprecated_open_output_wrappers_warn_and_delegate(monkeypatch, public_name, impl_name):
    satobj = _bare_capture()
    calls = []
    monkeypatch.setattr(type(satobj), impl_name,
                        lambda self, **kwargs: calls.append(kwargs) or "result")

    with pytest.warns(DeprecationWarning, match="to_l2a"):
        result = getattr(satobj, public_name)()

    assert result == "result"
    assert len(calls) == 1

"""Plan §Verification item 3 (part): the AC adapter registry exposes the
expected run_correction/open_output interface on each adapter - structural
checks only, per the plan: the actual subprocess/external-tool calls aren't
exercised (they weren't rewritten and aren't the target of this refactor's
correctness guarantees). No real data needed."""
import inspect

import pytest

from hypso.ac.adapters import (
    ACAdapter,
    AC_ADAPTERS,
    get_ac_adapter,
    registered_ac_adapters,
)


def test_registry_keys():
    assert {a.key for a in registered_ac_adapters()} == {"polymer", "acolite", "ocsmart"}


@pytest.mark.parametrize("key", ["polymer", "acolite", "ocsmart"])
def test_shared_interface(key):
    adapter = get_ac_adapter(key)
    assert isinstance(adapter, ACAdapter)
    assert callable(adapter.run_correction)
    assert callable(adapter.open_output)
    # every adapter method takes the capture object explicitly
    for name, member in inspect.getmembers(adapter, inspect.ismethod):
        if name.startswith("_"):
            continue
        params = list(inspect.signature(member).parameters)
        assert params and params[0] == "satobj", (key, name, params)


def test_namespace_matches_registry():
    for adapter in registered_ac_adapters():
        assert getattr(AC_ADAPTERS, adapter.key) is adapter


def test_unknown_tool_raises_keyerror_listing_known():
    with pytest.raises(KeyError) as exc:
        get_ac_adapter("6sv1")
    assert "polymer" in str(exc.value)


def test_hypso_base_delegates_through_registry():
    from hypso.HypsoCapture import HypsoCapture

    assert HypsoCapture.ac is AC_ADAPTERS


def test_polymer_extras():
    polymer = get_ac_adapter("polymer")
    for name in ("get_id_sensor", "get_srf_nc_path", "get_ssi_nc_path",
                 "get_esun_nc_path", "generate_srf_nc", "generate_ssi_nc",
                 "generate_esun_nc"):
        assert callable(getattr(polymer, name)), name


def test_ocsmart_extras():
    # stage_input/run_correction were merged into one run_correction
    # (confirmed zero external callers of the old two-call split) - see
    # ocsmart.py's module docstring for why staging can't safely happen
    # independently of the subprocess call.
    ocsmart = get_ac_adapter("ocsmart")
    assert callable(ocsmart.run_correction)
    assert callable(ocsmart.output_path)
    assert ocsmart.HYPSO_PREFIX == "HYPSO_HSI"


def test_polymer_open_output_rejects_unknown_input_level():
    # Regression test for a real bug fixed in this pass: open_output's
    # input_product_level match had no `case _`, so an unrecognized level
    # left polymer_l2_output_nc_file as None and the next line's .absolute()
    # call raised an unhelpful AttributeError instead of naming the problem.
    # No real data/Polymer needed - the bug is in argument validation, before
    # any file or tool is touched.
    polymer = get_ac_adapter("polymer")
    with pytest.raises(ValueError, match="Unsupported input_product_level"):
        polymer.open_output(satobj=None, input_product_level="l1a")

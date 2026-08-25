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
    from hypso.HypsoBase import HypsoBase

    assert HypsoBase.ac is AC_ADAPTERS


def test_polymer_extras():
    polymer = get_ac_adapter("polymer")
    for name in ("get_id_sensor", "get_srf_nc_path", "get_ssi_nc_path",
                 "get_esun_nc_path", "generate_srf_nc", "generate_ssi_nc",
                 "generate_esun_nc"):
        assert callable(getattr(polymer, name)), name


def test_ocsmart_extras():
    assert callable(get_ac_adapter("ocsmart").stage_input)

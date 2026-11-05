"""Guards the public API of the ntt.utils package after the split.

This split is behavior-preserving: every public name that lived in the old
``ntt/utils.py`` module must remain importable from the ``ntt.utils`` package.
"""

import importlib


def test_public_api_importable_from_package_root():
    utils = importlib.import_module("ntt.utils")
    for name in (
        "RankedLogger",
        "NTTBaseModel",
        "NTTMutableBaseModel",
        "Split",
        "SPLITS",
        "check_valid_split",
        "fields_by_json_schema_extra",
        "dict_config_to_list",
        "register_config",
        "print_config_tree",
        "log_config_tree_debug",
    ):
        assert hasattr(utils, name), f"ntt.utils.{name} missing after split"


def test_target_reference_resolves():
    # config/*.yaml uses `_target_: ntt.utils.dict_config_to_list`
    from ntt.utils import dict_config_to_list

    assert dict_config_to_list(a=1, b=None, c=2) == [1, 2]


def test_submodules_exist():
    for mod in (
        "ntt.utils.pylogger",
        "ntt.utils.base_model",
        "ntt.utils.schema",
        "ntt.utils.splits",
        "ntt.utils.config",
    ):
        importlib.import_module(mod)

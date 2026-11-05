"""Tokenizer specs dump to complete, re-loadable JSON.

Eval reports record ``tokenizer_config`` for provenance and comparability. Two
cases must work: a ``Dist``-valued field must not be dumped through the base-class
schema (``{"return_int": false}``, parameters and type lost), and an OmegaConf
container left inside a spec by Hydra must not fail the JSON dump.
"""

from __future__ import annotations

import hydra
import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf

from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec
from ntt.contrib.tokenizer.cosmos.cosmos_2d import CosmosTokenizer2DSpec
from ntt.core.components.dist import TriangleDist, UniformDist
from ntt.evaluation.utils.config_provenance import config_to_jsonable

_CALM = {"n_neurons": 128, "bin_size_ms": 1.0, "patch_length": 100}
_COSMOS_VQPT = {"bottleneck_type": "vqpt", "token_bin_size_ms": 16.0}


def _has_omegaconf(value) -> bool:
    if OmegaConf.is_config(value):
        return True
    if isinstance(value, dict):
        return any(_has_omegaconf(v) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_has_omegaconf(v) for v in value)
    return False


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (
            UniformDist(low=0.1, high=2.0),
            {"_target_": "ntt.core.components.dist.UniformDist", "low": 0.1, "high": 2.0},
        ),
        (
            TriangleDist(min_units=1, mode_units=2, max_units=4),
            {"_target_": "ntt.core.components.dist.TriangleDist", "mode_units": 2, "max_units": 4},
        ),
    ],
)
def test_dist_window_dumps_its_type_and_parameters(window, expected):
    spec = CalmTokenizerSpec(window_length_seconds=window, **_CALM)
    dumped = spec.model_dump(mode="json", by_alias=True)["window_length_seconds"]
    assert dumped.items() >= expected.items()


@pytest.mark.parametrize("mode", ["json", "python"])
def test_dist_window_round_trips_through_model_dump(mode):
    window = UniformDist(low=0.1, high=2.0, seed=3)
    spec = CalmTokenizerSpec(window_length_seconds=window, **_CALM)
    back = CalmTokenizerSpec.model_validate(spec.model_dump(mode=mode))
    assert isinstance(back.window_length_seconds, UniformDist)
    # Compare dumps: pydantic's == also compares the private ``_rng`` generator.
    assert back.window_length_seconds.model_dump() == window.model_dump()


def test_hydra_built_spec_holds_no_omegaconf_containers():
    """Hydra passes nested config values as DictConfig/ListConfig (``_convert_=none``)."""
    cfg = OmegaConf.create(
        {
            "_target_": "ntt.contrib.tokenizer.cosmos.cosmos_2d.CosmosTokenizer2DSpec",
            **_COSMOS_VQPT,
            "vqpt_kwargs": {"levels": [8, 5, 5, 5]},
        }
    )
    spec = hydra.utils.instantiate(cfg)
    assert spec.vqpt_kwargs == {"levels": [8, 5, 5, 5]}
    assert not _has_omegaconf(spec.vqpt_kwargs)


def test_config_to_jsonable_dumps_a_spec_with_a_leftover_listconfig():
    """Specs unpickled from checkpoints can still carry a ListConfig inside them."""
    spec = CosmosTokenizer2DSpec(**_COSMOS_VQPT, vqpt_kwargs={"levels": [8, 5, 5, 5]})
    spec.vqpt_kwargs["levels"] = ListConfig([8, 5, 5, 5])  # what such a checkpoint unpickles to

    dumped = config_to_jsonable(spec, name="tokenizer_config")

    assert dumped["KIND"] == "cosmos_2d"
    assert dumped["vqpt_kwargs"] == {"levels": [8, 5, 5, 5]}


def test_config_to_jsonable_still_raises_on_an_unknown_type():
    """Only OmegaConf containers get a fallback; other surprises still surface."""
    spec = CosmosTokenizer2DSpec(**_COSMOS_VQPT, vqpt_kwargs={"levels": [8, 5, 5, 5]})
    spec.vqpt_kwargs["levels"] = object()
    with pytest.raises(Exception, match="object"):
        config_to_jsonable(spec, name="tokenizer_config")


def test_config_to_jsonable_unchanged_for_plain_dictconfig():
    assert config_to_jsonable(DictConfig({"a": [1, 2]}), name="cfg") == {"a": [1, 2]}


def test_validation_converts_a_listconfig_nested_in_a_plain_dict():
    """A ``vqpt_kwargs`` plain dict may hold a ListConfig; validation converts it."""
    spec = CosmosTokenizer2DSpec.model_validate(
        {
            **_COSMOS_VQPT,
            "vqpt_kwargs": {"levels": ListConfig([8, 5, 5, 5]), "x": (DictConfig({}),)},
        }
    )

    assert not _has_omegaconf(spec.vqpt_kwargs)
    assert spec.model_dump(mode="json")["vqpt_kwargs"] == {"levels": [8, 5, 5, 5], "x": [{}]}

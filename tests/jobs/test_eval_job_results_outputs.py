"""Tests for the new top-level keys in NTTEvalJob's results.json.

Doesn't run a full eval — instantiates the job partially and calls
``_save_results`` directly. Real end-to-end eval is covered by the
existing ``tests/jobs/test_eval_all.py`` smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ntt.evaluation.compression import CompressionResult
from ntt.evaluation.config.schema import AggregationStrategy
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)


def _fake_metric(name: str, value: float, *, direction=MetricDirection.HIGHER_IS_BETTER):
    return MetricResult(
        name=name,
        category=MetricCategory.SINGLE_UNIT_STATISTICS,
        direction=direction,
        level=MetricLevel.POPULATION,
        raw_value=value,
    )


def _make_report(values: dict[str, float]) -> EvaluationReport:
    return EvaluationReport(
        results={k: _fake_metric(k, v) for k, v in values.items()},
    )


def _build_job(tmp_path: Path):
    """Construct an NTTEvalJob with the minimum private state for _save_results."""
    from ntt.jobs.eval import NTTEvalJob

    job = NTTEvalJob.model_construct(
        # ckpt_path non-None so _save_results' source-dict branch picks it.
        ckpt_path="/tmp/fake.ckpt",
        tokenizer_config=None,
        data_config=MagicMock(),
        eval_config={},
        output_path=str(tmp_path),
        aggregation_strategy=AggregationStrategy.MEAN,
        compare_poisson=False,
        poisson_n_realisations=5,
        poisson_use_all_neurons=False,
    )
    # _save_results reads these private attrs.
    job._spec = None
    job._poisson_rate_bin_ms = None
    job._compression = None
    # The provenance sidecar hard-requires a non-None
    # ``self._eval_config`` and serializes the ``data_config`` through
    # ``json.dumps``. In production ``__enter__`` sets the former and the
    # data config is a real pydantic model; ``model_construct`` skips
    # both, so we supply minimal serializable stand-ins here.
    #
    from ntt.evaluation.config.schema import EvaluationConfig

    job._eval_config = EvaluationConfig()
    # _provenance_data_config returns ``self.data_config`` which the test
    # leaves as MagicMock — not JSON-serializable. Stub the method to a
    # plain dict so the resulting ``data_config`` block in run_meta is
    # writable.
    job._provenance_data_config = lambda: {}
    return job


def test_save_results_includes_compression_block_when_set(tmp_path):
    job = _build_job(tmp_path)
    job._compression = CompressionResult(
        input_bits=800.0,
        output_bits=100.0,
        compression_rate=8.0,
        input_elements=800,
        output_elements=10,
        latent_dim_reduction=80.0,
        bits_per_dim=10.0,
        codes_per_token=1.0,
        bits_per_token=10.0,
        bits_per_neuron_per_window=200.0,
        n_neurons=4,
        t_tokens=5,
        u_units=1,
        latent_dim=2,
    )

    aggregated = _make_report({"m": 0.5})
    job._save_results(
        aggregated=aggregated,
        session_reports={},
    )

    payload = json.loads((tmp_path / "results.json").read_text())
    assert "compression" in payload
    assert payload["compression"]["compression_rate"] == 8.0
    assert payload["compression"]["bits_per_dim"] == 10.0
    assert payload["compression"]["bits_per_token"] == 10.0
    assert payload["compression"]["codes_per_token"] == 1.0
    assert payload["compression"]["n_neurons"] == 4


def _cosmos_spec_with_a_leftover_listconfig():
    from omegaconf import ListConfig

    from ntt.contrib.tokenizer.cosmos.cosmos_2d import CosmosTokenizer2DSpec

    spec = CosmosTokenizer2DSpec(
        bottleneck_type="vqpt", token_bin_size_ms=16.0, vqpt_kwargs={"levels": [8, 5, 5, 5]}
    )
    spec.vqpt_kwargs["levels"] = ListConfig(
        [8, 5, 5, 5]
    )  # as unpickled from checkpoints that hold one
    return spec


def _calm_spec_with_a_dist_window():
    from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec
    from ntt.core.components.dist import UniformDist

    return CalmTokenizerSpec(
        window_length_seconds=UniformDist(low=0.1, high=2.0),
        n_neurons=128,
        bin_size_ms=1.0,
        patch_length=100,
    )


@pytest.mark.parametrize(
    ("make_spec", "key", "expected"),
    [
        (_cosmos_spec_with_a_leftover_listconfig, "vqpt_kwargs", {"levels": [8, 5, 5, 5]}),
        (
            _calm_spec_with_a_dist_window,
            "window_length_seconds",
            {
                "_target_": "ntt.core.components.dist.UniformDist",
                "return_int": False,
                "low": 0.1,
                "high": 2.0,
                "seed": None,
            },
        ),
    ],
)
def test_save_results_records_the_full_tokenizer_config(tmp_path, make_spec, key, expected):
    """These specs must record their full config, not ``{"KIND": ...}`` only or a Dist as
    ``{"return_int": false}``."""
    job = _build_job(tmp_path)
    job._spec = make_spec()
    job._save_results(aggregated=_make_report({"m": 0.5}), session_reports={})

    for name in ("results.json", "run_meta.json"):
        saved = json.loads((tmp_path / name).read_text())["tokenizer_config"]
        assert saved[key] == expected, name


def test_save_results_serializes_nan_compression_as_null(tmp_path):
    """results.json must be RFC-8259 conformant: NaN serialized as null."""
    import math

    job = _build_job(tmp_path)
    job._compression = CompressionResult(
        input_bits=800.0,
        output_bits=math.nan,
        compression_rate=math.nan,
        input_elements=800,
        output_elements=10,
        latent_dim_reduction=80.0,
        bits_per_dim=math.nan,
        codes_per_token=1.0,
        bits_per_token=math.nan,
        bits_per_neuron_per_window=200.0,
        n_neurons=4,
        t_tokens=5,
        u_units=1,
        latent_dim=2,
    )

    job._save_results(
        aggregated=_make_report({"m": 0.5}),
        session_reports={},
    )

    raw = (tmp_path / "results.json").read_text()
    # Bare NaN/Infinity tokens are invalid per RFC 8259.
    assert "NaN" not in raw
    assert "Infinity" not in raw

    # And the values round-trip as None.
    payload = json.loads(raw)
    assert payload["compression"]["compression_rate"] is None
    assert payload["compression"]["output_bits"] is None
    assert payload["compression"]["bits_per_dim"] is None
    # Real floats survive.
    assert payload["compression"]["latent_dim_reduction"] == 80.0


def test_save_results_omits_compression_block_when_none(tmp_path):
    job = _build_job(tmp_path)
    job._compression = None  # explicit

    aggregated = _make_report({"m": 0.5})
    job._save_results(
        aggregated=aggregated,
        session_reports={},
    )

    payload = json.loads((tmp_path / "results.json").read_text())
    assert "compression" not in payload


def test_save_results_includes_generalization_when_tier_reports_present(tmp_path):
    from ntt.evaluation.tiering import TierMembership

    job = _build_job(tmp_path)

    # Two tiers — xtime baseline + one OOD.
    tier_reports = {
        "xtime": _make_report({"acc": 0.8}),
        "xsource": _make_report({"acc": 0.6}),
    }
    per_session = {
        "bsA/t": _make_report({"acc": 0.8}).to_dict(),
        "bsA/s": _make_report({"acc": 0.6}).to_dict(),
    }
    tier_map = {
        "xtime": TierMembership("xtime", exclude_brainsets={"bsA": frozenset({"s"})}),
        "xsource": TierMembership("xsource", include=frozenset({"bsA/s"})),
    }

    aggregated = _make_report({"acc": 0.7})
    job._save_results(
        aggregated=aggregated,
        session_reports={},
        tier_reports=tier_reports,
        generalization_block=job._compute_generalization(per_session, tier_map),
    )

    payload = json.loads((tmp_path / "results.json").read_text())
    assert "generalization" in payload
    gen = payload["generalization"]
    assert gen["baseline_tier"] == "xtime"
    assert gen["summary"]["n_retention_terms"] == 1
    assert gen["summary"]["mean_retention"] == pytest.approx(0.75)


def test_save_results_omits_generalization_when_no_tier_reports(tmp_path):
    job = _build_job(tmp_path)
    job._save_results(
        aggregated=_make_report({"acc": 0.7}),
        session_reports={},
        tier_reports=None,
        generalization_block=job._compute_generalization({}, None),
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    assert "generalization" not in payload


def test_save_results_saves_what_the_comparability_check_compares(tmp_path):
    """results.json saves the session ids, cross-session aggregation strategy and run
    settings that the aggregation's comparability check reads."""
    from types import SimpleNamespace

    from ntt.evaluation.aggregation.comparability import (
        input_geometry,
        recorded_settings,
        report_session_summary,
    )
    from ntt.evaluation.config.schema import EvaluationConfig

    job = _build_job(tmp_path)
    job.split = "val"
    job._spec = SimpleNamespace(
        KIND="identity",
        model_dump=lambda **_: {
            "KIND": "identity",
            "bin_size_ms": 1.0,
            "window_length_seconds": 1.0,
        },
    )
    aggregated = _make_report({"m": 0.5})
    aggregated.config = EvaluationConfig(max_eval_seconds=300.0).model_dump(mode="json")
    segment = MagicMock(original_shape=(10, 4), original=None, segment_index=0, is_partial=False)
    session_reports = {
        sid: [(segment, {"report": _make_report({"m": 0.5}), "metadata": {}})]
        for sid in ("s1", "s2")
    }
    job._save_results(aggregated=aggregated, session_reports=session_reports)

    saved = json.loads((tmp_path / "results.json").read_text())
    assert report_session_summary(saved) == {
        "session_ids": ["s1", "s2"],
        "n_sessions": 2,
        "aggregation_strategy": "mean",
    }
    settings = recorded_settings(
        saved, saved["aggregated"]["config"], input_geometry(saved["tokenizer_config"])
    )
    assert settings == {
        "split": "val",
        "bin_size_ms": 1.0,
        "window_length_seconds": 1.0,
        "scored_window_seconds": 1.0,
        "max_eval_seconds": 300.0,
        "max_windows": None,
        "max_neurons": None,
        "evaluate_all_neurons": True,
        "neuron_subsample_seed": 42,
        "conversion_bin_size_ms": 1.0,
        "metrics": None,
    }


class _UnitPatchTokenizer:
    """Stand-in tokenizer: one token per 100 bins, one latent unit per input neuron."""

    def encode(self, x):  # type: ignore[no-untyped-def]
        import torch

        n_bins, n_units = int(x.spike_matrix.shape[1]), int(x.spike_matrix.shape[2])
        return torch.zeros(1, max(1, n_bins // 100), n_units, 4)


@pytest.mark.parametrize(("configured", "expected"), [(256, 256), (None, 8)])
def test_compression_n_neurons_is_passed_to_compute_compression(tmp_path, configured, expected):
    """``compression_n_neurons`` sets the dummy population for specs without a fixed
    neuron count; unset, the dummy default applies. Either way the record states it."""
    from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizerSpec

    job = _build_job(tmp_path)
    job.compression_n_neurons = configured
    job._spec = IdentityTokenizerSpec(window_length_seconds=0.2, bin_size_ms=1.0)
    job._tokenizer = _UnitPatchTokenizer()  # type: ignore[assignment]

    job._compute_compression()

    assert job._compression is not None
    assert job._compression.n_neurons == expected


def test_run_computes_compression_before_the_evaluation(tmp_path, monkeypatch):
    """run() must call _compute_compression; nothing else writes the record."""
    import torch

    from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizerSpec

    class _Called(Exception):
        pass

    def _compute(self):
        raise _Called

    job = _build_job(tmp_path)
    job._spec = IdentityTokenizerSpec(window_length_seconds=0.2, bin_size_ms=1.0)
    job._tokenizer = _UnitPatchTokenizer()  # type: ignore[assignment]
    job._device = torch.device("cpu")
    monkeypatch.setattr(type(job), "_resolve_eval_window_seconds", lambda self: 0.2)
    monkeypatch.setattr(type(job), "_compute_compression", _compute)

    with pytest.raises(_Called):
        job.run()


@pytest.mark.parametrize("value", [0, -1])
def test_compression_n_neurons_must_be_positive(value):
    """The field itself rejects a non-positive count (checked on the field's own
    validator: building a whole job needs a checkpoint or tokenizer config)."""
    from typing import Annotated

    import pydantic

    from ntt.jobs.eval import NTTEvalJob

    field = NTTEvalJob.model_fields["compression_n_neurons"]
    adapter = pydantic.TypeAdapter(Annotated[field.annotation, *field.metadata])

    assert adapter.validate_python(256) == 256
    with pytest.raises(pydantic.ValidationError, match="greater than or equal to 1"):
        adapter.validate_python(value)


def test_save_results_records_the_firing_rate_reference(tmp_path):
    """results.json carries firing_rate_error scored against the population-mean-rate
    reference (with the Poisson floor kept) and each session's reference error."""
    from types import SimpleNamespace

    import numpy as np

    from ntt.evaluation.aggregation.comparison import (
        MetricComparison,
        with_alternative_floors,
    )
    from ntt.jobs.eval import _segment_time_bins

    original = np.array([2.0, 4.0, 6.0, 8.0])
    reconstructed = np.array([2.0, 4.0, 6.0, 10.0])
    firing_rate = MetricResult(
        name="firing_rate_error",
        category=MetricCategory.SINGLE_UNIT_STATISTICS,
        direction=MetricDirection.LOWER_IS_BETTER,
        level=MetricLevel.PER_UNIT,
        raw_value=np.abs(original - reconstructed),
        metadata={"rates_original_hz": original, "rates_reconstructed_hz": reconstructed},
        value_range=(0.0, None),
    )
    segment = SimpleNamespace(
        original_shape=(4, 100), original=None, segment_index=0, is_partial=False
    )
    session_reports = {
        "s1": [(segment, {"report": EvaluationReport(results={"firing_rate_error": firing_rate})})]
    }
    poisson = MetricComparison(
        name="firing_rate_error",
        category=MetricCategory.SINGLE_UNIT_STATISTICS,
        direction=MetricDirection.LOWER_IS_BETTER,
        level=MetricLevel.PER_UNIT,
        model_value=0.5,
        baseline_value=0.1,
        ceiling_value=0.0,
        normalized_score=-4.0,
        value_range=(0.0, None),
    )
    # As NTTEvalJob.run does after the Poisson comparison.
    comparisons, per_session = with_alternative_floors(
        {"firing_rate_error": poisson},
        {
            sid: [data["report"].results for _, data in segs]
            for sid, segs in session_reports.items()
        },
        AggregationStrategy.MEAN,
        segment_weights={
            sid: [_segment_time_bins(s) for s, _ in segs] for sid, segs in session_reports.items()
        },
    )
    job = _build_job(tmp_path)
    job._poisson_reference_kind = "inline"
    job._poisson_reference_window_seconds = None
    job._save_results(
        aggregated=EvaluationReport(results={"firing_rate_error": firing_rate}),
        session_reports=session_reports,
        comparisons_vs_poisson=comparisons,
        alternative_floor_per_session=per_session,
    )

    payload = json.loads((tmp_path / "results.json").read_text())
    saved = payload["comparison_vs_poisson"]["firing_rate_error"]
    assert saved["baseline_kind"] == "population_mean_rate"
    assert saved["baseline_value"] == pytest.approx(2.0)  # mean |[2, 4, 6, 8] - 5|
    assert saved["poisson_baseline_value"] == pytest.approx(0.1)
    assert saved["normalized_score"] == pytest.approx(0.75)  # 1 - 0.5 / 2
    assert payload["normalized_scores_vs_poisson"]["firing_rate_error"] == pytest.approx(0.75)
    session_metadata = payload["per_session"]["s1"]["results"]["firing_rate_error"]["metadata"]
    assert session_metadata["population_mean_rate_error_hz"] == pytest.approx(2.0)

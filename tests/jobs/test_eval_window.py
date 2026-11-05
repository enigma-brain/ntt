"""The eval window is explicit, recorded and checked (padded and all-neuron paths)."""

from __future__ import annotations

import json
import typing as T
from types import SimpleNamespace

import pytest

from ntt.contrib.tokenizer.ldns.ldns_tokenizer import LDNSTokenizerSpec
from ntt.core.data.components.config import SelectionConfig
from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig
from ntt.evaluation.aggregation.comparability import (
    _reference_mismatch,
    find_mixed_member_rows,
    recorded_settings,
)
from ntt.evaluation.config.schema import EvaluationConfig
from ntt.evaluation.neuron_selection import all_neuron_data_config
from ntt.jobs.ntt_base import NTTBaseJob

_SELECTIONS = [SelectionConfig(selection=[{"brainset": "toy_brainset", "sessions": ["small"]}])]
# LDNS is trained on 2.5 s windows (variable_t=True, one fixed window).
_SPEC = LDNSTokenizerSpec(window_length_seconds=2.5, bin_size_ms=10.0, n_neurons=16)


class _Job(NTTBaseJob):
    """An eval job as far as the datamodule / window handling goes."""

    KIND: T.Literal["test_eval_window"] = "test_eval_window"
    applies_eval_window: T.ClassVar[bool] = True
    eval_config: dict[str, T.Any] | None = None

    def run(self):  # type: ignore[no-untyped-def]
        return None


def _data_config() -> NTTDataModuleConfig:
    return NTTDataModuleConfig(
        root=".",
        train_selections=_SELECTIONS,
        window_length_seconds=1.0,
        bin_size_ms=10.0,
        input_type="binary_spikes",
    )


def _job(eval_window: float | str | None) -> _Job:
    eval_config = None if eval_window is None else {"eval_window_seconds": eval_window}
    return _Job(tokenizer_config=_SPEC, data_config=_data_config(), eval_config=eval_config)


@pytest.mark.parametrize(
    ("requested", "scored"), [(None, 1.0), (1.0, 1.0), ("native", 2.5), (5.0, 5.0)]
)
def test_padded_datamodule_loads_the_eval_window_and_the_record_says_so(requested, scored):
    """The padded path loads the eval window, not the spec's window."""
    with _job(requested) as job:
        assert job.datamodule.config.window_length_seconds == scored
        assert job._eval_window_record() == {
            "scored_window_seconds": scored,
            "eval_window_requested": 1.0 if requested is None else requested,
            "spec_window_length_seconds": 2.5,
        }


@pytest.mark.parametrize(("requested", "scored"), [(1.0, 1.0), ("native", 2.5)])
def test_all_neuron_datamodule_loads_the_same_window_as_the_record(requested, scored):
    from ntt.evaluation.utils.window_contract import apply_eval_window

    with _job(requested) as job:
        config = all_neuron_data_config(job.data_config, EvaluationConfig(), _SPEC)
        apply_eval_window(config, _SPEC, job._resolve_eval_window_seconds())
        dm = NTTDataModule(config)
    assert dm.config.window_length_seconds == scored
    assert job._eval_window_record()["scored_window_seconds"] == dm.config.window_length_seconds


def test_scoring_off_the_native_window_warns(caplog):
    with caplog.at_level("WARNING"):
        with _job(None):
            pass
    assert "built for 2.5 s" in caplog.text and "scored at 1 s" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        with _job("native"):
            pass
    assert "built for" not in caplog.text


def test_eval_config_accepts_native_and_rejects_nonpositive_windows():
    assert EvaluationConfig(eval_window_seconds="native").eval_window_seconds == "native"
    assert EvaluationConfig().eval_window_seconds == 1.0
    for bad in (0, -1.0, "nativ"):
        with pytest.raises(ValueError):
            EvaluationConfig(eval_window_seconds=bad)  # type: ignore[arg-type]


def test_recorded_scored_window_is_a_compared_setting():
    saved = {"scored_window_seconds": 1.0, "window_length_seconds": 4.0}
    assert recorded_settings(saved)["scored_window_seconds"] == 1.0


def _row(tokenizer: str, scored_window_seconds: float) -> dict:
    from tests.evaluation.aggregation.test_comparability import _comparable_row

    settings = json.dumps({"scored_window_seconds": scored_window_seconds})
    return _comparable_row(tokenizer, recon_settings=settings)


def test_comparability_flags_a_1s_vs_4s_scored_window():
    lines = find_mixed_member_rows([_row("perceiver", 1.0), _row("other", 4.0)])
    assert any(
        "recon_settings.scored_window_seconds" in line and "perceiver" in line for line in lines
    ), lines
    assert find_mixed_member_rows([_row("a", 4.0), _row("b", 4.0)]) == []


def test_reference_check_compares_the_scored_window_not_the_tokenizer_window():
    # The tokenizer was built for 4 s (and the reference was scored at 4 s), but the model
    # was scored at 1 s.
    recon = {"scored_window_seconds": 1.0, "tokenizer_config": {"window_length_seconds": 4.0}}
    reference = {"metadata": {"window_length_seconds": 4.0}}
    differences, _ = _reference_mismatch(recon, reference)
    assert any("window_length_seconds: model=1.0, poisson_reference=4.0" in d for d in differences)
    recon["scored_window_seconds"] = 4.0
    differences, _ = _reference_mismatch(recon, reference)
    assert not any("window_length_seconds" in d for d in differences)


def test_eval_window_record_reaches_run_meta_of_the_robustness_jobs(tmp_path):
    from ntt.jobs.eval_robustness_input import NTTEvalRobustnessInputJob

    record = {"scored_window_seconds": 2.5, "eval_window_requested": "native"}
    ri_cfg = SimpleNamespace(n_replicas=1, corruptions=[])
    meta = NTTEvalRobustnessInputJob._write_run_meta(
        tmp_path / "run_meta.json",
        ri_cfg=ri_cfg,
        skipped_counter={},
        total_pairs=0,
        eval_window=record,
    )
    assert meta["scored_window_seconds"] == 2.5
    assert (
        json.loads((tmp_path / "run_meta.json").read_text())["eval_window_requested"] == "native"
    )


# ── reports without scored_window_seconds ──────────────────────────────────


def _unrecorded_window_report(
    *, policy: str | None, window: float = 1.0, all_neurons: bool = True
) -> dict:
    """A recon results.json that does not record scored_window_seconds."""
    report: dict = {
        "evaluate_all_neurons": all_neurons,
        "tokenizer_config": {"bin_size_ms": 10.0, "window_length_seconds": 4.0},
        "aggregated": {"config": {"eval_window_seconds": window, "max_eval_seconds": 600.0}},
    }
    if policy is not None:
        report["neuron_selection"] = {"policy": policy, "sessions": {}}
    return report


def _settings_of(report: dict):
    from ntt.evaluation.aggregation.comparability import with_scored_window

    return with_scored_window(
        recorded_settings(report, report["aggregated"]["config"]),
        report,
        report["aggregated"]["config"],
    )


def test_all_neuron_report_without_the_key_gets_its_window_derived():
    settings = _settings_of(_unrecorded_window_report(policy="eval_unit_selection", window=5.0))
    assert settings["scored_window_seconds"] == 5.0


@pytest.mark.parametrize(
    "report",
    [
        _unrecorded_window_report(policy="unit_dropout_first_n"),  # padded: the spec's window
        _unrecorded_window_report(policy=None),  # no record of the units
        _unrecorded_window_report(policy="eval_unit_selection", all_neurons=False),
        _unrecorded_window_report(policy="eval_unit_selection", window="native"),  # type: ignore[arg-type]
    ],
)
def test_padded_or_unrecorded_reports_stay_unknown(report):
    assert "scored_window_seconds" not in _settings_of(report)


def test_a_recorded_window_is_not_overridden():
    report = {
        **_unrecorded_window_report(policy="eval_unit_selection"),
        "scored_window_seconds": 4.0,
    }
    assert _settings_of(report)["scored_window_seconds"] == 4.0


def test_a_window_the_aggregation_can_neither_read_nor_derive_stays_unknown():
    from ntt.jobs.aggregation import _robustness_settings

    settings = _robustness_settings({"config": {"split": "val", "scored_window_seconds": 1.0}})
    assert settings["scored_window_seconds"] == 1.0
    assert "scored_window_seconds" not in _robustness_settings({"config": {"split": "val"}})


def test_robustness_run_meta_window_is_derived_the_same_way():
    from ntt.jobs.aggregation import _robustness_settings

    run_meta = {
        "evaluate_all_neurons": True,
        "neuron_selection": {"policy": "eval_unit_selection", "sessions": {}},
        "eval_config": {"eval_window_seconds": 1.0},
    }
    settings = _robustness_settings({"config": {"split": "val"}}, run_meta)
    assert settings["scored_window_seconds"] == 1.0


def test_robustness_report_window_is_derived_from_its_config():
    """An M1/M2 report that does not record scored_window_seconds but has the window
    settings in its config gets the window derived from them."""
    from ntt.jobs.aggregation import _robustness_settings

    report = {
        "config": {
            "n_replicas": 10,
            "evaluate_all_neurons": True,
            "neuron_selection": {"policy": "eval_unit_selection"},
            "eval_window_seconds": 2.0,
        }
    }
    settings = _robustness_settings(report)
    assert settings["scored_window_seconds"] == 2.0 and settings["n_replicas"] == 10


def test_mixed_table_of_derived_1s_rows_and_a_recorded_4s_row_is_flagged_different():
    derived = _settings_of(_unrecorded_window_report(policy="eval_unit_selection"))
    rows = [_comparable_row_with(name, derived) for name in ("a", "b")]
    rows.append(_comparable_row_with("perceiver", {"scored_window_seconds": 4.0}))
    lines = find_mixed_member_rows(rows)
    differing = [ln for ln in lines if "recon_settings.scored_window_seconds" in ln]
    assert len(differing) == 1 and "perceiver" in differing[0], lines
    assert not any("could NOT check" in ln and "scored_window" in ln for ln in lines)


def _comparable_row_with(tokenizer: str, settings: dict) -> dict:
    from tests.evaluation.aggregation.test_comparability import _comparable_row

    return _comparable_row(tokenizer, recon_settings=json.dumps(settings))


def test_in_run_and_external_poisson_references_get_distinct_labels():
    from ntt.evaluation.aggregation.comparability import provenance_record

    def source(kind):
        meta = {"poisson_norm_sessions": {"source": "precomputed", "session_ids": ["s"]}}
        meta["poisson_reference_kind"] = kind
        return provenance_record({"metadata": meta})["poisson_norm_source"]

    assert source("external") != source("in_run")
    assert source("in_run") == source(None) == "precomputed by the recon eval job"


def test_record_reads_the_window_the_datamodule_loads_back():
    with _job("native") as job:
        job._scored_window_seconds = 3.0  # what the datamodule reports
        assert job._eval_window_record()["scored_window_seconds"] == 3.0


def test_the_tokenizers_own_window_is_not_compared_across_rows():
    rows = [
        _comparable_row_with(
            name, {"scored_window_seconds": 1.0, "window_length_seconds": spec_window}
        )
        for name, spec_window in (("perceiver", 4.0), ("other", 1.0))
    ]
    assert find_mixed_member_rows(rows) == []

"""max_eval_seconds is bypassed only for sessions a configured decoder can consume.

A session carrying a decoding target field (e.g. ``cursor_pos``) but not the
other fields its metric needs (e.g. ``movement_phase`` for phase filtering)
cannot be decoded, so it is capped like any other session.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ntt.core.components.types import BinnedSpikesInput
from ntt.evaluation.artifact import compute_decoding_bypass_fields
from ntt.evaluation.config import validate_config
from ntt.evaluation.core.types import MetricCategory
from ntt.evaluation.metrics.downstream_prediction import (
    BehavioralClassificationMetric,
    BehavioralRegressionMetric,
    required_behavioral_fields,
)
from ntt.evaluation.pipeline import EvaluationPipeline
from ntt.evaluation.utils.validation_collation import iter_collated_validation_segments

REPO = Path(__file__).resolve().parents[1]


def test_required_fields_follow_metric_config():
    regression = BehavioralRegressionMetric(
        target_field="cursor_pos",
        train_drop_invalid=False,
        eval_drop_invalid=False,
        eval_include_phases=["reach_period"],
    )
    assert required_behavioral_fields(regression) == {"cursor_pos", "movement_phase"}
    classification = BehavioralClassificationMetric(target_field="trial_target_id")
    assert required_behavioral_fields(classification) == {"trial_target_id", "trial_is_valid"}
    filtered = BehavioralClassificationMetric(
        target_field="trial_result", drop_invalid=False, filter_field="trial_is_valid"
    )
    assert required_behavioral_fields(filtered) == {"trial_result", "trial_is_valid"}


def test_hill_climb_decoders_all_need_movement_phase_or_their_target():
    """Shipped hill_climb decoders: every cursor regression needs movement_phase,
    so a session with cursor data but no movement_phase is not decodable."""
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=["job=poisson_baseline"])
    eval_dict = OmegaConf.to_container(cfg.job.eval_config, resolve=True)
    assert isinstance(eval_dict, dict)
    eval_config = validate_config(eval_dict)
    pipeline = EvaluationPipeline(config=eval_config)
    decoders = [m for _, m in pipeline.metrics if m.category == MetricCategory.DOWNSTREAM_DECODING]
    assert decoders
    present = {"cursor_pos", "cursor_vel", "cursor_acc", "cursor_timestamps"}  # odoherty-like
    assert not any(required_behavioral_fields(m) <= present for m in decoders)
    assert all(
        "movement_phase" in required_behavioral_fields(m)
        for m in decoders
        if getattr(m, "target_field", None) in ("cursor_pos", "cursor_vel")
    )
    entries = compute_decoding_bypass_fields(pipeline.metrics)
    assert any("+" in entry for entry in entries)
    assert all(entry == "+".join(sorted(entry.split("+"))) for entry in entries)


def _loader(fields: dict[str, bool], n_windows: int = 5, n_bins: int = 10) -> DataLoader:
    """One session of n_windows 1 s windows with the given behavioral fields."""
    batches: Any = []
    for w in range(n_windows):
        spike_matrix = torch.ones((1, n_bins, 2), dtype=torch.int32)
        behavior = {name: np.full((n_bins,), float(w)) for name, on in fields.items() if on}
        # The project collate stores behavior per sample as a list of dicts.
        per_sample_behavior: Any = [behavior]
        batches.append(
            BinnedSpikesInput(
                spike_matrix=spike_matrix,
                pad_and_gap_mask=torch.ones_like(spike_matrix),
                sampling_rate=10.0,
                unit_ids=torch.arange(2).unsqueeze(0),
                unit_ids_pad_mask=torch.ones(1, 2),
                duration_sec=torch.tensor([1.0]),
                session_id="sess",
                subject_id="subject",
                species="species",
                absolute_start=torch.tensor([float(w)], dtype=torch.float64),
                absolute_end=torch.tensor([float(w + 1)], dtype=torch.float64),
                behavioral_data=per_sample_behavior,
            )
        )
    return DataLoader(batches, batch_size=None)


def _tokenizer(batch: Any) -> dict:
    batch_size, n_bins, n_units = batch.spike_matrix.shape
    starts = (
        batch.absolute_start[:, None] + torch.arange(n_bins, dtype=torch.float64)[None, :] * 0.1
    )
    return {
        "reconstruction": batch.spike_matrix,
        "latents": torch.zeros(batch_size, n_bins, n_units, 1),
        "token_timestamps": starts + 0.05,
        "token_start_time": starts,
        "token_end_time": starts + 0.1,
    }


REGRESSION_NEEDS = frozenset({"cursor_pos", "movement_phase"})


@pytest.mark.parametrize(
    ("fields", "bypassed"),
    [
        ({"cursor_pos": True, "movement_phase": True}, True),  # consumable: full length
        ({"cursor_pos": True, "movement_phase": False}, False),  # target only: capped
    ],
)
def test_bypass_only_when_a_decoder_can_consume(fields, bypassed, caplog):
    with caplog.at_level(logging.INFO, logger="ntt.evaluation.utils.validation_collation"):
        (segment,) = iter_collated_validation_segments(
            _loader(fields),
            _tokenizer,
            token_bin_size_ms=100.0,
            device=torch.device("cpu"),
            max_eval_seconds=2.0,
            decoding_target_fields={REGRESSION_NEEDS},
        )
    assert segment.original is not None
    assert segment.original.shape[1] == (50 if bypassed else 20)
    assert segment.is_partial is not bypassed
    refused = [r for r in caplog.records if "not exempt from max_eval_seconds" in r.getMessage()]
    assert bool(refused) is not bypassed

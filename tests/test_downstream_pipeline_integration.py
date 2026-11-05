"""Integration test for downstream prediction with real data and poisson baseline tokenizer."""

import os
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ntt.contrib.tokenizer.poisson_baseline import (
    PoissonBaselineTokenizer,
    PoissonBaselineTokenizerSpec,
)
from ntt.core.components.types import BinnedSpikesInput, DataType
from ntt.core.data.components.transforms import DataToModelInput
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.metrics.downstream_prediction import (
    BehavioralClassificationMetric,
    BehavioralRegressionMetric,
)
from ntt.evaluation.utils.validation_collation import (
    CollatedValidationSegment,
    run_validation_evaluation_per_session,
)


@pytest.fixture
def poisson_tokenizer():
    """Create a simple poisson baseline tokenizer for testing."""
    spec = PoissonBaselineTokenizerSpec(
        window_length_seconds=1.0,
        bin_size_ms=100.0,  # Data loaded at 100ms bins
        rate_estimation_bin_ms=100.0,  # Internal rate computation at 100ms (no pooling)
    )
    return PoissonBaselineTokenizer(spec)


def create_synthetic_dataloader_with_behavioral_data():
    """Create a synthetic dataloader that mimics brainsets data with behavioral labels.

    Each session (batch) contains multiple trials with different target_ids and
    cursor positions, so that downstream behavioral metrics (classification,
    regression) have sufficient within-session variation for cross-validation.

    Returns:
        DataLoader yielding BinnedSpikesInput with behavioral_data
    """
    import numpy as np

    rng = np.random.RandomState(42)
    n_sessions = 3
    n_trials_per_session = 6  # 3 target classes × 2 trials each
    bins_per_trial = 10  # 1 second at 10 Hz
    n_time_bins = n_trials_per_session * bins_per_trial  # 60 bins = 6 seconds
    n_units = 20

    batches = []
    for session_idx in range(n_sessions):
        # Build per-trial target_ids and cursor positions
        trial_target_ids = np.tile([0, 1, 2], 2)  # [0,1,2,0,1,2]
        rng.shuffle(trial_target_ids)

        target_per_bin = np.repeat(trial_target_ids, bins_per_trial)
        trial_id_per_bin = np.repeat(np.arange(n_trials_per_session), bins_per_trial)

        # Cursor position: ramp within each trial, offset by target
        cursor_pos = np.zeros((n_time_bins, 1))
        for trial_i in range(n_trials_per_session):
            t_start = trial_i * bins_per_trial
            t_end = t_start + bins_per_trial
            base = trial_target_ids[trial_i] * 10.0
            cursor_pos[t_start:t_end, 0] = np.linspace(base, base + 5.0, bins_per_trial)

        # Spike matrix: random spikes modulated by target
        spike_matrix = torch.randint(0, 5, (1, n_time_bins, n_units))
        spike_matrix += torch.from_numpy(target_per_bin).unsqueeze(0).unsqueeze(-1)

        behavioral_data = [
            {
                "trial_target_id": target_per_bin,
                "cursor_pos": cursor_pos,
                "trial_id": trial_id_per_bin,
                "trial_is_valid": np.ones(n_time_bins, dtype=bool),
            }
        ]

        duration = n_time_bins / 10.0
        abs_start = session_idx * duration
        batch = BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix),
            sampling_rate=10.0,  # 10 Hz = 100ms bins
            unit_ids=torch.arange(n_units).unsqueeze(0),
            unit_ids_pad_mask=torch.ones(1, n_units),
            duration_sec=torch.tensor([duration]),
            session_id=f"synthetic_session_{session_idx}",
            subject_id="synthetic_subject",
            species="synthetic",
            absolute_start=torch.tensor([abs_start], dtype=torch.float64),
            absolute_end=torch.tensor([abs_start + duration], dtype=torch.float64),
            behavioral_data=behavioral_data,
        )
        batches.append(batch)

    return DataLoader(batches, batch_size=None)


def test_poisson_tokenizer_with_behavioral_data(poisson_tokenizer):
    """Test that poisson baseline tokenizer passes through behavioral data and enables downstream
    metrics."""

    # Create dataloader with behavioral data
    dataloader = create_synthetic_dataloader_with_behavioral_data()

    # Test a single forward pass to ensure behavioral data is accessible
    for batch in dataloader:
        # Check behavioral data is in batch (list of dicts, one per batch item)
        assert batch.behavioral_data is not None
        assert len(batch.behavioral_data) == 1  # batch_size=1
        assert "trial_target_id" in batch.behavioral_data[0]
        assert "cursor_pos" in batch.behavioral_data[0]

        # Run tokenizer
        output = poisson_tokenizer(batch)

        # Check output structure
        assert output.tokens is not None
        assert output.reconstruction is not None
        assert output.tokens.shape[0] == batch.spike_matrix.shape[0]  # Batch preserved

        # Behavioral data should still be in the original batch
        # (tokenizer doesn't consume or modify it)
        assert batch.behavioral_data is not None

        break  # Just test first batch


def test_downstream_metrics_with_poisson_tokenizer_synthetic(poisson_tokenizer):
    """Test full pipeline: dataloader -> poisson baseline tokenizer -> downstream metrics.

    Note: This test verifies that the full pipeline works (tokenizer -> collation
    -> evaluation) and that latents and behavioral_data propagate correctly.
    The downstream prediction metrics are tested separately with artificial data
    in test_downstream_metrics.py.
    """

    # Create evaluation config (uses default metrics)
    eval_config = EvaluationConfig()

    # Create dataloader
    dataloader = create_synthetic_dataloader_with_behavioral_data()

    # Run validation evaluation per session
    # This tests the full pipeline including latent and behavioral data extraction
    reports = run_validation_evaluation_per_session(
        dataloader=dataloader,
        tokenizer_fun=partial(poisson_tokenizer.reconstruct_from_input, return_latents=True),
        data_type=DataType.RATES,  # Rate tokenizer outputs rates
        bin_size_seconds=0.1,  # 100ms bins
        eval_config=eval_config,
        token_bin_size_ms=poisson_tokenizer.spec.token_bin_size_ms,
    )

    # Check that we got reports for all sessions
    assert len(reports) == 3  # 3 synthetic sessions

    # Check that each session has results
    for _session_id, session_reports in reports.items():
        assert len(session_reports) > 0

        for segment, report_data in session_reports:
            assert isinstance(segment, CollatedValidationSegment)

            # Note: segment.latents and segment.reconstructed are intentionally
            # cleared to None by run_validation_evaluation_per_session after
            # metrics are computed (memory release). Check that behavioral
            # metrics actually ran and produced finite results — that proves
            # latents were present and the downstream pipeline worked end-to-end.
            report = report_data["report"]
            behavioral_results = {k: v for k, v in report.results.items() if "behavioral" in k}
            assert len(behavioral_results) > 0, (
                f"No behavioral metrics in report.results. "
                f"Keys: {sorted(report.results.keys())}. "
                f"Skipped: {report.skipped_metrics}"
            )
            for metric_name, result in behavioral_results.items():
                assert np.isfinite(result.raw_value), (
                    f"Behavioral metric '{metric_name}' has non-finite value: {result.raw_value}"
                )

            # behavioral_data fields are NOT cleared after eval, so we can
            # check them directly.
            assert segment.behavioral_data_token_res is not None
            assert "trial_target_id" in segment.behavioral_data_token_res
            assert segment.behavioral_data_input_res is not None
            assert "cursor_pos" in segment.behavioral_data_input_res


@pytest.mark.parametrize(
    "kwargs, eval_config_kwargs, expected_param",
    [
        ({"max_windows": 10}, {}, "max_windows"),
        ({"max_bytes": 1_000}, {}, "max_bytes"),
        ({}, {"max_windows": 10}, "max_windows"),
        ({}, {"max_bytes": 1_000}, "max_bytes"),
    ],
)
def test_downstream_metrics_reject_session_splitting_caps(
    poisson_tokenizer, kwargs, eval_config_kwargs, expected_param
):
    """Downstream decoding must run on full collated sessions, so caps that
    would split a session into multiple segments (max_windows / max_bytes,
    whether passed as kwarg or via EvaluationConfig) must be rejected.

    max_eval_seconds is handled per-session — see
    test_max_eval_seconds_bypassed_for_decoding_capable_sessions.
    """
    dataloader = create_synthetic_dataloader_with_behavioral_data()

    with pytest.raises(ValueError, match=expected_param):
        run_validation_evaluation_per_session(
            dataloader=dataloader,
            tokenizer_fun=partial(poisson_tokenizer.reconstruct_from_input, return_latents=True),
            data_type=DataType.RATES,
            bin_size_seconds=0.1,
            eval_config=EvaluationConfig(**eval_config_kwargs),
            token_bin_size_ms=poisson_tokenizer.spec.token_bin_size_ms,
            **kwargs,
        )


def test_max_eval_seconds_bypassed_for_decoding_capable_sessions(poisson_tokenizer):
    """max_eval_seconds must NOT truncate sessions whose behavioral_data carries
    a target field used by a registered downstream decoding metric.

    The synthetic dataloader produces 6-second sessions with cursor_pos and
    trial_target_id. With max_eval_seconds=2.0 and default metrics (which
    include behavioral_regression on cursor_pos), the per-session cap should
    be silently bypassed and the resulting segments should span the full
    session duration.
    """
    dataloader = create_synthetic_dataloader_with_behavioral_data()

    reports = run_validation_evaluation_per_session(
        dataloader=dataloader,
        tokenizer_fun=partial(poisson_tokenizer.reconstruct_from_input, return_latents=True),
        data_type=DataType.RATES,
        bin_size_seconds=0.1,
        eval_config=EvaluationConfig(max_eval_seconds=2.0),
        token_bin_size_ms=poisson_tokenizer.spec.token_bin_size_ms,
    )

    assert len(reports) == 3, f"Expected 3 sessions, got {len(reports)}"
    for session_id, session_reports in reports.items():
        assert len(session_reports) == 1, (
            f"Session {session_id} should produce exactly one segment when the "
            f"cap is bypassed; got {len(session_reports)}."
        )
        segment, _report_data = session_reports[0]
        assert not segment.is_partial, (
            f"Session {session_id} was flagged as partial despite carrying "
            f"decoding-relevant behavioral fields; max_eval_seconds was not "
            f"bypassed as expected."
        )


# Possible data root locations to try
_DATA_ROOT_CANDIDATES = [
    os.environ.get("NTT_TEST_DATA_ROOT", ""),
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "data", "brainsets", "processed"),
]


def _find_data_root():
    """Find the first existing data root from candidates."""
    for candidate in filter(None, _DATA_ROOT_CANDIDATES):
        path = Path(os.path.expanduser(candidate))
        if path.exists():
            return str(path)
    return None


_DATA_ROOT = _find_data_root()


@pytest.mark.skipif(
    _DATA_ROOT is None,
    reason=f"No data root found. Tried: {_DATA_ROOT_CANDIDATES}",
)
@pytest.mark.slow
def test_full_pipeline_with_real_data_snapshot():
    """Snapshot test with a small slice of real Perich-Miller data.

    This test verifies that:
    1. Real brainsets data loads correctly with behavioral data
    2. Rate tokenizer processes it without errors
    3. Downstream metrics can access behavioral fields
    4. Pipeline produces valid (non-error) results

    Note: We don't check specific metric values since we're just testing
    that the pipeline works end-to-end without errors.
    """
    import yaml

    from ntt.core.data import NTTDataModule, NTTDataModuleConfig
    from ntt.core.data.components.config import DataLoaderConfig, SelectionConfig

    # Data root where processed Perich-Miller data exists
    DATA_ROOT = _DATA_ROOT

    # Load validation selection config
    selection_config_path = "config/data/selections/perich_miller_sample_val.yaml"
    with open(selection_config_path) as f:
        val_selection_dict = yaml.safe_load(f)
    val_selection = SelectionConfig(**val_selection_dict)

    # Create poisson baseline tokenizer spec
    # - bin_size_ms=1.0: data loaded at 1ms resolution for proper evaluation
    # - rate_estimation_bin_ms=100.0: internal rate computation at 100ms
    poisson_spec = PoissonBaselineTokenizerSpec(
        window_length_seconds=1.0,
        bin_size_ms=1.0,  # Data at 1ms for proper evaluation
        rate_estimation_bin_ms=100.0,  # Internal rate computation at 100ms
    )

    # Create datamodule config
    datamodule_config = NTTDataModuleConfig(
        root=DATA_ROOT,
        tokenizer_spec=poisson_spec,
        train_selections=[val_selection],  # Use same for train (not used in this test)
        val_selections=[val_selection],
        dataloader=DataLoaderConfig(
            batch_size=1,  # Batch size of 1
            num_workers=0,  # No multiprocessing for test
        ),
        seed=42,
    )

    # Create and setup datamodule
    datamodule = NTTDataModule(datamodule_config)
    datamodule.setup("validate")  # only the val loader is used

    # Get a bounded slice of the validation dataloader without splitting
    # sessions via max_windows. A very short slice (e.g. 5 one-second windows)
    # can stay within a single target/trial regime and cause all behavioral
    # metrics to skip. Using 600 windows (~10 minutes at 1 s/window) keeps
    # this as a bounded slow test while giving the main eval path enough data
    # for behavioral classification to run on the real snapshot.
    val_loader = islice(datamodule.val_dataloader(), 600)

    # Create poisson baseline tokenizer
    poisson_tokenizer = PoissonBaselineTokenizer(poisson_spec)

    # Create evaluation config (disable baseline for speed)
    eval_config = EvaluationConfig()

    # Run validation evaluation per session
    reports = run_validation_evaluation_per_session(
        dataloader=val_loader,
        tokenizer_fun=partial(poisson_tokenizer.reconstruct_from_input, return_latents=True),
        data_type=DataType.SPIKES,  # Original and reconstructed are spike counts
        bin_size_seconds=0.001,  # 1ms bins (matches data loading)
        eval_config=eval_config,
        token_bin_size_ms=poisson_spec.token_bin_size_ms,
    )

    # Verify we got reports
    assert len(reports) > 0, "Expected at least one session in reports"

    # Check each session's results
    for session_id, session_reports in reports.items():
        assert len(session_reports) > 0, f"Expected at least one segment for session {session_id}"

        for segment, report_data in session_reports:
            assert isinstance(segment, CollatedValidationSegment)

            # Note: segment.latents is intentionally cleared to None by
            # run_validation_evaluation_per_session after metrics are computed
            # (memory release). Check that behavioral metrics actually ran
            # and produced finite results.
            report = report_data["report"]
            behavioral_results = {k: v for k, v in report.results.items() if "behavioral" in k}
            assert len(behavioral_results) > 0, (
                f"No behavioral metrics for session {session_id}. "
                f"Keys: {sorted(report.results.keys())}. "
                f"Skipped: {report.skipped_metrics}"
            )
            for metric_name, result in behavioral_results.items():
                assert np.isfinite(result.raw_value), (
                    f"Behavioral metric '{metric_name}' has non-finite value: "
                    f"{result.raw_value} (session {session_id})"
                )

            # Verify behavioral data was propagated (both resolutions)
            assert segment.behavioral_data_input_res is not None, (
                f"Input-res behavioral data missing for session {session_id}"
            )

            # Check for expected Perich-Miller behavioral fields
            # Based on BEHAVIORAL_EXTRACTION_CONFIG for perich_miller_population_2018
            expected_fields = ["cursor_pos", "trial_target_id"]
            for field in expected_fields:
                assert field in segment.behavioral_data_input_res, (
                    f"Expected behavioral field '{field}' missing from input-res data "
                    f"for session {session_id}"
                )

            # Verify report exists and has results
            assert "report" in report_data
            report = report_data["report"]
            # Should have some metric results (not checking specific values)
            assert len(report.results) > 0 or len(report.skipped_metrics) > 0, (
                f"Expected some metrics to be computed or skipped for session {session_id}"
            )

    print(f"✓ Pipeline test passed with {len(reports)} session(s)")

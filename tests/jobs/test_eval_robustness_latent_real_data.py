"""Real-data integration checks for the actual latent_robustness (M2) windowed scoring path.

These tests need processed brainsets; they skip when no data root is found (see
``NTT_TEST_DATA_ROOT`` in tests/test_metric_windowed_equivalence_real_data.py).
"""

from __future__ import annotations

import csv
import json
from functools import partial
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ntt.contrib.tokenizer.compressed_sensing import (
    CompressedSensingTokenizer,
    CompressedSensingTokenizerSpec,
)
from ntt.contrib.tokenizer.poisson_baseline import PoissonBaselineTokenizer
from ntt.core.data import NTTDataModule, NTTDataModuleConfig
from ntt.core.data.components.config import DataLoaderConfig, SelectionConfig
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.config.schema import RobustnessLatentConfig
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.robustness_latent import evaluate_pair
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig
from ntt.evaluation.pipeline import EvaluationPipeline
from ntt.evaluation.utils.metric_selection import is_latent_robustness_reconstruction_metric
from ntt.evaluation.utils.validation_collation import (
    _concatenate_segments,
    iter_collated_validation_segments,
)
from ntt.jobs.eval_robustness_latent import NTTEvalRobustnessLatentJob
from tests.test_metric_windowed_equivalence import EXACT_METRICS
from tests.test_metric_windowed_equivalence_real_data import (
    _DATA_ROOT,
    _DATA_ROOT_CANDIDATES,
    MAIN_WINDOW_COUNT,
    SESSION_BRAINSET,
    SESSION_ID,
    SESSION_NAME,
    TOKENIZER_SPEC,
    RealWindowSession,
    real_perich_miller_window_session,
)

NOISY_EPSILON = 0.03
NOISY_N_SAMPLES = 3

# Deterministic Compressed-Sensing tokenizer at a 1 s window (no chunking wrapper
# needed since native window == latent_robustness (M2) eval window). Used for the eps=0 exact-match
# test: CS reconstruction is deterministic (seeded sensing matrix + FISTA, no
# per-call sampling), so the windowed latent_robustness (M2) eps=0 reconstruction reproduces the
# clean
# reconstruction exactly — unlike the stochastic Poisson baseline. Few windows +
# reduced FISTA iterations keep it fast; neither affects the eps=0 identity.
CS_WINDOW_COUNT = 8
CS_TOKENIZER_SPEC = CompressedSensingTokenizerSpec(
    window_length_seconds=1.0,
    bin_size_ms=1.0,
    n_neurons=10,
    compress_mode="per_neuron",
    latent_dim=10,
    trainable=False,
    random_seed=0,
    solver="fista",
    fista_iterations=100,
    fista_lambda=0.01,
    nonneg=True,
)


def _build_job_eval_data(job: NTTEvalRobustnessLatentJob, tensor: torch.Tensor) -> EvalData:
    bin_size_seconds = float(job._spec.bin_size_ms) / 1000.0
    return EvalData.from_tensor(
        tensor=tensor,
        data_type=job._spec.input_type,
        bin_size_seconds=bin_size_seconds,
        duration=tensor.shape[-1] * bin_size_seconds,
    )


def _make_real_data_job(
    output_path: Path,
    *,
    epsilons: list[float],
    n_eval_windows: int,
    n_samples_per_epsilon: int,
) -> NTTEvalRobustnessLatentJob:
    job = NTTEvalRobustnessLatentJob.model_construct(
        output_path=str(output_path),
        generate_plots=False,
        split="val",
        seed=0,
        data_config=SimpleNamespace(deterministic_transforms=True),
        mlflow_logging=MLflowLoggingConfig(enabled=False),
    )
    job._device = torch.device("cpu")
    job._spec = TOKENIZER_SPEC
    job._tokenizer = PoissonBaselineTokenizer(TOKENIZER_SPEC)
    job._eval_config = EvaluationConfig(
        compute_baseline=False,
        max_eval_seconds=None,
        robustness_latent=RobustnessLatentConfig(
            epsilons=epsilons,
            n_samples_per_epsilon=n_samples_per_epsilon,
            n_eval_windows=n_eval_windows,
            plot_examples=False,
        ),
    )
    return job


def _collect_selected_segments(
    job: NTTEvalRobustnessLatentJob,
    real_session: RealWindowSession,
):
    assert job._eval_config.robustness_latent is not None
    return job._collect_selected_window_segments(
        dataloader=real_session.raw_batches,
        tokenizer_fn=real_session.tokenizer_fun,
        rl_cfg=job._eval_config.robustness_latent,
        spec=job._spec,
        device=job._device,
        bin_size_seconds=float(job._spec.bin_size_ms) / 1000.0,
    )


def _make_real_data_cs_job(
    output_path: Path,
    *,
    epsilons: list[float],
    n_eval_windows: int,
    n_samples_per_epsilon: int,
) -> NTTEvalRobustnessLatentJob:
    """Like :func:`_make_real_data_job` but with the deterministic CS tokenizer."""
    job = NTTEvalRobustnessLatentJob.model_construct(
        output_path=str(output_path),
        generate_plots=False,
        split="val",
        seed=0,
        data_config=SimpleNamespace(deterministic_transforms=True),
        mlflow_logging=MLflowLoggingConfig(enabled=False),
    )
    job._device = torch.device("cpu")
    job._spec = CS_TOKENIZER_SPEC
    job._tokenizer = CompressedSensingTokenizer(CS_TOKENIZER_SPEC)
    job._eval_config = EvaluationConfig(
        compute_baseline=False,
        max_eval_seconds=None,
        robustness_latent=RobustnessLatentConfig(
            epsilons=epsilons,
            n_samples_per_epsilon=n_samples_per_epsilon,
            n_eval_windows=n_eval_windows,
            plot_examples=False,
        ),
    )
    return job


def _compute_contiguous_clean_expected_scores(
    job: NTTEvalRobustnessLatentJob,
    selected_segments: list,
) -> dict[str, float]:
    """Reference scores from a contiguous clean-vs-clean comparison.

    For a deterministic tokenizer this is what the integrated latent_robustness (M2) path must
    reproduce at eps=0 (corrupted reconstruction == clean reconstruction).
    """
    contiguous_segment = _concatenate_segments(selected_segments)
    assert contiguous_segment.reconstructed is not None
    clean_recon_t = contiguous_segment.reconstructed

    clean_recon_data = _build_job_eval_data(job, clean_recon_t)
    clean_cache = job._build_clean_cache(
        clean_original=clean_recon_data,
        clean_reconstructed=clean_recon_data,
        segment=contiguous_segment,
        sampling_rate=job._extract_sampling_rate(contiguous_segment),
        token_bin_size_ms=job._spec.token_bin_size_ms,
    )
    full_pipeline = EvaluationPipeline(config=job._eval_config)
    recon_metrics = [
        (display_name, metric)
        for display_name, metric in full_pipeline.metrics
        if is_latent_robustness_reconstruction_metric(metric.name, metric.category)
    ]

    pair_results = evaluate_pair(
        clean=clean_recon_data,
        noisy=clean_recon_data,
        metrics=recon_metrics,
        config=job._eval_config,
        shared_cache=clean_cache,
        return_cache=False,
    )

    scores: dict[str, float] = {}
    for metric_name, entry in pair_results.items():
        if entry.get("status") != "ok":
            continue
        scalar = job._metric_result_scalar(entry["result"])
        if scalar is not None:
            scores[metric_name] = float(scalar)

    scores["mse_match_miss"] = float(job._compute_mse_match_miss(clean_recon_t, clean_recon_t))
    return scores


@pytest.fixture(scope="module")
def real_perich_cs_window_session() -> SimpleNamespace:
    """First ``CS_WINDOW_COUNT`` perich windows + a deterministic CS tokenizer_fn.

    Lighter than ``real_perich_miller_window_session`` (only the pieces the eps=0
    job test consumes: ``raw_batches`` and ``tokenizer_fun``). CS native window is
    1 s == the latent_robustness (M2) eval window, so no chunking wrapper is needed.
    """
    assert _DATA_ROOT is not None
    selection = SelectionConfig(
        selection=[{"brainset": SESSION_BRAINSET, "sessions": [SESSION_NAME]}],
        config={"ignore_splits": True},
    )
    datamodule = NTTDataModule(
        NTTDataModuleConfig(
            root=_DATA_ROOT,
            tokenizer_spec=CS_TOKENIZER_SPEC,
            train_selections=[selection],
            val_selections=[selection],
            dataloader=DataLoaderConfig(batch_size=1, num_workers=0, pin_memory=False),
            seed=42,
        )
    )
    datamodule.setup("fit")
    raw_batches = list(islice(datamodule.val_dataloader(), CS_WINDOW_COUNT))
    assert len(raw_batches) == CS_WINDOW_COUNT, (
        f"Expected {CS_WINDOW_COUNT} validation windows from {SESSION_ID}, got {len(raw_batches)}."
    )
    tokenizer = CompressedSensingTokenizer(CS_TOKENIZER_SPEC)
    tokenizer_fun = partial(tokenizer.reconstruct_from_input, return_latents=True)
    return SimpleNamespace(raw_batches=raw_batches, tokenizer_fun=tokenizer_fun)


def _expected_metric_names(job: NTTEvalRobustnessLatentJob) -> set[str]:
    full_pipeline = EvaluationPipeline(config=job._eval_config)
    # Display names, not class names: the job keys its rows by the configured
    # display name so repeated metric classes stay distinct.
    recon_metrics = [
        display_name
        for display_name, metric in full_pipeline.metrics
        if is_latent_robustness_reconstruction_metric(metric.name, metric.category)
        and metric.aggregation_fallback_policy != "contiguous_only"
    ]
    return set(recon_metrics) | {"mse_match_miss"}


def _run_job_and_read_rows(
    job: NTTEvalRobustnessLatentJob,
    real_session: RealWindowSession,
    monkeypatch,
    *,
    epsilon: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    monkeypatch.setattr(
        job,
        "_build_eval_loader_and_tokenizer_fn",
        # Mirror the real keyword-only signature; the fixed real batches do not
        # depend on the requested eval window length, so it is accepted and ignored.
        lambda *, eval_window_length_seconds=None: (
            real_session.raw_batches,
            real_session.tokenizer_fun,
        ),
    )
    result_locals = job.run()

    rows = [
        json.loads(line)
        for line in (Path(job.output_path) / "results.jsonl").read_text().splitlines()
    ]
    session_rows = [
        row
        for row in rows
        if row["session_id"] == SESSION_ID and float(row["epsilon"]) == float(epsilon)
    ]
    assert session_rows, "Integrated latent_robustness (M2) run did not emit any session rows."
    assert all("k" not in row for row in session_rows)
    assert all("segment_index" not in row for row in session_rows)
    return session_rows, result_locals


@pytest.mark.skipif(
    _DATA_ROOT is None,
    reason=f"No data root found. Tried: {_DATA_ROOT_CANDIDATES}",
)
@pytest.mark.slow
def test_m2_job_cs_zero_epsilon_matches_contiguous_session_on_real_data(
    tmp_path: Path,
    monkeypatch,
    real_perich_cs_window_session: SimpleNamespace,
) -> None:
    # Deterministic-tokenizer counterpart of the eps=0 exact-match check (which is
    # obsolete for the stochastic Poisson baseline, where eps=0 is now an independent
    # draw). CS reconstruction is deterministic, so at eps=0 the corrupted
    # reconstruction equals the clean reconstruction and the windowed latent_robustness (M2) path
    # must
    # reproduce the contiguous clean-vs-clean reference exactly.
    job = _make_real_data_cs_job(
        tmp_path / "m2_cs_exact",
        epsilons=[0.0],
        n_eval_windows=CS_WINDOW_COUNT,
        n_samples_per_epsilon=1,
    )
    selected_segments, _session_order, processed_window_ids = _collect_selected_segments(
        job,
        real_perich_cs_window_session,
    )
    assert processed_window_ids == {SESSION_ID: list(range(CS_WINDOW_COUNT))}

    expected_scores = _compute_contiguous_clean_expected_scores(job, selected_segments)
    session_rows, _result_locals = _run_job_and_read_rows(
        job,
        real_perich_cs_window_session,
        monkeypatch,
        epsilon=0.0,
    )
    actual_scores: dict[str, float] = {}
    for row in session_rows:
        if row["status"] != "ok":
            continue
        metadata = row.get("metadata", {})
        assert metadata.get("n_samples_requested") == 1
        assert metadata.get("n_samples_ok") == 1
        assert metadata.get("n_samples_skipped") == 0
        assert metadata.get("score_std") == pytest.approx(0.0)
        actual_scores[str(row["metric"])] = float(row["raw_value"])

    exact_metric_names = {name for name, _factory in EXACT_METRICS}
    comparable = sorted(exact_metric_names & expected_scores.keys() & actual_scores.keys())
    assert comparable, (
        "No exact latent_robustness (M2) metrics were comparable on the CS real-data job path."
    )
    for metric_name in comparable:
        np.testing.assert_allclose(
            actual_scores[metric_name],
            expected_scores[metric_name],
            rtol=1e-6,
            atol=1e-9,
            err_msg=(
                f"Integrated latent_robustness (M2) eps=0 path for {metric_name} should match the "
                f"contiguous "
                "clean-session computation for the deterministic CS tokenizer."
            ),
        )


@pytest.mark.skipif(
    _DATA_ROOT is None,
    reason=f"No data root found. Tried: {_DATA_ROOT_CANDIDATES}",
)
@pytest.mark.slow
def test_m2_job_noisy_real_data_emits_one_row_per_metric_and_finite_replica_spread(
    tmp_path: Path,
    monkeypatch,
    real_perich_miller_window_session: RealWindowSession,
) -> None:
    job = _make_real_data_job(
        tmp_path / "m2_job_noisy",
        epsilons=[NOISY_EPSILON],
        n_eval_windows=MAIN_WINDOW_COUNT,
        n_samples_per_epsilon=NOISY_N_SAMPLES,
    )
    selected_segments, _session_order, processed_window_ids = _collect_selected_segments(
        job,
        real_perich_miller_window_session,
    )
    assert processed_window_ids == {SESSION_ID: list(range(MAIN_WINDOW_COUNT))}

    session_rows, result_locals = _run_job_and_read_rows(
        job,
        real_perich_miller_window_session,
        monkeypatch,
        epsilon=NOISY_EPSILON,
    )
    assert len(result_locals["csv_rows"]) == NOISY_N_SAMPLES

    expected_metric_names = _expected_metric_names(job)
    assert {str(row["metric"]) for row in session_rows} == expected_metric_names
    assert len(session_rows) == len(expected_metric_names)

    ok_rows = [row for row in session_rows if row["status"] == "ok"]
    assert ok_rows, (
        "Integrated latent_robustness (M2) noisy run produced no successful metric rows."
    )
    for row in ok_rows:
        metadata = row.get("metadata", {})
        assert metadata.get("n_samples_requested") == NOISY_N_SAMPLES
        assert 0 < int(metadata.get("n_samples_ok", 0)) <= NOISY_N_SAMPLES
        assert metadata.get("n_samples_skipped") == (
            NOISY_N_SAMPLES - int(metadata.get("n_samples_ok", 0))
        )
        assert np.isfinite(float(metadata["score_std"]))

    mse_row = next(row for row in session_rows if row["metric"] == "mse_match_miss")
    assert mse_row["status"] == "ok"
    assert np.isfinite(float(mse_row["metadata"]["score_std"]))

    with (Path(job.output_path) / "results.csv").open() as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 1
    assert np.isfinite(float(csv_rows[0]["mse_match_miss_std"]))

"""End-to-end smoke test for the latent_robustness (M2) job pipeline.

Exercises ``NTTEvalRobustnessLatentJob._write_csv`` /
``_write_run_meta`` and the eval-time figure emission
(``_emit_figures``) against synthetic inputs so the test stays fast
and model-free, mirroring the structure of
``tests/jobs/test_eval_robustness_input.py``.

The smoke focuses on the contract that the job leaves a
``results.csv`` plus a complete ``curves/`` and ``rasters/`` PNG/SVG/PDF
triplet on disk when ``plot_examples=True``.
"""

from __future__ import annotations

import csv
import json

import matplotlib

# Headless backend — the tests run on CI / cluster nodes without a display.
matplotlib.use("Agg")

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch_brain.data.dataset import DatasetIndex

from ntt.contrib.tokenizer.pca_baseline import (
    PCABaselineTokenizer,
    PCABaselineTokenizerSpec,
)
from ntt.contrib.tokenizer.poisson_baseline import (
    PoissonBaselineTokenizer,
    PoissonBaselineTokenizerSpec,
)
from ntt.contrib.tokenizer.wanneural.wanneural import (
    WanNeuralTokenizer,
    WanNeuralTokenizerSpec,
)
from ntt.core.data.datamodule import collate
from ntt.evaluation.aggregation.comparability import report_session_summary
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.config.schema import RobustnessLatentConfig
from ntt.evaluation.core.types import DataType, MetricCategory, MetricDirection, MetricLevel
from ntt.evaluation.metrics.base import BaseMetric, IntermediateRequest
from ntt.evaluation.metrics.robustness_latent_stats import LatentStats
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig
from ntt.evaluation.neuron_selection import NeuronSelection
from ntt.jobs.eval_robustness_latent import NTTEvalRobustnessLatentJob
from tests.fixtures import WindowedSpikeDataset


def _make_job_skeleton() -> NTTEvalRobustnessLatentJob:
    """Construct a minimally-validated job instance for unit-testing
    helper methods without going through ``__enter__``.

    We only need ``_emit_figures`` bound to ``self``; that method does
    not touch the tokenizer or datamodule. ``model_construct`` skips
    validation so we don't need to instantiate a real spec.
    """
    return NTTEvalRobustnessLatentJob.model_construct()


class _FakeLatentTokenizer:
    def __init__(self, spec):
        self.spec = spec

    def reconstruct(self, latents: torch.Tensor):
        return SimpleNamespace(reconstruction=latents)


class _RecordingFakeLatentTokenizer(_FakeLatentTokenizer):
    def __init__(self, spec):
        super().__init__(spec)
        self.decode_batch_sizes: list[int] = []

    def reconstruct(self, latents: torch.Tensor):
        self.decode_batch_sizes.append(int(latents.shape[0]))
        return super().reconstruct(latents)


class _StrictFixedInputChunkTokenizer:
    def __init__(self):
        self.spec = SimpleNamespace(
            KIND="fixed_input_test",
            variable_t=False,
            effective_window_length_seconds=0.1,
            token_bin_size_ms=100.0,
        )
        self.seen_token_shapes: list[tuple[int, ...]] = []

    def reconstruct(self, tokens: torch.Tensor):
        self.seen_token_shapes.append(tuple(tokens.shape))
        assert tokens.shape[1] == 1, (
            "Expected latent_robustness (M2) to split fixed-input tokens per native chunk."
        )
        return SimpleNamespace(reconstruction=tokens.squeeze(-1).repeat_interleave(2, dim=1))


class _StubPerceiverTokenizer:
    def __init__(self):
        self.spec = SimpleNamespace(KIND="perceiver_s4")
        self.absolute_end_chunks: list[torch.Tensor] = []

    def _get_embeddings(self, batch):
        self.absolute_end_chunks.append(batch.absolute_end.detach().cpu().clone())
        embeddings = torch.zeros(
            (batch.spike_matrix.shape[0], 1, 1), device=batch.spike_matrix.device
        )
        return embeddings, None

    @property
    def count_wrapper(self):
        return self

    def decode(self, noisy_latents_decode: torch.Tensor, embeddings: torch.Tensor):
        # Mirrors PerceiverTokenizer.count_wrapper.decode(z, e) -> (B, units, time).
        batch_n = int(noisy_latents_decode.shape[0])
        return torch.zeros(
            (batch_n, 2, 400),
            dtype=noisy_latents_decode.dtype,
            device=noisy_latents_decode.device,
        )


class _StubSessionAggregatedMetric(BaseMetric):
    name = "dummy_session_metric"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = []
    supported_data_types = {DataType.RATES}
    preferred_data_type = DataType.RATES
    aggregation_fallback_policy = "session_aggregated"

    def compute(self, intermediates):
        raise AssertionError("dummy_session_metric should only use session aggregation hooks")

    def _accumulate_window(self, session_state, window_intermediates) -> None:
        values = session_state.setdefault("values", [])
        values.append(
            float(np.asarray(window_intermediates.reconstructed.counts_per_bin()).mean())
        )
        session_state["n_windows_seen"] = session_state.get("n_windows_seen", 0) + 1

    def _finalize_session_state(self, session_state):
        return self._make_result(
            float(np.mean(np.asarray(session_state["values"], dtype=float))),
            metadata={"n_windows_seen": int(session_state["n_windows_seen"])},
        )


class _StubPerWindowMetric(BaseMetric):
    name = "dummy_window_metric"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = []
    supported_data_types = {DataType.RATES}
    preferred_data_type = DataType.RATES
    aggregation_fallback_policy = "per_window"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        return [IntermediateRequest("segment_absolute_start")]

    def compute(self, intermediates):
        return self._make_result(float(intermediates["segment_absolute_start"]))


class _StubPipeline:
    def __init__(self, config):
        self.metrics = [
            ("dummy_session_metric", _StubSessionAggregatedMetric()),
            ("dummy_window_metric", _StubPerWindowMetric()),
        ]


class _ScaledSessionAggregatedMetric(BaseMetric):
    """Session-aggregated stub whose value depends on a constructor param.

    Stands in for ``acg_correlation`` — one registered class configured
    several times (``acg_correlation`` / ``_5ms`` / ``_10ms``) with different
    parameters, so the per-variant values genuinely differ.
    """

    name = "scaled_session_metric"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = []
    _default_params = {"scale": 1.0}
    supported_data_types = {DataType.RATES}
    preferred_data_type = DataType.RATES
    aggregation_fallback_policy = "session_aggregated"

    def compute(self, intermediates):
        raise AssertionError("scaled_session_metric should only use session aggregation hooks")

    def _accumulate_window(self, session_state, window_intermediates) -> None:
        values = session_state.setdefault("values", [])
        values.append(
            float(np.asarray(window_intermediates.reconstructed.counts_per_bin()).mean())
            * float(self.params["scale"])
        )
        session_state["n_windows_seen"] = session_state.get("n_windows_seen", 0) + 1

    def _finalize_session_state(self, session_state):
        return self._make_result(
            float(np.mean(np.asarray(session_state["values"], dtype=float))),
            metadata={"n_windows_seen": int(session_state["n_windows_seen"])},
        )


class _ScaledPerWindowMetric(BaseMetric):
    """Per-window stub whose value depends on a constructor param."""

    name = "scaled_window_metric"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = []
    _default_params = {"offset": 0.0}
    supported_data_types = {DataType.RATES}
    preferred_data_type = DataType.RATES
    aggregation_fallback_policy = "per_window"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        return [IntermediateRequest("segment_absolute_start")]

    def compute(self, intermediates):
        return self._make_result(
            float(intermediates["segment_absolute_start"]) + float(self.params["offset"])
        )


class _RepeatedMetricClassStubPipeline:
    """Two display names per metric class, exactly like the M2 metric YAML."""

    def __init__(self, config):
        self.metrics = [
            ("scaled_session_metric", _ScaledSessionAggregatedMetric(scale=1.0)),
            ("scaled_session_metric_x10", _ScaledSessionAggregatedMetric(scale=10.0)),
            ("scaled_window_metric", _ScaledPerWindowMetric(offset=0.0)),
            ("scaled_window_metric_plus100", _ScaledPerWindowMetric(offset=100.0)),
        ]


_EMPTY_NEURON_SELECTION = NeuronSelection(
    policy="unit_dropout_first_n", max_neurons=None, seed=42, padded_to=None, sessions={}
)


def _make_run_job(tmp_path: Path) -> NTTEvalRobustnessLatentJob:
    spec = SimpleNamespace(
        KIND="fake_latent_tokenizer",
        window_length_seconds=1.0,
        effective_window_length_seconds=1.0,
        bin_size_ms=1.0,
        input_type=DataType.RATES,
        output_type=DataType.RATES,
        token_bin_size_ms=1.0,
        latent_type=None,
        variable_t=True,
        model_dump=lambda **_: {
            "KIND": "fake_latent_tokenizer",
            "bin_size_ms": 1.0,
            "window_length_seconds": 1.0,
        },
    )
    job = NTTEvalRobustnessLatentJob.model_construct(
        output_path=str(tmp_path),
        generate_plots=False,
        split="val",
        data_config=SimpleNamespace(deterministic_transforms=True),
        mlflow_logging=MLflowLoggingConfig(enabled=False),
        evaluate_all_neurons=False,
    )
    # The fakes have no units table; tests/evaluation/test_neuron_selection.py covers it.
    job._neuron_selection = _EMPTY_NEURON_SELECTION
    job._device = torch.device("cpu")
    job._spec = spec
    job._tokenizer = _FakeLatentTokenizer(spec)
    job._eval_config = EvaluationConfig(
        compute_baseline=False,
        max_eval_seconds=None,
        robustness_latent=RobustnessLatentConfig(
            epsilons=[0.25],
            n_samples_per_epsilon=2,
            n_eval_windows=2,
            plot_examples=False,
        ),
    )
    return job


class _FakeEvalDMConfig:
    def __init__(self, window_length_seconds: float):
        self.window_length_seconds = float(window_length_seconds)

    def model_copy(self, deep: bool = False):
        del deep
        return _FakeEvalDMConfig(self.window_length_seconds)


class _FakeExistingEvalDM:
    def __init__(self, *, window_length_seconds: float, loader_label: str):
        self.config = _FakeEvalDMConfig(window_length_seconds)
        self._loader_label = loader_label

    @property
    def window_length(self) -> float:
        return float(self.config.window_length_seconds)

    def val_dataloader(self):
        return self._loader_label


def _make_window_segment(
    session_id: str,
    segment_index: int,
    *,
    start_seconds: float,
    window_values_tu: list[list[float]],
):
    latents = torch.tensor(window_values_tu, dtype=torch.float32).unsqueeze(0)
    reconstructed = latents.squeeze(0).transpose(0, 1).contiguous()
    return SimpleNamespace(
        session_id=session_id,
        segment_index=segment_index,
        segment_absolute_start=start_seconds,
        latents=latents,
        reconstructed=reconstructed,
        metadata={},
        token_timestamps=None,
        behavioral_data_token_res=None,
        behavioral_data_input_res=None,
        strip_original=lambda: None,
    )


def _zero_noise(
    shape: tuple[int, ...],
    eps: float,
    sigma_per_dim: np.ndarray,
    distribution: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.zeros(shape, device=device, dtype=dtype)


def _flip_first_latent_dim_noise(
    shape: tuple[int, ...],
    eps: float,
    sigma_per_dim: np.ndarray,
    distribution: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    del eps, sigma_per_dim, distribution, generator
    noise = torch.zeros(shape, device=device, dtype=dtype)
    noise[..., 0] = 2.0
    return noise


def test_eval_robustness_latent_emit_figures_writes_curves_and_rasters(tmp_path: Path):
    """``_emit_figures`` should leave a complete plot directory on disk.

    Builds a tiny ``results.csv`` so ``plot_latent_corruption_a`` has
    valid input, plus a single captured-raster entry so
    ``plot_latent_corruption_b`` runs once. Asserts PNG+SVG+PDF triplets
    exist for both the curve and the raster.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Minimal results.csv in the per-(window, ε, k) raw schema (which
    # ``plot_latent_corruption_a`` aggregates on-the-fly).
    csv_path = out_dir / "results.csv"
    csv_path.write_text(
        "session_id,segment_index,epsilon,k,acg_correlation,ccg_correlation,mse_match_miss\n"
        "sess0,0,0.01,0,0.9,0.85,0.001\n"
        "sess0,0,0.01,1,0.92,0.87,0.0011\n"
        "sess0,0,0.1,0,0.7,0.65,0.01\n"
        "sess0,0,0.1,1,0.72,0.66,0.011\n"
    )

    # Synthetic captured rasters: one segment for sess0, three ε panels.
    rng = np.random.default_rng(0)
    clean = (rng.random((50, 4)) > 0.8).astype(np.float32)  # [T, U]
    noisy = (rng.random((50, 4)) > 0.7).astype(np.float32)
    captured = {
        "sess0": {
            0: {
                "clean": clean,
                "noisy_per_eps": {0.01: [noisy], 0.1: [noisy]},
                "metric_per_eps": {
                    0.01: {"acg_correlation": 0.91, "mse_match_miss": 0.001},
                    0.1: {"acg_correlation": 0.71, "mse_match_miss": 0.01},
                },
            }
        }
    }

    rl_cfg = RobustnessLatentConfig(
        epsilons=[0.01, 0.1],
        n_samples_per_epsilon=2,
        plot_examples=True,
    )
    job = _make_job_skeleton()
    job._emit_figures(
        output_dir=out_dir,
        rl_cfg=rl_cfg,
        bin_size_seconds=0.001,
        captured_rasters=captured,
    )

    # Curve: PNG + SVG + PDF triplet.
    curves_dir = out_dir / "curves"
    for ext in ("png", "svg", "pdf"):
        assert (curves_dir / f"latent-corruption-a.{ext}").exists(), (
            f"missing curves/latent-corruption-a.{ext}"
        )

    # Raster: PNG + SVG + PDF triplet.
    rasters_dir = out_dir / "rasters"
    for ext in ("png", "svg", "pdf"):
        path = rasters_dir / f"latent-corruption-b_sess0_w0.{ext}"
        assert path.exists(), f"missing {path}"


def test_eval_robustness_latent_skips_rasters_when_plot_examples_false(tmp_path: Path):
    """``plot_examples=False`` should still emit the aggregate curve but
    no rasters.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    csv_path = out_dir / "results.csv"
    csv_path.write_text(
        "session_id,segment_index,epsilon,k,acg_correlation\n"
        "sess0,0,0.01,0,0.9\n"
        "sess0,0,0.1,0,0.7\n"
    )

    rl_cfg = RobustnessLatentConfig(
        epsilons=[0.01, 0.1], n_samples_per_epsilon=1, plot_examples=False
    )
    job = _make_job_skeleton()
    job._emit_figures(
        output_dir=out_dir,
        rl_cfg=rl_cfg,
        bin_size_seconds=0.001,
        captured_rasters={},
    )

    # Curve still emitted.
    assert (out_dir / "curves" / "latent-corruption-a.png").exists()
    # No rasters dir created.
    assert not (out_dir / "rasters").exists()


def test_units_time_to_TU_helper():
    """``_units_time_to_TU`` should transpose ``(U, T)`` → ``(T, U)``."""
    from ntt.jobs.eval_robustness_latent import _units_time_to_TU

    arr = np.zeros((4, 50), dtype=np.float32)  # (U=4, T=50)
    out = _units_time_to_TU(arr)
    assert out.shape == (50, 4)

    # Tensor input is also accepted.
    out2 = _units_time_to_TU(torch.from_numpy(arr))
    assert out2.shape == (50, 4)


def test_eval_robustness_latent_run_rejects_flush_caps(tmp_path: Path):
    job = _make_run_job(tmp_path)
    job._eval_config.max_windows = 1
    job._eval_config.max_bytes = 1024

    with pytest.raises(
        ValueError, match="does not accept evaluation.max_windows or evaluation.max_bytes"
    ):
        job.run()


def test_eval_robustness_latent_run_emits_session_level_rows(tmp_path: Path, monkeypatch):
    job = _make_run_job(tmp_path)
    segments = [
        _make_window_segment(
            "sess0", 0, start_seconds=0.0, window_values_tu=[[1.0, 1.0], [1.0, 1.0]]
        ),
        _make_window_segment(
            "sess0", 1, start_seconds=1.0, window_values_tu=[[3.0, 3.0], [3.0, 3.0]]
        ),
        _make_window_segment(
            "sess1", 0, start_seconds=0.0, window_values_tu=[[5.0, 5.0], [5.0, 5.0]]
        ),
        _make_window_segment(
            "sess1", 1, start_seconds=1.0, window_values_tu=[[7.0, 7.0], [7.0, 7.0]]
        ),
    ]

    monkeypatch.setattr(
        "ntt.jobs.eval_robustness_latent.EvaluationPipeline",
        _StubPipeline,
    )
    monkeypatch.setattr(
        NTTEvalRobustnessLatentJob,
        "_draw_noise_from_generator",
        staticmethod(_zero_noise),
    )
    monkeypatch.setattr(
        job,
        "_build_eval_loader_and_tokenizer_fn",
        lambda *args, **kwargs: ("unused_loader", lambda batch: {"reconstruction": batch}),
    )
    monkeypatch.setattr(
        job,
        "_collect_selected_window_segments",
        lambda **kwargs: (
            segments,
            ["sess0", "sess1"],
            {"sess0": [0, 1], "sess1": [0, 1]},
        ),
    )
    monkeypatch.setattr(
        job,
        "_compute_sigma",
        lambda segments, seed: LatentStats(
            sigma_joint=1.0,
            sigma_per_dim=np.ones((2,), dtype=np.float32),
        ),
    )

    result_locals = job.run()

    jsonl_path = tmp_path / "results.jsonl"
    with jsonl_path.open() as fh:
        rows = [json.loads(line) for line in fh]

    assert len(rows) == 6
    rows_by_key = {(row["session_id"], row["metric"], float(row["epsilon"])): row for row in rows}

    assert rows_by_key[("sess0", "dummy_session_metric", 0.25)]["raw_value"] == pytest.approx(2.0)
    assert rows_by_key[("sess1", "dummy_session_metric", 0.25)]["raw_value"] == pytest.approx(6.0)
    assert rows_by_key[("sess0", "dummy_window_metric", 0.25)]["raw_value"] == pytest.approx(0.5)
    assert rows_by_key[("sess1", "dummy_window_metric", 0.25)]["raw_value"] == pytest.approx(0.5)
    assert rows_by_key[("sess0", "mse_match_miss", 0.25)]["raw_value"] == pytest.approx(0.0)
    assert rows_by_key[("sess1", "mse_match_miss", 0.25)]["raw_value"] == pytest.approx(0.0)
    assert (
        rows_by_key[("sess0", "dummy_session_metric", 0.25)]["metadata"]["n_samples_requested"]
        == 2
    )
    assert rows_by_key[("sess0", "dummy_session_metric", 0.25)]["metadata"]["n_samples_ok"] == 2
    assert (
        rows_by_key[("sess0", "dummy_session_metric", 0.25)]["metadata"]["n_samples_skipped"] == 0
    )
    assert rows_by_key[("sess0", "dummy_session_metric", 0.25)]["metadata"][
        "score_std"
    ] == pytest.approx(0.0)
    assert "k" not in rows_by_key[("sess0", "dummy_session_metric", 0.25)]
    assert all(row["status"] == "ok" for row in rows)

    assert len(result_locals["csv_rows"]) == 4

    with (tmp_path / "results.csv").open() as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 2
    assert "segment_index" not in csv_rows[0]
    assert float(csv_rows[0]["dummy_session_metric_mean"]) == pytest.approx(2.0)
    assert float(csv_rows[1]["dummy_session_metric_mean"]) == pytest.approx(6.0)
    assert float(csv_rows[0]["dummy_window_metric_mean"]) == pytest.approx(0.5)
    assert float(csv_rows[0]["mse_match_miss_mean"]) == pytest.approx(0.0)

    with (tmp_path / "results_per_session.csv").open() as fh:
        per_session_rows = list(csv.DictReader(fh))
    assert len(per_session_rows) == 2
    assert float(per_session_rows[0]["dummy_session_metric"]) == pytest.approx(2.0)
    assert float(per_session_rows[1]["dummy_session_metric"]) == pytest.approx(6.0)
    assert int(per_session_rows[0]["n_rows"]) == 2

    # The report itself saves what the aggregation's comparability check compares, both
    # on disk and in the in-memory report that the MLflow upload serializes.
    from ntt.jobs.aggregation import _robustness_settings

    report = json.loads((tmp_path / "eval_report.json").read_text())
    assert result_locals["aggregated_report"].to_dict()["config"] == report["config"]
    assert report_session_summary(report) == {
        "session_ids": ["sess0", "sess1"],
        "n_sessions": 2,
        "aggregation_strategy": "mean",
    }
    assert _robustness_settings(report) == {
        "split": "val",
        "bin_size_ms": 1.0,
        "window_length_seconds": 1.0,
        "scored_window_seconds": 1.0,
        "max_eval_seconds": None,
        "max_windows": None,
        "max_neurons": None,
        "evaluate_all_neurons": False,
        "neuron_subsample_seed": 42,
        "conversion_bin_size_ms": 1.0,
        "epsilons": [0.25],
        "n_samples_per_epsilon": 2,
        "noise_distribution": "gaussian",
        "n_eval_windows": 2,
        "metrics": None,
    }


def test_eval_robustness_latent_run_keeps_repeated_metric_class_variants_distinct(
    tmp_path: Path, monkeypatch
):
    """Two display names for ONE metric class must not collapse.

    ``config/evaluation/metrics/robustness_latent.yaml`` configures
    ``acg_correlation``/``_5ms``/``_10ms`` (and the ccg trio) from a single
    registered class. Keying every result and session-accumulator
    bucket by ``metric.name`` would make the variants share one bucket: the extra
    ones would be computed at full GPU cost and then silently discarded, and the
    session-aggregated ones would merge their running sums into a mixture that
    matches no configured variant.

    Also pins that the FIRST (base) display name keeps its exact key —
    ``aggregation.category_scores.resolve_m2_acg`` matches
    ``name.split("/", 1)[0] == "acg_correlation"`` to build the M2 member of
    the ``latent_structure_and_robustness`` composite.
    """
    job = _make_run_job(tmp_path)
    segments = [
        _make_window_segment(
            "sess0", 0, start_seconds=0.0, window_values_tu=[[1.0, 1.0], [1.0, 1.0]]
        ),
        _make_window_segment(
            "sess0", 1, start_seconds=1.0, window_values_tu=[[3.0, 3.0], [3.0, 3.0]]
        ),
    ]

    monkeypatch.setattr(
        "ntt.jobs.eval_robustness_latent.EvaluationPipeline",
        _RepeatedMetricClassStubPipeline,
    )
    monkeypatch.setattr(
        NTTEvalRobustnessLatentJob,
        "_draw_noise_from_generator",
        staticmethod(_zero_noise),
    )
    monkeypatch.setattr(
        job,
        "_build_eval_loader_and_tokenizer_fn",
        lambda *args, **kwargs: ("unused_loader", lambda batch: {"reconstruction": batch}),
    )
    monkeypatch.setattr(
        job,
        "_collect_selected_window_segments",
        lambda **kwargs: (segments, ["sess0"], {"sess0": [0, 1]}),
    )
    monkeypatch.setattr(
        job,
        "_compute_sigma",
        lambda segments, seed: LatentStats(
            sigma_joint=1.0,
            sigma_per_dim=np.ones((2,), dtype=np.float32),
        ),
    )

    job.run()

    with (tmp_path / "results.jsonl").open() as fh:
        rows = [json.loads(line) for line in fh]

    values_by_metric = {row["metric"]: row for row in rows}
    # One row per configured display name (+ mse_match_miss), none dropped.
    assert set(values_by_metric) == {
        "scaled_session_metric",
        "scaled_session_metric_x10",
        "scaled_window_metric",
        "scaled_window_metric_plus100",
        "mse_match_miss",
    }
    assert len(rows) == len(values_by_metric)
    assert all(row["status"] == "ok" for row in rows)

    # Each variant reports ITS OWN value, not a shared/merged one.
    assert values_by_metric["scaled_session_metric"]["raw_value"] == pytest.approx(2.0)
    assert values_by_metric["scaled_session_metric_x10"]["raw_value"] == pytest.approx(20.0)
    assert values_by_metric["scaled_window_metric"]["raw_value"] == pytest.approx(0.5)
    assert values_by_metric["scaled_window_metric_plus100"]["raw_value"] == pytest.approx(100.5)

    # Same story in the wide CSV and in the report keys consumed downstream.
    with (tmp_path / "results.csv").open() as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 1
    assert float(csv_rows[0]["scaled_session_metric_mean"]) == pytest.approx(2.0)
    assert float(csv_rows[0]["scaled_session_metric_x10_mean"]) == pytest.approx(20.0)

    report = json.loads((tmp_path / "eval_report.json").read_text())
    report_keys = set(report["results"])
    assert "scaled_session_metric/eps_0.25" in report_keys
    assert "scaled_session_metric_x10/eps_0.25" in report_keys
    assert "scaled_window_metric/eps_0.25" in report_keys
    assert "scaled_window_metric_plus100/eps_0.25" in report_keys


def test_latent_robustness_metric_config_variants_are_distinct_display_names():
    """The M2 metric YAML must expose all of {1, 5, 10} ms correlogram variants.

    Composed the way the job composes it (repo-root config, ``job=...``).
    Pins both halves of the contract: the four extra variants exist as their
    own display names AND the base ``acg_correlation``/``ccg_correlation``
    keys are unchanged (``resolve_m2_acg`` matches ``acg_correlation``
    literally, so renaming it would silently drop M2 from the headline
    ``latent_structure_and_robustness`` composite).
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from ntt.evaluation.config import instantiate_metrics_from_config
    from ntt.evaluation.utils.metric_selection import (
        is_latent_robustness_reconstruction_metric,
    )

    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="main", overrides=["job=eval_robustness_latent"])

    pairs = instantiate_metrics_from_config(
        OmegaConf.to_container(cfg.job.eval_config.metrics, resolve=True)
    )
    selected = [
        (display_name, metric)
        for display_name, metric in pairs
        if is_latent_robustness_reconstruction_metric(metric.name, metric.category)
    ]
    display_names = [display_name for display_name, _ in selected]

    assert len(display_names) == len(set(display_names)), display_names
    for expected in (
        "acg_correlation",
        "acg_correlation_5ms",
        "acg_correlation_10ms",
        "ccg_correlation",
        "ccg_correlation_5ms",
        "ccg_correlation_10ms",
    ):
        assert expected in display_names

    # Three display names share the ``acg_correlation`` class (same for ccg):
    # that is exactly the collapse the job must not perform.
    by_display = dict(selected)
    assert by_display["acg_correlation_5ms"].name == "acg_correlation"
    assert by_display["acg_correlation_10ms"].name == "acg_correlation"
    assert by_display["ccg_correlation_5ms"].name == "ccg_correlation"
    assert by_display["ccg_correlation_10ms"].name == "ccg_correlation"

    bin_sizes = {
        name: by_display[name].params["correlogram_bin_size_ms"]
        for name in ("acg_correlation", "acg_correlation_5ms", "acg_correlation_10ms")
    }
    assert bin_sizes == {
        "acg_correlation": 1.0,
        "acg_correlation_5ms": 5.0,
        "acg_correlation_10ms": 10.0,
    }


def test_eval_robustness_latent_run_records_token_distance_for_wanneural_bsq_without_quantize(
    tmp_path: Path,
    monkeypatch,
):
    spec = WanNeuralTokenizerSpec(
        n_neurons=2,
        window_length_seconds=0.004,
        bin_size_ms=1.0,
        token_bin_size_ms=1.0,
        dim=4,
        z_dim=2,
        num_levels=1,
        dim_mult=(1,),
        num_res_blocks=1,
        attn_levels=(),
        num_attn_heads=1,
        temporal_patch_size=1,
        temporal_downsample=(),
        k_t=1,
        k_t_down=1,
        s_t_down=1,
        k_t_init=1,
        s_t_init=1,
        neuron_kernel_size=1,
        bottleneck_type="bsq",
        bsq_codebook_dim=2,
        output_activation="none",
        reconstruct_sample="round",
        input_type=DataType.RATES,
        output_type=DataType.RATES,
    )
    tokenizer = WanNeuralTokenizer(spec)
    assert not hasattr(tokenizer, "quantize")

    clean_indices = torch.zeros((1, spec.num_latent_time_steps * spec.n_neurons), dtype=torch.long)
    clean_codes_bctn = tokenizer.bottleneck.indices_to_codes(
        clean_indices,
        T=spec.num_latent_time_steps,
    )
    clean_latents = clean_codes_bctn.permute(0, 2, 3, 1).contiguous()
    clean_reconstruction = (
        tokenizer.reconstruct(clean_latents).reconstruction.squeeze(0).transpose(0, 1).contiguous()
    )
    segments = [
        SimpleNamespace(
            session_id="sess0",
            segment_index=0,
            segment_absolute_start=0.0,
            latents=clean_latents,
            reconstructed=clean_reconstruction,
            metadata={"sampling_rate": 1000.0},
            token_timestamps=None,
            behavioral_data_token_res=None,
            behavioral_data_input_res=None,
            strip_original=lambda: None,
        )
    ]

    job = NTTEvalRobustnessLatentJob.model_construct(
        output_path=str(tmp_path),
        generate_plots=False,
        split="val",
        data_config=SimpleNamespace(deterministic_transforms=True),
        mlflow_logging=MLflowLoggingConfig(enabled=False),
        evaluate_all_neurons=False,
    )
    # The fakes have no units table; tests/evaluation/test_neuron_selection.py covers it.
    job._neuron_selection = _EMPTY_NEURON_SELECTION
    job._device = torch.device("cpu")
    job._spec = spec
    job._tokenizer = tokenizer
    job._eval_config = EvaluationConfig(
        compute_baseline=False,
        max_eval_seconds=None,
        robustness_latent=RobustnessLatentConfig(
            epsilons=[0.25],
            n_samples_per_epsilon=1,
            n_eval_windows=1,
            plot_examples=False,
        ),
    )

    monkeypatch.setattr(
        "ntt.jobs.eval_robustness_latent.EvaluationPipeline",
        _StubPipeline,
    )
    monkeypatch.setattr(
        NTTEvalRobustnessLatentJob,
        "_draw_noise_from_generator",
        staticmethod(_flip_first_latent_dim_noise),
    )
    monkeypatch.setattr(
        job,
        "_build_eval_loader_and_tokenizer_fn",
        lambda *args, **kwargs: ("unused_loader", lambda batch: {"reconstruction": batch}),
    )
    monkeypatch.setattr(
        job,
        "_collect_selected_window_segments",
        lambda **kwargs: (segments, ["sess0"], {"sess0": [0]}),
    )
    monkeypatch.setattr(
        job,
        "_compute_sigma",
        lambda segments, seed: LatentStats(
            sigma_joint=1.0,
            sigma_per_dim=np.ones((spec.z_dim,), dtype=np.float32),
        ),
    )

    job.run()

    with (tmp_path / "run_meta.json").open() as fh:
        run_meta = json.load(fh)

    assert "token_distance" in run_meta
    assert run_meta["token_distance"] == [
        {
            "session_id": "sess0",
            "eps": 0.25,
            "mean_fraction_changed": pytest.approx(1.0),
            "std_fraction_changed": pytest.approx(0.0),
            "n_codes": spec.num_latent_time_steps * spec.n_neurons,
            "n_records": 1,
        }
    ]


def test_windowed_latent_noise_draws_are_deterministic_per_window():
    sigma_per_dim = np.asarray([1.0, 2.0], dtype=np.float32)
    first = NTTEvalRobustnessLatentJob._draw_noise(
        shape=(1, 4, 2),
        eps=0.25,
        sigma_per_dim=sigma_per_dim,
        distribution="gaussian",
        seed=7,
        k=2,
        session_id="sess0",
        window_index=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    repeated = NTTEvalRobustnessLatentJob._draw_noise(
        shape=(1, 4, 2),
        eps=0.25,
        sigma_per_dim=sigma_per_dim,
        distribution="gaussian",
        seed=7,
        k=2,
        session_id="sess0",
        window_index=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    other_window = NTTEvalRobustnessLatentJob._draw_noise(
        shape=(1, 4, 2),
        eps=0.25,
        sigma_per_dim=sigma_per_dim,
        distribution="gaussian",
        seed=7,
        k=2,
        session_id="sess0",
        window_index=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, other_window)


def test_windowed_codes_corruption_draws_are_deterministic_per_window():
    clean_codes = np.asarray(
        [
            [[0, 1], [2, 3]],
        ],
        dtype=np.int64,
    )
    first_corrupted, first_mask = NTTEvalRobustnessLatentJob._draw_corruption_for_codes(
        clean_codes,
        eps=0.3,
        vocab_size=32,
        seed=5,
        k=1,
        session_id="sess0",
        window_index=0,
    )
    repeated_corrupted, repeated_mask = NTTEvalRobustnessLatentJob._draw_corruption_for_codes(
        clean_codes,
        eps=0.3,
        vocab_size=32,
        seed=5,
        k=1,
        session_id="sess0",
        window_index=0,
    )
    other_window_corrupted, other_window_mask = (
        NTTEvalRobustnessLatentJob._draw_corruption_for_codes(
            clean_codes,
            eps=0.3,
            vocab_size=32,
            seed=5,
            k=1,
            session_id="sess0",
            window_index=1,
        )
    )

    np.testing.assert_array_equal(first_corrupted, repeated_corrupted)
    np.testing.assert_array_equal(first_mask, repeated_mask)
    assert not np.array_equal(first_corrupted, other_window_corrupted)
    assert not np.array_equal(first_mask, other_window_mask)


def test_eval_robustness_latent_respects_max_decode_batch_size(
    tmp_path: Path,
    monkeypatch,
):
    job = _make_run_job(tmp_path)
    recording_tokenizer = _RecordingFakeLatentTokenizer(job._spec)
    job._tokenizer = recording_tokenizer
    assert job._eval_config.robustness_latent is not None
    job._eval_config.robustness_latent.max_decode_batch_size = 2
    segments = [
        _make_window_segment(
            "sess0", 0, start_seconds=0.0, window_values_tu=[[1.0, 1.0], [1.0, 1.0]]
        ),
        _make_window_segment(
            "sess0", 1, start_seconds=1.0, window_values_tu=[[3.0, 3.0], [3.0, 3.0]]
        ),
    ]

    monkeypatch.setattr(
        "ntt.jobs.eval_robustness_latent.EvaluationPipeline",
        _StubPipeline,
    )
    monkeypatch.setattr(
        NTTEvalRobustnessLatentJob,
        "_draw_noise_from_generator",
        staticmethod(_zero_noise),
    )
    monkeypatch.setattr(
        job,
        "_build_eval_loader_and_tokenizer_fn",
        lambda *args, **kwargs: ("unused_loader", lambda batch: {"reconstruction": batch}),
    )
    monkeypatch.setattr(
        job,
        "_collect_selected_window_segments",
        lambda **kwargs: (segments, ["sess0"], {"sess0": [0, 1]}),
    )
    monkeypatch.setattr(
        job,
        "_compute_sigma",
        lambda segments, seed: LatentStats(
            sigma_joint=1.0,
            sigma_per_dim=np.ones((2,), dtype=np.float32),
        ),
    )

    job.run()

    assert recording_tokenizer.decode_batch_sizes == [2, 2]


def test_decode_reconstruction_batch_splits_chunk_stitched_fixed_input_tokens():
    job = NTTEvalRobustnessLatentJob.model_construct()
    tokenizer = _StrictFixedInputChunkTokenizer()
    batch_tensor = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2, 1)

    decoded = job._decode_reconstruction_batch(
        tokenizer=tokenizer,
        batch_tensor=batch_tensor,
        corruption_target="latents",
        segment=SimpleNamespace(session_id="sess0", metadata={}),
        clean_recon_t=torch.zeros((2, 6), dtype=torch.float32),
        sampling_rate=10.0,
        max_decode_batch_size=None,
    )

    expected = batch_tensor.squeeze(-1).repeat_interleave(2, dim=1)
    torch.testing.assert_close(decoded, expected)
    # The chunked-decode path merges the ``n_chunks`` slices into the batch dim
    # and calls ``tokenizer.reconstruct`` once with batch size ``B * n_chunks``:
    # 1 call of shape ``(6, 1, 2, 1)`` rather than 3 calls of shape
    # ``(2, 1, 2, 1)`` — same per-chunk token width (``shape[1] == 1``),
    # so fixed-input tokenizers see exactly the input they expect.
    assert tokenizer.seen_token_shapes == [(6, 1, 2, 1)]


def _make_fitted_pca_tokenizer(n_neurons: int = 8, n_components: int = 2) -> PCABaselineTokenizer:
    """PCA's latents are ``[B, 1, N, k]``: one token per window, a k-dim latent per neuron."""
    torch.manual_seed(0)
    spec = PCABaselineTokenizerSpec(
        n_neurons=n_neurons,
        window_length_seconds=1.0,
        bin_size_ms=1.0,
        n_components=n_components,
    )
    tokenizer = PCABaselineTokenizer(spec).train()
    for _ in range(4):
        tokenizer._accumulate_running_stats(torch.poisson(torch.full((4 * n_neurons, 1000), 0.02)))
    assert tokenizer.fit_from_running_stats()
    return tokenizer.eval()


def test_decode_reconstruction_batch_keeps_pca_unit_axis():
    """PCA's latents are ``[B, N, k]``: collation must not read the neurons as 1 s
    tokens (which would strip all but neuron 0 and make M2 skip every metric on a
    1-neuron decode)."""
    job = NTTEvalRobustnessLatentJob.model_construct()
    tokenizer = _make_fitted_pca_tokenizer()
    latents = torch.randn(3, 1, 8, 2)

    decoded = job._decode_reconstruction_batch(
        tokenizer=tokenizer,
        batch_tensor=latents,
        corruption_target="latents",
        segment=SimpleNamespace(session_id="sess0", metadata={}),
        clean_recon_t=torch.zeros((8, 1000), dtype=torch.float32),
        sampling_rate=1000.0,
        max_decode_batch_size=None,
    )

    torch.testing.assert_close(decoded, tokenizer.reconstruct(latents).reconstruction)
    assert decoded.shape == (3, 1000, 8)


def test_decode_reconstruction_batch_splits_pca_windows_stitched_along_token_axis():
    """An eval window spanning two native PCA windows: the collator concatenates the two
    ``[B, 1, N, k]`` latents along the token axis, and M2 splits them back per window."""
    job = NTTEvalRobustnessLatentJob.model_construct()
    tokenizer = _make_fitted_pca_tokenizer()
    first_window, second_window = torch.randn(3, 1, 8, 2), torch.randn(3, 1, 8, 2)

    decoded = job._decode_reconstruction_batch(
        tokenizer=tokenizer,
        batch_tensor=torch.cat([first_window, second_window], dim=1),
        corruption_target="latents",
        segment=SimpleNamespace(session_id="sess0", metadata={}),
        clean_recon_t=torch.zeros((8, 2000), dtype=torch.float32),
        sampling_rate=1000.0,
        max_decode_batch_size=None,
    )

    expected = torch.cat(
        [
            tokenizer.reconstruct(first_window).reconstruction,
            tokenizer.reconstruct(second_window).reconstruction,
        ],
        dim=1,
    )
    torch.testing.assert_close(decoded, expected)


class _ShortDecodeFakeLatentTokenizer(_FakeLatentTokenizer):
    """Decodes one time bin short, like Cosmos after the token-padding strip."""

    def reconstruct(self, latents: torch.Tensor):
        return SimpleNamespace(reconstruction=latents[:, :-1, :])


class _ShapeCheckingSessionMetric(_StubSessionAggregatedMetric):
    """Session metric that, like the real ones, rejects original/reconstructed shape mismatch."""

    name = "shape_checking_session_metric"

    def _accumulate_window(self, session_state, window_intermediates) -> None:
        original_shape = np.asarray(window_intermediates.original.counts_per_bin()).shape
        reconstructed_shape = np.asarray(window_intermediates.reconstructed.counts_per_bin()).shape
        if original_shape != reconstructed_shape:
            raise ValueError(f"shape mismatch: {original_shape} vs {reconstructed_shape}")
        super()._accumulate_window(session_state, window_intermediates)


class _ShapeCheckingStubPipeline:
    def __init__(self, config):
        self.metrics = [
            ("shape_checking_session_metric", _ShapeCheckingSessionMetric()),
            ("dummy_window_metric", _StubPerWindowMetric()),
        ]


def test_eval_robustness_latent_run_truncates_both_sides_when_noisy_decode_is_short(
    tmp_path: Path, monkeypatch
):
    """Cosmos: within the wavelet-boundary tolerance, the clean side must be truncated
    for ``original`` as well as ``reconstructed``, otherwise every shape-checking
    session metric is skipped."""
    job = _make_run_job(tmp_path)
    job._tokenizer = _ShortDecodeFakeLatentTokenizer(job._spec)
    window_values_tu = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    segments = [
        _make_window_segment("sess0", 0, start_seconds=0.0, window_values_tu=window_values_tu),
        _make_window_segment("sess0", 1, start_seconds=1.0, window_values_tu=window_values_tu),
    ]

    monkeypatch.setattr(
        "ntt.jobs.eval_robustness_latent.EvaluationPipeline",
        _ShapeCheckingStubPipeline,
    )
    monkeypatch.setattr(
        NTTEvalRobustnessLatentJob,
        "_draw_noise_from_generator",
        staticmethod(_zero_noise),
    )
    monkeypatch.setattr(
        job,
        "_build_eval_loader_and_tokenizer_fn",
        lambda *args, **kwargs: ("unused_loader", lambda batch: {"reconstruction": batch}),
    )
    monkeypatch.setattr(
        job,
        "_collect_selected_window_segments",
        lambda **kwargs: (segments, ["sess0"], {"sess0": [0, 1]}),
    )
    monkeypatch.setattr(
        job,
        "_compute_sigma",
        lambda segments, seed: LatentStats(
            sigma_joint=1.0,
            sigma_per_dim=np.ones((2,), dtype=np.float32),
        ),
    )

    job.run()

    with (tmp_path / "results.jsonl").open() as fh:
        rows = {row["metric"]: row for row in map(json.loads, fh)}
    assert rows["shape_checking_session_metric"]["status"] == "ok", rows
    # Zero noise: the noisy decode is the first three bins of the window (mean 3.5 Hz-units).
    assert rows["shape_checking_session_metric"]["raw_value"] == pytest.approx(3.5)
    assert rows["mse_match_miss"]["status"] == "ok", rows
    assert rows["mse_match_miss"]["raw_value"] == pytest.approx(0.0)


def test_collect_selected_window_segments_drops_trimmed_overlapping_windows_for_m2(
    tmp_path: Path,
) -> None:
    job = _make_run_job(tmp_path)
    assert job._eval_config.robustness_latent is not None

    spikes = torch.zeros((2, 1500), dtype=torch.int32)
    spikes[:, [100, 700, 1200]] = 1
    dataset = WindowedSpikeDataset(
        spikes=spikes,
        sampling_rate=1000.0,
        session_id="sess_overlap",
        length=2,
        window_size_samples=1000,
    )
    indices = [
        DatasetIndex("sess_overlap", 0.0, 1.0),
        DatasetIndex("sess_overlap", 0.5, 1.5),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    selected_segments, session_order, processed_window_ids = job._collect_selected_window_segments(
        dataloader=loader,
        tokenizer_fn=lambda batch: {"reconstruction": batch.spike_matrix},
        rl_cfg=job._eval_config.robustness_latent,
        spec=job._spec,
        device=torch.device("cpu"),
        bin_size_seconds=0.001,
    )

    assert session_order == ["sess_overlap"]
    assert processed_window_ids == {"sess_overlap": [0]}
    assert len(selected_segments) == 1
    assert selected_segments[0].segment_absolute_start == pytest.approx(0.0)


def test_collect_selected_window_segments_skips_partial_tail_windows(
    monkeypatch, tmp_path: Path
) -> None:
    job = _make_run_job(tmp_path)
    assert job._eval_config.robustness_latent is not None

    full_window = SimpleNamespace(
        session_id="sess_partial",
        segment_index=0,
        segment_absolute_start=0.0,
        original=torch.zeros((2, 1000), dtype=torch.float32),
        reconstructed=torch.zeros((2, 1000), dtype=torch.float32),
    )
    partial_window = SimpleNamespace(
        session_id="sess_partial",
        segment_index=1,
        segment_absolute_start=1.0,
        original=torch.zeros((2, 270), dtype=torch.float32),
        reconstructed=torch.zeros((2, 270), dtype=torch.float32),
    )

    monkeypatch.setattr(
        "ntt.jobs.eval_robustness_latent.iter_collated_validation_segments",
        lambda *args, **kwargs: iter([full_window, partial_window]),
    )

    selected_segments, session_order, processed_window_ids = job._collect_selected_window_segments(
        dataloader="unused_loader",
        tokenizer_fn=lambda batch: {"reconstruction": batch},
        rl_cfg=job._eval_config.robustness_latent,
        spec=job._spec,
        device=torch.device("cpu"),
        bin_size_seconds=0.001,
    )

    assert session_order == ["sess_partial"]
    assert processed_window_ids == {"sess_partial": [0]}
    assert selected_segments == [full_window]


def test_build_eval_loader_and_tokenizer_fn_rebuilds_dataloader_at_requested_eval_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = _make_run_job(tmp_path)
    job._datamodule = _FakeExistingEvalDM(
        window_length_seconds=0.1,
        loader_label="native_loader",
    )

    rebuild = {}

    class _RebuiltEvalDM:
        def __init__(self, config, tokenizer_spec=None):
            rebuild["window_length_seconds"] = float(config.window_length_seconds)
            rebuild["tokenizer_spec"] = tokenizer_spec
            self.config = config

        def setup(self, stage: str) -> None:
            rebuild["stage"] = stage

        def val_dataloader(self):
            return "rebuilt_loader"

    monkeypatch.setattr("ntt.jobs.eval_robustness_latent.NTTDataModule", _RebuiltEvalDM)

    dataloader, _tokenizer_fn = job._build_eval_loader_and_tokenizer_fn(
        eval_window_length_seconds=1.0
    )

    assert dataloader == "rebuilt_loader"
    assert rebuild == {
        "window_length_seconds": pytest.approx(1.0),
        "tokenizer_spec": None,
        "stage": "validate",
    }


def test_collect_selected_window_segments_accepts_long_fixed_input_eval_windows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    job = _make_run_job(tmp_path)
    assert job._eval_config.robustness_latent is not None

    long_window = SimpleNamespace(
        session_id="sess_ldns",
        segment_index=0,
        segment_absolute_start=0.0,
        original=torch.zeros((2, 512), dtype=torch.float32),
        reconstructed=torch.zeros((2, 512), dtype=torch.float32),
    )
    seen_kwargs = {}

    def _fake_iter(*args, **kwargs):
        del args
        seen_kwargs.update(kwargs)
        return iter([long_window])

    monkeypatch.setattr(
        "ntt.jobs.eval_robustness_latent.iter_collated_validation_segments",
        _fake_iter,
    )

    ldns_spec = SimpleNamespace(
        KIND="ldns",
        bin_size_ms=5.0,
        input_type=DataType.SPIKES,
        token_bin_size_ms=5.0,
        latent_type=None,
        variable_t=False,
        effective_window_length_seconds=2.56,
    )

    selected_segments, session_order, processed_window_ids = job._collect_selected_window_segments(
        dataloader="unused_loader",
        tokenizer_fn=lambda batch: {"reconstruction": batch},
        rl_cfg=job._eval_config.robustness_latent,
        spec=ldns_spec,
        device=torch.device("cpu"),
        bin_size_seconds=0.005,
        eval_window_length_seconds=2.56,
    )

    assert session_order == ["sess_ldns"]
    assert processed_window_ids == {"sess_ldns": [0]}
    assert selected_segments == [long_window]
    assert seen_kwargs["window_length_seconds"] == pytest.approx(2.56)


def test_write_run_meta_records_eval_window_length_seconds(tmp_path: Path) -> None:
    out_path = tmp_path / "run_meta.json"
    rl_cfg = RobustnessLatentConfig(
        epsilons=[0.25],
        n_samples_per_epsilon=1,
        n_eval_windows=1,
        plot_examples=False,
    )

    NTTEvalRobustnessLatentJob._write_run_meta(
        out_path,
        stats=LatentStats(
            sigma_joint=1.0,
            sigma_per_dim=np.ones((2,), dtype=np.float32),
        ),
        skipped_counter={},
        total_pairs=0,
        rl_cfg=rl_cfg,
        eval_window_length_seconds=2.56,
    )

    with out_path.open() as fh:
        run_meta = json.load(fh)

    assert run_meta["eval_window_length_seconds"] == pytest.approx(2.56)


def test_resolve_eval_window_length_seconds_uses_authoritative_eval_window() -> None:
    """The eval window is the single source of truth: the latent_robustness (M2) window-length
    resolver returns the configured ``eval_window_seconds`` regardless of the
    tokenizer's native window. Tokenizer compatibility with that window is
    validated separately at ``__enter__`` via ``apply_eval_window``."""
    job = _make_job_skeleton()
    ldns_like_spec = SimpleNamespace(
        KIND="ldns",
        variable_t=False,
        effective_window_length_seconds=2.56,
    )

    # Default eval window (no eval_config) is 1.0 s, not the 2.56 s native window.
    job._eval_config = None
    job.eval_config = None
    assert job._resolve_eval_window_length_seconds(ldns_like_spec) == pytest.approx(1.0)

    # An explicit eval_window_seconds is honored as the source of truth.
    job._eval_config = SimpleNamespace(eval_window_seconds=0.5)
    assert job._resolve_eval_window_length_seconds(ldns_like_spec) == pytest.approx(0.5)


def test_reconstruct_noisy_latents_perceiver_uses_full_window_duration_per_sample(
    tmp_path: Path,
) -> None:
    job = _make_run_job(tmp_path)
    tokenizer = _StubPerceiverTokenizer()
    segment = SimpleNamespace(
        session_id="sess0",
        metadata={
            "unit_ids": [0, 1],
            "subject_id": "subject0",
            "species": "species0",
        },
    )
    noisy_latents = torch.zeros((11, 3, 2, 4), dtype=torch.float32)
    clean_recon_t = torch.zeros((2, 400), dtype=torch.float32)

    decoded = job._reconstruct_noisy_latents_perceiver(
        tokenizer=tokenizer,
        noisy_latents=noisy_latents,
        segment=segment,
        clean_recon_t=clean_recon_t,
        sampling_rate=100.0,
    )

    assert isinstance(decoded.reconstruction, torch.Tensor)
    assert decoded.reconstruction.shape == (11, 400, 2)  # (N, time, units)
    assert len(tokenizer.absolute_end_chunks) == 2
    for absolute_end in tokenizer.absolute_end_chunks:
        np.testing.assert_allclose(absolute_end.numpy(), np.full_like(absolute_end.numpy(), 4.0))


def test_reconstruct_noisy_latents_perceiver_round_trips_real_tokenizer(tmp_path: Path) -> None:
    """With zero noise, latent_robustness (M2)'s perceiver decode must reproduce
    ``reconstruct_from_input`` on the real PerceiverTokenizer.

    The M2 path must not call ``tokenizer.decode(z, embeddings)``: the
    extracted PerceiverTokenizer.decode takes no embeddings (it always raises),
    which would crash M2 with a TypeError on every perceiver checkpoint. The stub
    tokenizer above cannot catch this.
    """
    from tests.test_perceiver_tokenizer_extraction import (
        N_UNITS_PER_SESSION,
        _make_batched_input,
        _make_tokenizer,
        _write_embeddings_file,
    )

    tokenizer = _make_tokenizer(_write_embeddings_file(tmp_path)).eval()
    batch = _make_batched_input(batch_size=1, n_units=N_UNITS_PER_SESSION, n_bins=8)
    with torch.no_grad():
        clean = tokenizer.reconstruct_from_input(batch, return_latents=True)
    clean_recon_t = clean["reconstruction"][0].T  # (units, time)
    n_replicas = 3
    segment = SimpleNamespace(
        session_id="session_0",
        metadata={
            "unit_ids": list(range(N_UNITS_PER_SESSION)),
            "subject_id": "subject_0",
            "species": "mouse",
        },
    )

    job = _make_run_job(tmp_path)
    with torch.no_grad():
        decoded = job._reconstruct_noisy_latents_perceiver(
            tokenizer=tokenizer,
            noisy_latents=clean["latents"].repeat(n_replicas, 1, 1, 1),  # (N, L, M, D)
            segment=segment,
            clean_recon_t=clean_recon_t,
            sampling_rate=100.0,
        )

    # Same (N, time, units) layout as reconstruct_from_input, which is what
    # _to_units_time stitches into (units, total_time).
    torch.testing.assert_close(
        decoded.reconstruction, clean["reconstruction"].expand(n_replicas, -1, -1)
    )
    torch.testing.assert_close(job._to_units_time(decoded.reconstruction[:1]), clean_recon_t)


def test_eval_robustness_latent_run_meta_writes_config_provenance(tmp_path: Path):
    """Core provenance fields should be written with a config snapshot sidecar."""
    meta_path = tmp_path / "run_meta.json"
    stats = LatentStats(sigma_joint=1.5, sigma_per_dim=np.array([1.0, 2.0]))
    rl_cfg = SimpleNamespace(
        epsilons=[0.01],
        n_samples_per_epsilon=2,
        noise_distribution="gaussian",
        n_eval_windows=5,
    )
    spec = SimpleNamespace(
        KIND="identity",
        model_dump=lambda mode: {"KIND": "identity", "token_bin_size_ms": 20.0},
    )

    NTTEvalRobustnessLatentJob._write_run_meta(
        meta_path,
        stats=stats,
        skipped_counter={},
        total_pairs=2,
        rl_cfg=rl_cfg,
        spec=spec,
        checkpoint="model.ckpt",
        eval_config={"eval_window_seconds": 1.0, "robustness_latent": {"n_eval_windows": 5}},
        data_config={"selection_name": "dev-val", "dataset_yaml": "datasets/dev.yaml"},
        hydra_overrides=["job=eval_robustness_latent"],
        composed_config_yaml="job:\n  KIND: eval_robustness_latent\n",
    )

    meta = json.loads(meta_path.read_text())
    assert meta["eval_config"]["robustness_latent"]["n_eval_windows"] == 5
    assert meta["data_config"]["dataset_yaml"] == "datasets/dev.yaml"
    assert meta["hydra_overrides"] == ["job=eval_robustness_latent"]
    assert meta["composed_config_snapshot_path"] == "config_snapshot.yaml"
    assert (tmp_path / "config_snapshot.yaml").read_text() == (
        "job:\n  KIND: eval_robustness_latent\n"
    )

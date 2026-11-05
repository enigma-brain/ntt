"""Tests for downstream prediction metrics with artificial data."""

import numpy as np
import pytest
import torch

from ntt.contrib.tokenizer.ldns.ldns_tokenizer import LDNSTokenizer, LDNSTokenizerSpec
from ntt.contrib.tokenizer.simple_vae import SimpleVAETokenizer, SimpleVAETokenizerSpec
from ntt.core.data.components.behavior_transforms import MOVEMENT_PHASE_LABELS
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.config.metric_config import instantiate_metrics_from_config
from ntt.evaluation.core.types import DataType, EvalData
from ntt.evaluation.metrics.downstream_prediction import (
    BehavioralClassificationMetric,
    BehavioralRegressionMetric,
    BehavioralTemporalDecoderMetric,
)
from ntt.evaluation.metrics.downstream_prediction_utils import (
    FeatureReducer,
    _apply_chunk_threshold_filter,
    _compute_chunk_token_indices,
    align_behavioral_to_chunks,
    chunk_by_temporal_window,
    preprocess_downstream_data,
)
from ntt.evaluation.pipeline import EvaluationPipeline
from tests.fixtures import make_dummy_binned_input


def _make_temporal_decoder_intermediates_from_output(
    output: dict,
    *,
    temporal_history: bool,
    token_bin_size_ms: float = 20.0,
) -> dict[str, object]:
    """Build temporal-decoder intermediates from a tokenizer output dict."""
    latents = output["latents"].detach().cpu().numpy()
    token_timestamps = output["token_timestamps"].detach().cpu().numpy()

    batch_size = latents.shape[0]
    tokens_per_trial = latents.shape[1]
    latent_features = latents.reshape(batch_size, tokens_per_trial, -1).astype(np.float32)

    target = np.zeros((batch_size, tokens_per_trial), dtype=np.float32)
    for token_idx in range(tokens_per_trial):
        current = latent_features[:, token_idx, 0]
        if temporal_history:
            prev = latent_features[:, max(token_idx - 1, 0), min(1, latent_features.shape[-1] - 1)]
            target[:, token_idx] = 1.1 * current - 0.4 * prev
        else:
            aux = latent_features[:, token_idx, min(1, latent_features.shape[-1] - 1)]
            target[:, token_idx] = 1.2 * current - 0.3 * aux

    return {
        "latents": latents,
        "behavioral_data_token_res": {
            "cursor_pos": target.reshape(-1),
            "trial_id": np.repeat(np.arange(batch_size), tokens_per_trial),
        },
        "token_timestamps": token_timestamps.reshape(-1),
        "token_bin_size_ms": token_bin_size_ms,
    }


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_intermediates(latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0):
    """Build intermediates dict with token_bin_size_ms included."""
    return {
        "latents": latents,
        "behavioral_data_token_res": behavioral_data,
        "token_timestamps": token_timestamps,
        "token_bin_size_ms": token_bin_size_ms,
    }


# ── chunk_by_temporal_window tests ──────────────────────────────────────


class TestChunkByTemporalWindow:
    """Core tests for chunk_by_temporal_window."""

    def test_single_token_per_chunk(self):
        """When chunk_window_ms == token_bin_size_ms, each token is its own chunk."""
        T, D = 10, 4
        latents = np.arange(T * D, dtype=float).reshape(T, D)
        timestamps = np.arange(T, dtype=float) * 0.005  # 5ms bins

        chunked, ts, indices, tpc = chunk_by_temporal_window(
            latents,
            timestamps,
            token_bin_size_ms=5.0,
            chunk_window_ms=5.0,
        )
        assert tpc == 1
        assert chunked.shape == (T, D)
        assert len(indices) == T
        np.testing.assert_array_equal(chunked, latents)

    def test_multiple_tokens_per_chunk(self):
        """100ms window with 5ms tokens → 20 tokens per chunk."""
        T, D = 100, 3
        latents = np.ones((T, D))
        timestamps = np.arange(T, dtype=float) * 0.005

        chunked, ts, indices, tpc = chunk_by_temporal_window(
            latents,
            timestamps,
            token_bin_size_ms=5.0,
            chunk_window_ms=100.0,
        )
        assert tpc == 20
        assert chunked.shape[1] == 20 * D
        assert len(indices) == chunked.shape[0]
        # All chunks should have the same content (all-ones)
        np.testing.assert_allclose(chunked, np.ones_like(chunked))

    def test_chunk_timestamps_are_first_token(self):
        """Chunk timestamps should be the timestamp of the first token in each chunk."""
        T, D = 20, 2
        timestamps = np.arange(T, dtype=float) * 0.01  # 10ms bins
        latents = np.zeros((T, D))

        chunked, ts, indices, tpc = chunk_by_temporal_window(
            latents,
            timestamps,
            token_bin_size_ms=10.0,
            chunk_window_ms=50.0,
        )
        assert tpc == 5
        # First token of chunk 0 is at index 0, chunk 1 at index 5
        np.testing.assert_allclose(ts[0], timestamps[0])
        np.testing.assert_allclose(ts[1], timestamps[5])

    def test_non_integer_ratio_raises(self):
        """Misaligned chunk_window_ms / token_bin_size_ms should raise ValueError."""
        T, D = 100, 2
        latents = np.ones((T, D))
        timestamps = np.arange(T, dtype=float) * 0.033  # 33ms bins

        # 100ms / 33ms = 3.03 → chunk larger than token but not a whole multiple
        with pytest.raises(ValueError, match="must be a multiple of"):
            chunk_by_temporal_window(
                latents,
                timestamps,
                token_bin_size_ms=33.0,
                chunk_window_ms=100.0,
            )

    def test_exact_divisibility(self):
        """Exact divisibility (e.g. 99ms / 33ms = 3) should work."""
        T, D = 100, 2
        latents = np.ones((T, D))
        timestamps = np.arange(T, dtype=float) * 0.033  # 33ms bins

        # 99ms / 33ms = 3 exactly
        chunked, ts, indices, tpc = chunk_by_temporal_window(
            latents,
            timestamps,
            token_bin_size_ms=33.0,
            chunk_window_ms=99.0,
        )
        assert tpc == 3
        assert chunked.shape[1] == 3 * D

    def test_window_too_small_raises(self):
        """Sub-token chunk (chunk_window_ms < token_bin_size_ms) raises a clear,
        actionable 'not applicable' message naming both sizes and the fix.

        This is the common large-window-tokenizer case (e.g. a 1000ms-token
        model evaluated with a 200ms behavioral chunk): the skip reason should
        read as 'not applicable at this resolution', not a generic error.
        """
        latents = np.ones((10, 2))
        timestamps = np.arange(10, dtype=float)

        with pytest.raises(ValueError, match="not applicable") as excinfo:
            chunk_by_temporal_window(
                latents,
                timestamps,
                token_bin_size_ms=1000.0,
                chunk_window_ms=200.0,
            )
        msg = str(excinfo.value)
        # Names both sizes and points at the concrete fix.
        assert "200" in msg and "1000" in msg
        assert "too coarse" in msg
        assert "multiple of" in msg

    def test_not_enough_tokens_raises(self):
        """Too few tokens for even one chunk should raise ValueError."""
        latents = np.ones((3, 2))
        timestamps = np.arange(3, dtype=float) * 0.005

        with pytest.raises(ValueError, match="Not enough tokens"):
            chunk_by_temporal_window(
                latents,
                timestamps,
                token_bin_size_ms=5.0,
                chunk_window_ms=100.0,
            )

    def test_chunk_indices_are_sequential(self):
        """Returned chunk_indices should be 0, 1, 2, ..."""
        T, D = 50, 2
        latents = np.ones((T, D))
        timestamps = np.arange(T, dtype=float) * 0.01

        _, _, indices, _ = chunk_by_temporal_window(
            latents,
            timestamps,
            token_bin_size_ms=10.0,
            chunk_window_ms=50.0,
        )
        np.testing.assert_array_equal(indices, np.arange(len(indices)))


# ── FeatureReducer tests ───────────────────────────────────────────────


class TestFeatureReducer:
    """Core tests for FeatureReducer."""

    def test_truncate_reduces_features(self):
        X = np.random.default_rng(0).normal(size=(50, 100))
        reducer = FeatureReducer(method="truncate", n_features=32)
        X_out = reducer.fit_transform(X)
        assert X_out.shape == (50, 32)
        # Should be the first 32 columns
        np.testing.assert_array_equal(X_out, X[:, :32])

    def test_truncate_fewer_features_than_target(self):
        """When input has fewer features than n_features, keep all."""
        X = np.random.default_rng(0).normal(size=(50, 10))
        reducer = FeatureReducer(method="truncate", n_features=32)
        with pytest.warns(UserWarning, match="less than.*n_features"):
            X_out = reducer.fit_transform(X)
        assert X_out.shape == (50, 10)

    def test_pca_reduces_features(self):
        X = np.random.default_rng(0).normal(size=(100, 50))
        reducer = FeatureReducer(method="pca", n_features=10)
        X_train = reducer.fit_transform(X[:80])
        X_val = reducer.transform(X[80:])
        assert X_train.shape == (80, 10)
        assert X_val.shape == (20, 10)

    def test_srp_reduces_features(self):
        X = np.random.default_rng(0).normal(size=(100, 50))
        reducer = FeatureReducer(method="srp", n_features=10)
        X_out = reducer.fit_transform(X)
        assert X_out.shape == (100, 10)

    def test_none_passes_through(self):
        X = np.random.default_rng(0).normal(size=(50, 100))
        reducer = FeatureReducer(method="none", n_features=32)
        X_out = reducer.fit_transform(X)
        assert X_out.shape == (50, 100)  # unchanged

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown feature_reduction"):
            FeatureReducer(method="invalid")

    def test_transform_before_fit_raises(self):
        reducer = FeatureReducer(method="truncate")
        with pytest.raises(AssertionError, match="fit.*must be called"):
            reducer.transform(np.ones((5, 10)))


# ── align_behavioral_to_chunks tests ───────────────────────────────────


class TestAlignBehavioralToChunks:
    """Core tests for align_behavioral_to_chunks (index-based)."""

    def test_basic_regression_alignment(self):
        """Simple alignment: 5 chunks, 100ms window, 1000Hz behavioral data."""
        n_behavioral = 600  # 600ms of data at 1kHz
        chunk_indices = np.arange(5)  # 5 chunks at 0, 100, 200, 300, 400ms
        behavioral = {
            "cursor_pos": np.arange(n_behavioral, dtype=float),
            "trial_id": np.ones(n_behavioral),
        }

        y, grouping, keep_mask, meta = align_behavioral_to_chunks(
            chunk_indices=chunk_indices,
            chunk_window_ms=100.0,
            behavioral_data_input_res=behavioral,
            sampling_rate=1000.0,
            target_field="cursor_pos",
            mode="regression",
        )
        assert keep_mask.all()  # all chunks valid
        assert y.shape[0] == 5
        # Each chunk should have 100 behavioral bins (100ms * 1kHz)
        assert y.shape[1] == 100
        assert meta["K_input_res_per_chunk"] == 100

    def test_incomplete_chunks_dropped(self):
        """Chunks that extend beyond behavioral data should be dropped."""
        n_behavioral = 250  # only 250ms of data
        chunk_indices = np.arange(3)  # chunks at 0, 100, 200ms
        behavioral = {
            "cursor_pos": np.arange(n_behavioral, dtype=float),
            "trial_id": np.ones(n_behavioral),
        }

        y, _, keep_mask, meta = align_behavioral_to_chunks(
            chunk_indices=chunk_indices,
            chunk_window_ms=100.0,
            behavioral_data_input_res=behavioral,
            sampling_rate=1000.0,
            target_field="cursor_pos",
            mode="regression",
        )
        # Chunk 2 starts at 200ms, needs 100 bins → ends at 300ms > 250ms
        assert keep_mask[0] and keep_mask[1]
        assert not keep_mask[2]
        assert meta["n_chunks_dropped_incomplete"] == 1

    def test_trial_boundary_uses_majority_grouping(self):
        """Chunks spanning a trial boundary should use majority grouping value (no drop)."""
        n_behavioral = 300
        chunk_indices = np.arange(3)
        trial_ids = np.zeros(n_behavioral)
        trial_ids[150:] = 1  # trial boundary at 150ms

        behavioral = {
            "cursor_pos": np.ones(n_behavioral),
            "trial_id": trial_ids,
        }

        y, grouping_values, keep_mask, meta = align_behavioral_to_chunks(
            chunk_indices=chunk_indices,
            chunk_window_ms=100.0,
            behavioral_data_input_res=behavioral,
            sampling_rate=1000.0,
            target_field="cursor_pos",
            mode="regression",
        )
        # All chunks kept (no trial-boundary drop)
        assert keep_mask.all()
        assert len(y) == 3
        # Chunk 1 (100-200ms) spans boundary at 150ms: 50 bins trial 0, 50 bins trial 1
        # Majority is trial 0 (first half) — tie-breaking by np.unique picks lower value
        assert grouping_values[0] == 0  # chunk 0: all trial 0
        assert grouping_values[1] == 0  # chunk 1: 50/50, majority = 0 (tie goes to first unique)
        assert grouping_values[2] == 1  # chunk 2: all trial 1

    def test_classification_majority_vote(self):
        """Classification mode should use majority vote."""
        n_behavioral = 200
        chunk_indices = np.arange(2)
        labels = np.zeros(n_behavioral, dtype=int)
        labels[:80] = 1  # first 80ms: class 1, then class 0

        behavioral = {
            "label": labels,
            "trial_id": np.ones(n_behavioral),
        }

        y, _, keep_mask, _ = align_behavioral_to_chunks(
            chunk_indices=chunk_indices,
            chunk_window_ms=100.0,
            behavioral_data_input_res=behavioral,
            sampling_rate=1000.0,
            target_field="label",
            mode="classification",
        )
        # Chunk 0: 80 class-1, 20 class-0 → majority class 1 (80%)
        assert keep_mask[0]
        assert y[0] == 1
        # Chunk 1: all class 0
        assert y[1] == 0

    def test_behavioral_downsampling(self):
        """behavioral_bin_size_ms should downsample behavioral targets."""
        n_behavioral = 200
        chunk_indices = np.array([0])
        behavioral = {
            "cursor_pos": np.arange(n_behavioral, dtype=float),
            "trial_id": np.ones(n_behavioral),
        }

        y, _, _, meta = align_behavioral_to_chunks(
            chunk_indices=chunk_indices,
            chunk_window_ms=100.0,
            behavioral_data_input_res=behavioral,
            sampling_rate=1000.0,
            target_field="cursor_pos",
            mode="regression",
            behavioral_bin_size_ms=10.0,
        )
        # 100ms window, 10ms behavioral bins → 10 bins per chunk
        assert meta["K_per_chunk"] == 10
        assert y.shape[1] == 10


# ── Existing metric integration tests (updated for new API) ────────────


def test_behavioral_regression_synthetic():
    """Test regression metric with synthetic data where latents predict target."""
    rng = np.random.default_rng(42)
    n_samples = 100
    latent_dim = 10

    latents = rng.normal(0, 1, size=(n_samples, latent_dim))
    weights = np.array([2.0, -1.5, 1.0])
    target = latents[:, :3] @ weights + rng.normal(0, 0.1, size=n_samples)

    # 100ms token bins
    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    trial_ids = np.repeat(np.arange(10), 10)
    behavioral_data = {"cursor_pos": target, "trial_id": trial_ids}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    # chunk_window_ms=100 with token_bin_size_ms=100 → 1 token/chunk (no concatenation)
    metric = BehavioralRegressionMetric(
        target_field="cursor_pos",
        cv=3,
        train_drop_invalid=False,
        eval_drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    result = metric.compute(intermediates)

    assert result.raw_value > 0.5, f"R² too low: {result.raw_value}"
    assert "fold_alphas" in result.metadata
    assert "n_splits" in result.metadata
    assert "mean_samples_train" in result.metadata
    assert "mean_samples_val" in result.metadata
    assert "n_unique_values" in result.metadata


def test_behavioral_regression_multidim_target():
    """Test regression with multi-dimensional target (e.g., 2D cursor position)."""
    rng = np.random.default_rng(42)
    n_samples = 100
    latent_dim = 10

    latents = rng.normal(0, 1, size=(n_samples, latent_dim))
    weights_x = np.array([1.5, -1.0])
    weights_y = np.array([-0.5, 2.0])
    target_x = latents[:, :2] @ weights_x + rng.normal(0, 0.1, size=n_samples)
    target_y = latents[:, :2] @ weights_y + rng.normal(0, 0.1, size=n_samples)
    target = np.column_stack([target_x, target_y])

    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    trial_ids = np.repeat(np.arange(10), 10)
    behavioral_data = {"cursor_pos": target, "trial_id": trial_ids}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralRegressionMetric(
        target_field="cursor_pos",
        cv=3,
        train_drop_invalid=False,
        eval_drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    result = metric.compute(intermediates)

    assert result.raw_value > 0.4
    assert not np.isnan(result.raw_value)


def test_behavioral_regression_no_relationship():
    """Test regression when latents don't predict target (should give low R²)."""
    rng = np.random.default_rng(42)
    n_samples = 100

    latents = rng.normal(0, 1, size=(n_samples, 10))
    target = rng.normal(0, 1, size=n_samples)

    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    trial_ids = np.repeat(np.arange(10), 10)
    behavioral_data = {"cursor_vel": target, "trial_id": trial_ids}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralRegressionMetric(
        target_field="cursor_vel",
        cv=3,
        train_drop_invalid=False,
        eval_drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    result = metric.compute(intermediates)

    assert result.raw_value < 0.2, f"R² unexpectedly high: {result.raw_value}"


def test_behavioral_regression_insufficient_unique_values():
    """Test regression raises ValueError with < 2 unique values."""
    rng = np.random.default_rng(42)
    n_samples = 100
    latents = rng.normal(0, 1, size=(n_samples, 10))
    target = np.ones(n_samples)

    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    trial_ids = np.repeat(np.arange(10), 10)
    behavioral_data = {"cursor_pos": target, "trial_id": trial_ids}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralRegressionMetric(
        target_field="cursor_pos",
        train_drop_invalid=False,
        eval_drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    with pytest.raises(ValueError, match="< 2 unique values"):
        metric.compute(intermediates)


def test_behavioral_regression_missing_field():
    """Test regression raises ValueError when target field is missing."""
    rng = np.random.default_rng(42)
    latents = rng.normal(0, 1, size=(100, 10))

    behavioral_data = {"cursor_pos": rng.normal(0, 1, size=100)}
    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": behavioral_data,
        "token_bin_size_ms": 100.0,
    }

    metric = BehavioralRegressionMetric(target_field="nonexistent_field")
    with pytest.raises(ValueError, match="not found"):
        metric.compute(intermediates)


def test_behavioral_regression_missing_token_bin_size_ms():
    """Test regression raises informative error when token_bin_size_ms is missing."""
    rng = np.random.default_rng(42)
    n_samples = 50
    latents = rng.normal(0, 1, size=(n_samples, 10))
    target = rng.normal(0, 1, size=n_samples)
    trial_ids = np.repeat(np.arange(5), 10)
    behavioral_data = {"cursor_pos": target, "trial_id": trial_ids}

    # No token_bin_size_ms in intermediates
    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": behavioral_data,
        "token_timestamps": np.arange(n_samples, dtype=float) * 0.1,
    }

    metric = BehavioralRegressionMetric(
        target_field="cursor_pos", train_drop_invalid=False, eval_drop_invalid=False
    )
    with pytest.raises(ValueError, match="token_bin_size_ms is required"):
        metric.compute(intermediates)


def test_behavioral_regression_preprocessing_and_sample_trials():
    """Test regression preprocessing and qualitative sample-trial outputs.

    The synthetic target is a 2D cursor trajectory built as a linear function
    of the latent features, so ridge regression should recover it well. The
    test also injects invalid bins at the beginning and restricts evaluation to
    the reach phase, which exercises the new preprocessing path. Because the
    signal is linear and low-noise, we expect validation R^2 to be comfortably
    above 0.5 even after filtering and temporal concatenation.
    """
    rng = np.random.default_rng(0)
    n_trials = 8
    samples_per_trial = 20
    n_samples = n_trials * samples_per_trial

    latents = rng.normal(size=(n_samples, 6))
    cursor_x = 2.0 * latents[:, 0] - 0.5 * latents[:, 1]
    cursor_y = -1.0 * latents[:, 0] + 1.5 * latents[:, 2]
    cursor = np.column_stack([cursor_x, cursor_y])

    trial_ids = np.repeat(np.arange(n_trials), samples_per_trial)
    movement_phase = np.full(n_samples, MOVEMENT_PHASE_LABELS["reach_period"])
    trial_is_valid = np.ones(n_samples, dtype=int)

    trial_is_valid[:8] = 0
    movement_phase[:8] = MOVEMENT_PHASE_LABELS["inter_trial"]

    # 20ms token bins
    intermediates = _make_intermediates(
        latents,
        {
            "cursor_pos": cursor,
            "trial_id": trial_ids,
            "movement_phase": movement_phase,
            "trial_is_valid": trial_is_valid,
        },
        np.arange(n_samples, dtype=float) * 0.02,
        token_bin_size_ms=20.0,
    )

    # chunk_window_ms=80 with token_bin_size_ms=20 → 4 tokens/chunk → 24 features
    metric = BehavioralRegressionMetric(
        target_field="cursor_pos",
        train_drop_invalid=True,
        train_include_phases=["reach_period"],
        eval_drop_invalid=True,
        eval_include_phases=["reach_period"],
        chunk_window_ms=80.0,
        feature_reduction="none",
        sample_trial_count=4,
        cv=3,
    )
    result = metric.compute(intermediates)

    assert result.raw_value > 0.5
    assert result.metadata["chunk_window_ms"] == 80.0
    assert result.metadata["tokens_per_chunk"] == 4
    assert 1 <= len(result.metadata["sample_trials"]) <= 4
    assert {"trial_id", "timestamps", "y_true", "y_pred"} <= set(
        result.metadata["sample_trials"][0]
    )


def test_behavioral_regression_expands_chunked_trajectories_before_storing():
    """Chunked regression metadata should be stored at behavioral-bin resolution."""
    metric = BehavioralRegressionMetric(sample_trial_count=2)

    y_true = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    y_pred = y_true + 0.5
    timestamps = np.array([10.0, 20.0, 30.0])
    trial_ids = np.array([101, 101, 202])

    sample_trials = metric._build_sample_trials(
        y_val=y_true,
        y_pred=y_pred,
        timestamps=timestamps,
        trial_ids=trial_ids,
        k_per_chunk=2,
        behavioral_bin_size_ms=50.0,
    )

    by_trial = {trial["trial_id"]: trial for trial in sample_trials}
    trial_101 = by_trial[101]
    assert np.array(trial_101["y_true"]).shape == (4, 2)
    np.testing.assert_allclose(
        np.array(trial_101["timestamps"]),
        [10.0, 10.05, 20.0, 20.05],
    )
    np.testing.assert_allclose(
        np.array(trial_101["y_true"]),
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    )

    concat = metric._build_concatenated_trials(
        y_val=y_true,
        y_pred=y_pred,
        timestamps=timestamps,
        trial_ids=trial_ids,
        n_trials=2,
        k_per_chunk=2,
        behavioral_bin_size_ms=50.0,
    )

    assert concat is not None
    assert concat["trial_boundaries"] == [0, 4, 6]
    np.testing.assert_allclose(
        np.array(concat["timestamps"]),
        [10.0, 10.05, 20.0, 20.05, 30.0, 30.05],
    )
    assert np.array(concat["y_true"]).shape == (6, 2)


def test_behavioral_temporal_decoder_synthetic_temporal_context():
    """Temporal decoder should learn targets that depend on recent latent history."""
    rng = np.random.default_rng(0)
    n_trials = 12
    tokens_per_trial = 48
    latent_dim = 6

    latents = rng.normal(size=(n_trials, tokens_per_trial, latent_dim)).astype(np.float32)
    target = np.zeros((n_trials, tokens_per_trial), dtype=np.float32)
    for t in range(tokens_per_trial):
        current = latents[:, t, 0]
        prev_1 = latents[:, max(t - 1, 0), 1]
        prev_2 = latents[:, max(t - 2, 0), 2]
        target[:, t] = (
            1.2 * current - 0.7 * prev_1 + 0.5 * prev_2 + rng.normal(0.0, 0.03, size=n_trials)
        )

    trial_ids = np.repeat(np.arange(n_trials), tokens_per_trial)
    token_timestamps = np.concatenate(
        [
            trial_id * 10.0 + np.arange(tokens_per_trial, dtype=float) * 0.02
            for trial_id in range(n_trials)
        ]
    )
    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": {
            "cursor_pos": target.reshape(-1),
            "trial_id": trial_ids,
        },
        "token_timestamps": token_timestamps,
        "token_bin_size_ms": 20.0,
    }

    # num_epochs raised vs. original per-sequence loop: batched training
    # does 1 gradient step per epoch (vs N per-sequence steps), so more
    # epochs are needed for the same convergence on small synthetic data.
    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        hidden_channels=16,
        kernel_size=5,
        dropout=0.0,
        lr=5e-3,
        weight_decay=0.0,
        num_epochs=200,
        eval_every=10,
        random_seed=0,
        drop_invalid=False,
    )
    result = metric.compute(intermediates)

    assert result.raw_value > 0.8, f"Temporal decoder R² too low: {result.raw_value}"
    assert result.metadata["best_epoch"] >= 0
    assert result.metadata["n_samples_val"] > 0
    assert result.metadata["n_sequences_train"] > 0
    assert result.metadata["n_sequences_val"] > 0
    assert result.metadata["evaluation_split"] == "val"
    assert result.metadata["sequence_mode"] == "full_group_sequences"
    assert result.metadata["n_groups_train_split_by_time_gap"] == 0
    assert result.metadata["n_groups_val_split_by_time_gap"] == 0
    assert "per_channel_r2" in result.metadata
    assert result.metadata["sample_trials"]


def test_behavioral_temporal_decoder_with_ldns_tokenizer_latents():
    """Temporal decoder should run on actual LDNS latent outputs."""
    torch.manual_seed(0)
    spec = LDNSTokenizerSpec(
        n_neurons=8,
        window_length_seconds=0.32,
        bin_size_ms=10.0,
        hidden_dim=16,
        latent_dim=4,
        num_encoder_blocks=1,
        num_decoder_blocks=0,
        num_lin_per_mlp=2,
        bidirectional=False,
        latent_beta=0.1,
        latent_td_beta=0.2,
        latent_td_k=2,
        mask_prob=0.0,
        neuron_dropout_prob=0.0,
        validate_from_data=False,
    )
    tokenizer = LDNSTokenizer(spec).eval()
    batch = make_dummy_binned_input(
        n_neurons=spec.n_neurons,
        time_bins=int(spec.window_length_seconds * 1000 / spec.bin_size_ms),
        bin_size_ms=spec.bin_size_ms,
        batch_size=12,
    )
    with torch.no_grad():
        output = tokenizer.reconstruct_from_input(batch, return_latents=True)

    intermediates = _make_temporal_decoder_intermediates_from_output(
        output,
        temporal_history=True,
    )
    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        hidden_channels=8,
        kernel_size=3,
        dropout=0.0,
        lr=5e-3,
        weight_decay=0.0,
        num_epochs=12,
        eval_every=2,
        random_seed=0,
        drop_invalid=False,
    )
    result = metric.compute(intermediates)

    assert np.isfinite(result.raw_value)
    assert result.metadata["latent_layout"] == "windowed_unit_latents"
    assert result.metadata["n_sequences_train"] > 0
    assert result.metadata["n_sequences_val"] > 0


def test_behavioral_temporal_decoder_with_simple_vae_latents():
    """Temporal decoder should raise for compress-time tokenizers with single-token windows."""
    torch.manual_seed(0)
    spec = SimpleVAETokenizerSpec(
        latent_dim=4,
        hidden_dims=(16, 8),
        validate_from_data=False,
    )
    tokenizer = SimpleVAETokenizer(spec).eval()
    batch = make_dummy_binned_input(
        n_neurons=1,
        time_bins=int(spec.window_length_seconds * 1000 / spec.bin_size_ms),
        bin_size_ms=spec.bin_size_ms,
        batch_size=20,
    )
    with torch.no_grad():
        output = tokenizer.reconstruct_from_input(batch, return_latents=True)

    intermediates = _make_temporal_decoder_intermediates_from_output(
        output,
        temporal_history=False,
    )
    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        hidden_channels=8,
        kernel_size=1,
        dropout=0.0,
        lr=1e-2,
        weight_decay=0.0,
        num_epochs=10,
        eval_every=1,
        random_seed=0,
        drop_invalid=False,
    )
    with pytest.raises(ValueError, match="all training sequences have length <= 1"):
        metric.compute(intermediates)


def test_preprocess_downstream_data_ragged_windowed_latents():
    """Ragged per-window latents should flatten in window order."""
    latents = [
        np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        (np.arange(6, dtype=np.float32) + 100).reshape(1, 3, 2),
        (np.arange(18, dtype=np.float32) + 200).reshape(3, 3, 2),
    ]
    token_timestamps = np.arange(6, dtype=float) * 0.02
    behavioral_data_token_res = {
        "cursor_pos": np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        "trial_id": np.array([0, 0, 1, 2, 2, 2]),
    }

    X, behavior_tok, _behavior_input, _chunk_ts, _chunk_idx, metadata = preprocess_downstream_data(
        latents,
        behavioral_data_token_res,
        token_timestamps,
        token_bin_size_ms=20.0,
        chunk_window_ms=20.0,
    )

    expected = np.concatenate(
        [
            latents[0].reshape(2, 6),
            latents[1].reshape(1, 6),
            latents[2].reshape(3, 6),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(
        behavior_tok["cursor_pos"], behavioral_data_token_res["cursor_pos"]
    )
    assert metadata["latent_layout"] == "ragged_window_tokens"
    assert metadata["window_token_counts"] == [2, 1, 3]
    assert metadata["is_discrete_index_like"] is False


def test_preprocess_downstream_data_flattens_discrete_token_grid():
    """Discrete token grids should become one scalar feature per token."""
    latents = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    token_timestamps = np.arange(6, dtype=float) * 0.05
    behavioral_data_token_res = {
        "cursor_pos": np.linspace(0.0, 1.0, 6, dtype=np.float32),
        "trial_id": np.array([0, 0, 0, 1, 1, 1]),
    }

    X, _behavior_tok, _behavior_input, _chunk_ts, _chunk_idx, metadata = (
        preprocess_downstream_data(
            latents,
            behavioral_data_token_res,
            token_timestamps,
            token_bin_size_ms=50.0,
            chunk_window_ms=50.0,
        )
    )

    assert X.shape == (6, 1)
    np.testing.assert_array_equal(X[:, 0], np.array([1, 2, 3, 4, 5, 6]))
    assert metadata["latent_layout"] == "token_grid_scalar_feature"
    assert metadata["window_token_counts"] == [3, 3]
    assert metadata["is_discrete_index_like"] is True


def test_preprocess_chunk_equals_token_bin_preserves_token_level_data():
    """chunk_window_ms == token_bin_size_ms should produce 1-token chunks,
    preserving the same X values and behavioral data as unfiltered input.

    """
    rng = np.random.default_rng(99)
    n_tokens = 50
    latent_dim = 8
    token_bin_size_ms = 20.0

    latents = rng.normal(size=(n_tokens, latent_dim)).astype(np.float32)
    token_timestamps = np.arange(n_tokens, dtype=float) * (token_bin_size_ms / 1000)
    behavioral_data_token_res = {
        "cursor_pos": rng.normal(size=n_tokens).astype(np.float32),
        "trial_id": np.repeat(np.arange(5), 10),
    }

    X, behavior_tok, behavior_input, chunk_ts, chunk_idx, metadata = preprocess_downstream_data(
        latents,
        behavioral_data_token_res,
        token_timestamps,
        token_bin_size_ms=token_bin_size_ms,
        chunk_window_ms=token_bin_size_ms,
    )

    # 1 token per chunk → X should have same number of rows, each row = 1 token's features
    assert X.shape == (n_tokens, latent_dim), f"Expected ({n_tokens}, {latent_dim}), got {X.shape}"
    np.testing.assert_array_equal(X, latents)

    # Behavioral data at token resolution should be passed through unchanged
    assert behavior_tok is not None
    np.testing.assert_array_equal(
        behavior_tok["cursor_pos"], behavioral_data_token_res["cursor_pos"]
    )
    np.testing.assert_array_equal(behavior_tok["trial_id"], behavioral_data_token_res["trial_id"])

    # Chunk timestamps should match token timestamps
    np.testing.assert_allclose(chunk_ts, token_timestamps, atol=1e-9)

    # Each chunk index should be its own position (0, 1, 2, ...)
    np.testing.assert_array_equal(chunk_idx, np.arange(n_tokens))

    assert metadata["tokens_per_chunk"] == 1


def test_preprocess_downstream_data_invariant_to_token_bin_size_expected_but_fails_today():
    """Expected property (currently false): same chunk window duration ⇒ same chunks/labels.

    This is false because the token resolution influences the fraction labels in the chunk,
    and all other filters applied at token level. So we end up with different chunks being
    included/excluded if we have different token resolutions (but same chunk size).

    Similarly, we would like (but false:)
    same chunk window duration, same token bin size ⇒ same chunks/labels.
    This is true for classification. This is _not_ true for regression, where chunk
    filtering depends on behavioral data _at target resolution_.
    If tokenizers run on different input resolution, they will have different chunks
    excluded/included, _even_ if they have the same token bin size.

    The test:

    We construct two "tokenizers" over the same 1s recording with chunk_window_ms=200ms:
    - one produces 1ms tokens
    - the other produces 5ms tokens

    We then apply a token-level invalid mask defined in continuous time such that the
    chunk drop decision is sensitive to sampling resolution (fraction of invalid tokens
    per chunk crosses the 0.25 threshold for 1ms, but not for 5ms when using token centers).

    This test is intended to FAIL on the current implementation, documenting the problem.
    """
    rng = np.random.default_rng(0)
    chunk_window_ms = 200.0

    def _make_inputs(*, token_bin_ms: float):
        duration_ms = 1000.0
        n_tokens = int(duration_ms / token_bin_ms)
        # Token centers (important for the failure case)
        token_ts_ms = (np.arange(n_tokens, dtype=float) * token_bin_ms) + (token_bin_ms / 2.0)
        token_timestamps = token_ts_ms / 1000.0

        latents = rng.normal(size=(n_tokens, 3)).astype(np.float32)

        # Invalid for first 51ms of the *first* 200ms chunk only; the second chunk
        # is fully valid so the test always has at least one retained chunk to compare.
        chunk_id = np.floor(token_ts_ms / chunk_window_ms).astype(int)
        invalid = (((token_ts_ms % chunk_window_ms) < 51.0) & (chunk_id == 0)).astype(int)
        trial_is_valid = 1 - invalid

        behavioral_data_token_res = {
            "trial_is_valid": trial_is_valid,
        }
        return latents, behavioral_data_token_res, token_timestamps

    lat_1, beh_1, ts_1 = _make_inputs(token_bin_ms=1.0)
    lat_5, beh_5, ts_5 = _make_inputs(token_bin_ms=5.0)

    X1, _beh_tok1, _beh_in1, _chunk_ts1, chunk_idx1, meta1 = preprocess_downstream_data(
        lat_1,
        beh_1,
        ts_1,
        token_bin_size_ms=1.0,
        chunk_window_ms=chunk_window_ms,
        drop_invalid=True,
        chunk_drop_threshold=0.25,
    )
    X5, _beh_tok5, _beh_in5, _chunk_ts5, chunk_idx5, meta5 = preprocess_downstream_data(
        lat_5,
        beh_5,
        ts_5,
        token_bin_size_ms=5.0,
        chunk_window_ms=chunk_window_ms,
        drop_invalid=True,
        chunk_drop_threshold=0.25,
    )

    # Different token bin sizes can drop different chunks when
    # drop_invalid=True because we drop chunks based on the *fraction of invalid tokens*.
    # Example: in a 200ms chunk with 51ms invalid time,
    # - 1ms tokens: 51/200 = 0.255 > 0.25 ⇒ dropped
    # - 5ms tokens (centers): 10/40 = 0.25 ≤ 0.25 ⇒ kept
    assert not np.array_equal(chunk_idx1, chunk_idx5)
    assert X1.shape[0] != X5.shape[0]
    assert meta1.get("n_chunks_dropped_by_filter_detail") != meta5.get(
        "n_chunks_dropped_by_filter_detail"
    )


def test_preprocess_downstream_data_invariant_to_token_bin_size_when_drop_invalid_false():
    """When drop_invalid=False, chunk retention should not depend on token bin size."""
    rng = np.random.default_rng(0)
    chunk_window_ms = 200.0

    def _make_inputs(*, token_bin_ms: float):
        duration_ms = 1000.0
        n_tokens = int(duration_ms / token_bin_ms)
        token_ts_ms = (np.arange(n_tokens, dtype=float) * token_bin_ms) + (token_bin_ms / 2.0)
        token_timestamps = token_ts_ms / 1000.0

        latents = rng.normal(size=(n_tokens, 3)).astype(np.float32)

        # Same continuous-time invalid interval as above, but drop_invalid=False so it should not
        # matter.
        chunk_id = np.floor(token_ts_ms / chunk_window_ms).astype(int)
        invalid = (((token_ts_ms % chunk_window_ms) < 51.0) & ((chunk_id % 2) == 0)).astype(int)
        trial_is_valid = 1 - invalid

        behavioral_data_token_res = {
            "trial_is_valid": trial_is_valid,
        }
        return latents, behavioral_data_token_res, token_timestamps

    lat_1, beh_1, ts_1 = _make_inputs(token_bin_ms=1.0)
    lat_5, beh_5, ts_5 = _make_inputs(token_bin_ms=5.0)

    X1, _beh_tok1, _beh_in1, _chunk_ts1, chunk_idx1, _meta1 = preprocess_downstream_data(
        lat_1,
        beh_1,
        ts_1,
        token_bin_size_ms=1.0,
        chunk_window_ms=chunk_window_ms,
        drop_invalid=False,
        chunk_drop_threshold=0.25,
    )
    X5, _beh_tok5, _beh_in5, _chunk_ts5, chunk_idx5, _meta5 = preprocess_downstream_data(
        lat_5,
        beh_5,
        ts_5,
        token_bin_size_ms=5.0,
        chunk_window_ms=chunk_window_ms,
        drop_invalid=False,
        chunk_drop_threshold=0.25,
    )

    np.testing.assert_array_equal(chunk_idx1, chunk_idx5)
    assert X1.shape[0] == X5.shape[0]


def test_behavioral_temporal_decoder_rejects_single_token_per_window_latents():
    """Single-token-per-window latents should raise (temporal decoding is meaningless)."""
    rng = np.random.default_rng(7)
    n_trials = 16
    latents = rng.normal(size=(n_trials, 1, 4)).astype(np.float32)
    target = (1.5 * latents[:, 0, 0] - 0.7 * latents[:, 0, 1]).astype(np.float32)

    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": {
            "cursor_pos": target,
            "trial_id": np.arange(n_trials),
        },
        "token_timestamps": np.arange(n_trials, dtype=float) * 0.05,
        "token_bin_size_ms": 50.0,
    }

    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        hidden_channels=8,
        kernel_size=1,
        dropout=0.0,
        lr=1e-2,
        weight_decay=0.0,
        drop_invalid=False,
        num_epochs=8,
        eval_every=1,
        random_seed=0,
    )
    with pytest.raises(ValueError, match="all training sequences have length <= 1"):
        metric.compute(intermediates)


def test_behavioral_temporal_decoder_rejects_discrete_token_indices():
    """Raw discrete token ids should be rejected by the temporal decoder."""
    rng = np.random.default_rng(8)
    n_trials = 12
    tokens_per_trial = 4
    latents = rng.integers(0, 16, size=(n_trials, tokens_per_trial), dtype=np.int64)
    target = rng.normal(size=n_trials * tokens_per_trial).astype(np.float32)
    token_timestamps = np.concatenate(
        [
            trial_id * 5.0 + np.arange(tokens_per_trial, dtype=float) * 0.02
            for trial_id in range(n_trials)
        ]
    )

    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": {
            "cursor_pos": target,
            "trial_id": np.repeat(np.arange(n_trials), tokens_per_trial),
        },
        "token_timestamps": token_timestamps,
        "token_bin_size_ms": 20.0,
    }

    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        drop_invalid=False,
        num_epochs=2,
        eval_every=1,
    )
    with pytest.raises(ValueError, match="continuous latent features|discrete token indices"):
        metric.compute(intermediates)


def test_evaluation_pipeline_skips_temporal_decoder_for_discrete_token_indices():
    """Pipeline should skip, not fail, when temporal decoder sees raw token ids."""
    n_trials = 10
    tokens_per_trial = 3
    latents = np.arange(n_trials * tokens_per_trial, dtype=np.int64).reshape(
        n_trials, tokens_per_trial
    )
    behavioral_data_token_res = {
        "cursor_pos": np.linspace(-1.0, 1.0, n_trials * tokens_per_trial, dtype=np.float32),
        "trial_id": np.repeat(np.arange(n_trials), tokens_per_trial),
    }
    token_timestamps = np.concatenate(
        [
            trial_id * 2.0 + np.arange(tokens_per_trial, dtype=float) * 0.05
            for trial_id in range(n_trials)
        ]
    )

    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        drop_invalid=False,
        num_epochs=2,
        eval_every=1,
    )
    pipeline = EvaluationPipeline(
        metrics=[metric],
        config=EvaluationConfig(),
    )
    original = EvalData.from_tensor(
        torch.zeros((2, 20), dtype=torch.float32),
        data_type=DataType.SPIKES,
        bin_size_seconds=0.01,
        duration=0.2,
    )
    reconstructed = EvalData.from_tensor(
        torch.zeros((2, 20), dtype=torch.float32),
        data_type=DataType.SPIKES,
        bin_size_seconds=0.01,
        duration=0.2,
    )

    report = pipeline.run(
        original,
        reconstructed,
        latents=latents,
        behavioral_data_token_res=behavioral_data_token_res,
        token_timestamps=token_timestamps,
        token_bin_size_ms=20.0,
    )

    assert "behavioral_temporal_decoder" in report.skipped_metrics
    assert "discrete token indices" in report.skipped_metrics["behavioral_temporal_decoder"]


def test_behavioral_classification_synthetic():
    """Test classification metric with synthetic data where latents predict classes."""
    rng = np.random.default_rng(42)
    n_samples = 300
    latent_dim = 10

    n_per_class = n_samples // 3

    latents_0 = rng.normal(5, 0.3, size=(n_per_class, latent_dim))
    latents_0[:, 1:] = rng.normal(0, 0.3, size=(n_per_class, latent_dim - 1))

    latents_1 = rng.normal(-5, 0.3, size=(n_per_class, latent_dim))
    latents_1[:, 1:] = rng.normal(0, 0.3, size=(n_per_class, latent_dim - 1))

    latents_2 = rng.normal(0, 0.3, size=(n_per_class, latent_dim))
    latents_2[:, 0] = rng.normal(0, 0.3, size=n_per_class)
    latents_2[:, 1] = rng.normal(5, 0.3, size=n_per_class)

    latents = np.vstack([latents_0, latents_1, latents_2])
    labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    shuffle_idx = rng.permutation(n_samples)
    latents = latents[shuffle_idx]
    labels = labels[shuffle_idx]

    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    behavioral_data = {"trial_target_id": labels}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralClassificationMetric(
        target_field="trial_target_id",
        grouping_field=None,
        cv=3,
        drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    result = metric.compute(intermediates)

    assert result.raw_value > 0.95, f"Accuracy too low: {result.raw_value}"
    assert "n_splits" in result.metadata
    assert "chance_accuracy" in result.metadata
    assert 0.3 < result.metadata["chance_accuracy"] < 0.4


def test_behavioral_classification_binary():
    """Test classification with binary labels."""
    rng = np.random.default_rng(42)
    n_samples = 100

    latents_0 = rng.normal(1, 0.3, size=(50, 10))
    latents_1 = rng.normal(-1, 0.3, size=(50, 10))
    latents = np.vstack([latents_0, latents_1])
    labels = np.array([0] * 50 + [1] * 50)

    shuffle_idx = rng.permutation(n_samples)
    latents = latents[shuffle_idx]
    labels = labels[shuffle_idx]

    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    behavioral_data = {"trial_result": labels}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralClassificationMetric(
        target_field="trial_result",
        grouping_field=None,
        cv=3,
        drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    result = metric.compute(intermediates)

    assert result.raw_value > 0.85
    assert "chance_accuracy" in result.metadata


def test_behavioral_classification_insufficient_classes():
    """Test classification raises ValueError with < 2 unique classes."""
    rng = np.random.default_rng(42)
    n_samples = 100
    latents = rng.normal(0, 1, size=(n_samples, 10))
    labels = np.zeros(n_samples, dtype=int)

    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    behavioral_data = {"trial_target_id": labels}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralClassificationMetric(
        target_field="trial_target_id",
        grouping_field=None,
        drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    with pytest.raises(ValueError, match="< 2 classes"):
        metric.compute(intermediates)


def test_behavioral_classification_trial_result_special_filtering():
    """Test the trial-result special case used in the notebook workflow.

    The synthetic data contains trials with a reach half and a return half.
    Only the return half carries the informative class-dependent latent shift,
    while the reach half is intentionally uninformative and labelled with a
    third value that should be filtered out. This checks that return-only phase
    filtering, result-value filtering, and trial-wise grouping work together.
    """
    rng = np.random.default_rng(1)
    n_trials = 20
    samples_per_trial = 30

    latents = []
    trial_result = []
    trial_id = []
    movement_phase = []
    trial_is_valid = []
    timestamps = []

    for trial in range(n_trials):
        result = 0 if trial % 2 == 0 else 2
        for sample in range(samples_per_trial):
            in_return = sample >= samples_per_trial // 2
            phase = (
                MOVEMENT_PHASE_LABELS["return_period"]
                if in_return
                else MOVEMENT_PHASE_LABELS["reach_period"]
            )
            center = 2.0 if result == 2 else -2.0
            lat = rng.normal(loc=center if in_return else 0.0, scale=0.2, size=5)
            latents.append(lat)
            trial_result.append(result if in_return else 1)
            trial_id.append(trial)
            movement_phase.append(phase)
            trial_is_valid.append(1)
            timestamps.append(len(timestamps) * 0.02)

    # 20ms token bins, chunk_window_ms=60 → 3 tokens/chunk → 15 features
    intermediates = _make_intermediates(
        np.asarray(latents),
        {
            "trial_result": np.asarray(trial_result),
            "trial_id": np.asarray(trial_id),
            "movement_phase": np.asarray(movement_phase),
            "trial_is_valid": np.asarray(trial_is_valid),
        },
        np.asarray(timestamps),
        token_bin_size_ms=20.0,
    )

    metric = BehavioralClassificationMetric(
        target_field="trial_result",
        grouping_field="trial_id",
        grouping_transition="change",
        drop_invalid=True,
        include_phases=["return_period"],
        filter_field="trial_result",
        filter_values=[0, 2],
        chunk_window_ms=60.0,
        feature_reduction="none",
        cv=3,
    )
    result = metric.compute(intermediates)

    assert result.raw_value > 0.8
    assert result.metadata["filter_field"] == "trial_result"
    assert result.metadata["include_phases"] == ["return_period"]


def test_behavioral_classification_no_relationship():
    """Test classification when latents don't predict labels (should give ~chance)."""
    rng = np.random.default_rng(42)
    n_samples = 150

    latents = rng.normal(0, 1, size=(n_samples, 10))
    labels = rng.integers(0, 3, size=n_samples)

    token_timestamps = np.linspace(0, n_samples * 0.1, n_samples)
    behavioral_data = {"trial_target_id": labels}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralClassificationMetric(
        target_field="trial_target_id",
        grouping_field=None,
        cv=3,
        drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    result = metric.compute(intermediates)

    chance = result.metadata["chance_accuracy"]
    assert result.raw_value < chance + 0.15, f"Accuracy unexpectedly high: {result.raw_value}"


def test_temporal_alignment_mismatch_returns_error():
    """Test that misaligned latents and behavioral data raises ValueError."""
    rng = np.random.default_rng(42)

    n_latents = 100
    latents = rng.normal(0, 1, size=(n_latents, 10))

    # Target has 120 samples (misaligned)
    n_target = 120
    target = rng.normal(0, 1, size=n_target)

    token_timestamps = np.linspace(0, n_latents * 0.1, n_latents)
    trial_ids = np.repeat(np.arange(10), 10)
    behavioral_data = {"cursor_pos": target, "trial_id": trial_ids}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=100.0
    )

    metric = BehavioralRegressionMetric(
        target_field="cursor_pos",
        cv=3,
        train_drop_invalid=False,
        eval_drop_invalid=False,
        chunk_window_ms=100.0,
        feature_reduction="none",
    )
    with pytest.raises(ValueError, match="length.*!=|[Mm]isalignment"):
        metric.compute(intermediates)


def test_behavioral_temporal_decoder_temporal_alignment_mismatch_returns_error():
    rng = np.random.default_rng(42)
    latents = rng.normal(0, 1, size=(100, 10))
    target = rng.normal(0, 1, size=100)
    trial_ids = np.repeat(np.arange(10), 10)

    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": {"cursor_pos": target, "trial_id": trial_ids},
        "token_timestamps": np.linspace(0, 10, 90),
        "token_bin_size_ms": 20.0,
    }

    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        drop_invalid=False,
        num_epochs=2,
        eval_every=1,
    )
    with pytest.raises(ValueError, match="length.*!=|[Mm]isalignment"):
        metric.compute(intermediates)


def test_behavioral_temporal_decoder_requires_two_groups():
    rng = np.random.default_rng(1)
    n_trials = 1
    tokens_per_trial = 40
    latents = rng.normal(size=(n_trials, tokens_per_trial, 4))
    target = latents[..., 0] - 0.5 * latents[..., 1]
    trial_ids = np.repeat(np.arange(n_trials), tokens_per_trial)
    token_timestamps = np.arange(n_trials * tokens_per_trial, dtype=float) * 0.02

    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": {"cursor_pos": target.reshape(-1), "trial_id": trial_ids},
        "token_timestamps": token_timestamps,
        "token_bin_size_ms": 20.0,
    }

    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        drop_invalid=False,
        num_epochs=2,
        eval_every=1,
    )
    with pytest.raises(ValueError, match="at least 2 contiguous groups"):
        metric.compute(intermediates)


def test_behavioral_temporal_decoder_short_groups_do_not_require_windows():
    rng = np.random.default_rng(2)
    n_trials = 6
    tokens_per_trial = 4
    latents = rng.normal(size=(n_trials, tokens_per_trial, 3))
    target = latents[..., 0]
    trial_ids = np.repeat(np.arange(n_trials), tokens_per_trial)
    token_timestamps = np.arange(n_trials * tokens_per_trial, dtype=float) * 0.02

    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": {"cursor_pos": target.reshape(-1), "trial_id": trial_ids},
        "token_timestamps": token_timestamps,
        "token_bin_size_ms": 20.0,
    }

    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        hidden_channels=8,
        kernel_size=5,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        drop_invalid=False,
        num_epochs=4,
        eval_every=1,
    )
    result = metric.compute(intermediates)

    assert np.isfinite(result.raw_value)
    assert result.metadata["sequence_mode"] == "full_group_sequences"
    assert result.metadata["n_sequences_train"] > 0
    assert result.metadata["n_sequences_val"] > 0


def test_behavioral_temporal_decoder_splits_groups_on_timestamp_gaps():
    rng = np.random.default_rng(3)
    n_trials = 6
    tokens_per_trial = 6
    latents = rng.normal(size=(n_trials, tokens_per_trial, 3)).astype(np.float32)
    target = latents[..., 0] - 0.25 * latents[..., 1]
    trial_ids = np.repeat(np.arange(n_trials), tokens_per_trial)

    token_timestamps = []
    for trial_id in range(n_trials):
        base = trial_id * 10.0
        token_timestamps.extend(
            [
                base + 0.00,
                base + 0.02,
                base + 0.04,
                base + 0.30,
                base + 0.32,
                base + 0.34,
            ]
        )
    token_timestamps = np.asarray(token_timestamps)

    intermediates = {
        "latents": latents,
        "behavioral_data_token_res": {"cursor_pos": target.reshape(-1), "trial_id": trial_ids},
        "token_timestamps": token_timestamps,
        "token_bin_size_ms": 20.0,
    }

    metric = BehavioralTemporalDecoderMetric(
        target_field="cursor_pos",
        hidden_channels=8,
        kernel_size=5,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        drop_invalid=False,
        num_epochs=4,
        eval_every=1,
    )
    result = metric.compute(intermediates)

    assert np.isfinite(result.raw_value)
    assert result.metadata["n_groups_train_split_by_time_gap"] > 0
    assert result.metadata["n_groups_val_split_by_time_gap"] > 0
    assert result.metadata["n_sequences_train"] > result.metadata["n_groups_train"]
    assert result.metadata["n_sequences_val"] > result.metadata["n_groups_val"]


def test_instantiate_repeated_downstream_metrics_from_config():
    """Config should support repeated downstream metric instances."""
    metrics = instantiate_metrics_from_config(
        {
            "cursor_pos_regression": {
                "metric_class": "behavioral_regression",
                "target_field": "cursor_pos",
                "train_drop_invalid": True,
            },
            "cursor_vel_regression": {
                "metric_class": "behavioral_regression",
                "target_field": "cursor_vel",
                "train_drop_invalid": True,
            },
            "cursor_pos_temporal_decoder": {
                "metric_class": "behavioral_temporal_decoder",
                "target_field": "cursor_pos",
                "drop_invalid": True,
                "num_epochs": 2,
                "eval_every": 1,
            },
            "trial_result_classification": {
                "metric_class": "behavioral_classification",
                "target_field": "trial_result",
                "include_phases": ["return_period"],
                "filter_field": "trial_result",
                "filter_values": [0, 2],
            },
        }
    )

    assert [name for name, _metric in metrics] == [
        "cursor_pos_regression",
        "cursor_vel_regression",
        "cursor_pos_temporal_decoder",
        "trial_result_classification",
    ]

    assert isinstance(metrics[2][1], BehavioralTemporalDecoderMetric)


# ── Feature reduction integration test ─────────────────────────────────


def test_regression_with_truncation():
    """Regression with default truncation works."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    latent_dim = 10

    latents = rng.normal(0, 1, size=(n_samples, latent_dim))
    weights = np.array([2.0, -1.5, 1.0])
    target = latents[:, :3] @ weights + rng.normal(0, 0.1, size=n_samples)

    # 10ms token bins, 100ms chunk → 10 tokens/chunk → 100 features → truncated to 32
    # 100 chunks from 1000 tokens — enough for k-fold with 32 features
    token_timestamps = np.arange(n_samples, dtype=float) * 0.01
    trial_ids = np.repeat(np.arange(100), 10)
    behavioral_data = {"cursor_pos": target, "trial_id": trial_ids}
    intermediates = _make_intermediates(
        latents, behavioral_data, token_timestamps, token_bin_size_ms=10.0
    )

    metric = BehavioralRegressionMetric(
        target_field="cursor_pos",
        cv=3,
        train_drop_invalid=False,
        eval_drop_invalid=False,
        chunk_window_ms=100.0,
        n_features=32,
        feature_reduction="truncate",
    )
    result = metric.compute(intermediates)

    assert result.metadata["n_features_after_reduction"] == 32
    assert result.metadata["feature_reduction"] == "truncate"
    # The signal is in the first 3 dims of each token, which survive truncation
    # (threshold lowered for k-fold CV on small data)
    assert result.raw_value > 0.1


# ── Threshold filtering tests ────────────────────────────────────────


class TestChunkThresholdFiltering:
    """Core tests for 25% threshold chunk filtering.

    Validates that preprocess_downstream_data drops chunks only when
    >threshold fraction of tokens fail a filter, not all-or-nothing.
    """

    def _make_data(self, n_tokens=20, token_bin_size_ms=10.0, chunk_window_ms=100.0):
        """Build minimal inputs for preprocess_downstream_data.


        Returns 20 tokens at 10ms each, forming 2 chunks of 10 tokens.
        All behavioral fields default to "passing" values.
        """
        D = 4
        latents = np.ones((n_tokens, D))
        timestamps = np.arange(n_tokens, dtype=float) * (token_bin_size_ms / 1000.0)
        # Default: all tokens in reach_period, all valid
        from ntt.core.data.components.behavior_transforms import MOVEMENT_PHASE_LABELS

        behavior = {
            "movement_phase": np.full(
                n_tokens, MOVEMENT_PHASE_LABELS["reach_period"], dtype=float
            ),
            "trial_is_valid": np.ones(n_tokens, dtype=float),
            "trial_id": np.zeros(n_tokens, dtype=float),
        }
        return latents, behavior, timestamps

    def test_phase_filter_below_threshold_keeps_chunk(self):
        """Chunk with 20% failing tokens (< 25% threshold) should be kept."""
        latents, behavior, timestamps = self._make_data()

        # Chunk 0: tokens 0-9. Set 2 of 10 tokens (20%) to inter_trial
        behavior["movement_phase"][:2] = MOVEMENT_PHASE_LABELS["inter_trial"]

        X, _, _, _, _, meta = preprocess_downstream_data(
            latents,
            behavior,
            timestamps,
            token_bin_size_ms=10.0,
            include_phases=["reach_period"],
            chunk_window_ms=100.0,
            chunk_drop_threshold=0.25,
        )
        # Both chunks should survive (chunk 0: 20% fail, chunk 1: 0% fail)
        assert X.shape[0] == 2
        assert meta["n_chunks_dropped_by_filter"] == 0

    def test_phase_filter_above_threshold_drops_chunk(self):
        """Chunk with 30% failing tokens (> 25% threshold) should be dropped."""
        latents, behavior, timestamps = self._make_data()

        # Chunk 0: tokens 0-9. Set 3 of 10 tokens (30%) to inter_trial
        behavior["movement_phase"][:3] = MOVEMENT_PHASE_LABELS["inter_trial"]

        X, _, _, _, _, meta = preprocess_downstream_data(
            latents,
            behavior,
            timestamps,
            token_bin_size_ms=10.0,
            include_phases=["reach_period"],
            chunk_window_ms=100.0,
            chunk_drop_threshold=0.25,
        )
        # Chunk 0 dropped (30% > 25%), chunk 1 kept
        assert X.shape[0] == 1
        assert meta["n_chunks_dropped_by_filter"] == 1
        assert meta["n_chunks_dropped_by_filter_detail"]["phase"] == 1

    def test_exactly_at_threshold_keeps_chunk(self):
        """Chunk with exactly 25% failing tokens (== threshold) should be kept.

        The condition is fail_fraction <= threshold, so equality keeps.
        """
        # 20 tokens, 10ms each, chunk_window=200ms → 1 chunk of 20 tokens
        latents, behavior, timestamps = self._make_data(n_tokens=20, chunk_window_ms=200.0)

        # Set 5 of 20 tokens (25%) to inter_trial
        behavior["movement_phase"][:5] = MOVEMENT_PHASE_LABELS["inter_trial"]

        X, _, _, _, _, meta = preprocess_downstream_data(
            latents,
            behavior,
            timestamps,
            token_bin_size_ms=10.0,
            include_phases=["reach_period"],
            chunk_window_ms=200.0,
            chunk_drop_threshold=0.25,
        )
        assert X.shape[0] == 1
        assert meta["n_chunks_dropped_by_filter"] == 0

    def test_validity_filter_uses_threshold(self):
        """drop_invalid should also use threshold, not all-or-nothing."""
        latents, behavior, timestamps = self._make_data()

        # Chunk 0: 2 of 10 tokens (20%) invalid → kept
        behavior["trial_is_valid"][:2] = 0

        X, _, _, _, _, meta = preprocess_downstream_data(
            latents,
            behavior,
            timestamps,
            token_bin_size_ms=10.0,
            drop_invalid=True,
            chunk_window_ms=100.0,
            chunk_drop_threshold=0.25,
        )
        assert X.shape[0] == 2
        assert meta["n_chunks_dropped_by_filter"] == 0

    def test_independent_filters_combine(self):
        """Phase and validity filters are applied independently.

        A chunk can survive one filter but fail another.
        """
        latents, behavior, timestamps = self._make_data()

        # Chunk 0: 20% phase fail AND 20% validity fail on DIFFERENT tokens
        # → each filter alone passes (20% < 25%), chunk should survive
        behavior["movement_phase"][:2] = MOVEMENT_PHASE_LABELS["inter_trial"]
        behavior["trial_is_valid"][5:7] = 0

        X, _, _, _, _, meta = preprocess_downstream_data(
            latents,
            behavior,
            timestamps,
            token_bin_size_ms=10.0,
            include_phases=["reach_period"],
            drop_invalid=True,
            chunk_window_ms=100.0,
            chunk_drop_threshold=0.25,
        )
        # Both chunks should survive
        assert X.shape[0] == 2
        assert meta["n_chunks_dropped_by_filter"] == 0

    def test_apply_chunk_threshold_filter_directly(self):
        """Direct unit test of _apply_chunk_threshold_filter."""
        # 2 chunks of 4 tokens each
        token_mask = np.array(
            [
                True,
                True,
                False,
                True,  # chunk 0: 1/4 fail = 25%
                True,
                False,
                False,
                True,
            ]
        )  # chunk 1: 2/4 fail = 50%
        all_token_indices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

        # threshold=0.25: chunk 0 keeps (25% == threshold), chunk 1 drops (50%)
        keep = _apply_chunk_threshold_filter(token_mask, all_token_indices, 0.25)
        assert keep[0]
        assert not keep[1]

        # threshold=0.50: both keep (50% == threshold)
        keep = _apply_chunk_threshold_filter(token_mask, all_token_indices, 0.50)
        assert keep[0]
        assert keep[1]


# ── chunk_window_ms sweep tests ────────────────────────────────────────


class TestChunkWindowSweep:
    """Core tests for chunk_window_ms sweep (list of values)."""

    def _make_regression_data(self, n_samples=200, latent_dim=10, token_bin_ms=50.0):
        """Helper: create synthetic regression data with predictable relationship."""
        rng = np.random.default_rng(42)
        latents = rng.normal(0, 1, size=(n_samples, latent_dim))
        weights = np.array([2.0, -1.5, 1.0])
        target = latents[:, :3] @ weights + rng.normal(0, 0.1, size=n_samples)
        token_timestamps = np.linspace(0, n_samples * token_bin_ms / 1000, n_samples)
        trial_ids = np.repeat(np.arange(n_samples // 10), 10)
        behavioral_data = {"cursor_pos": target, "trial_id": trial_ids}
        return _make_intermediates(
            latents, behavioral_data, token_timestamps, token_bin_size_ms=token_bin_ms
        )

    def _make_classification_data(self, n_samples=300, latent_dim=10, token_bin_ms=50.0):
        """Helper: create synthetic classification data."""
        rng = np.random.default_rng(42)
        n_per_class = n_samples // 3
        latents_0 = rng.normal(5, 0.3, size=(n_per_class, latent_dim))
        latents_0[:, 1:] = rng.normal(0, 0.3, size=(n_per_class, latent_dim - 1))
        latents_1 = rng.normal(-5, 0.3, size=(n_per_class, latent_dim))
        latents_1[:, 1:] = rng.normal(0, 0.3, size=(n_per_class, latent_dim - 1))
        latents_2 = rng.normal(0, 0.3, size=(n_per_class, latent_dim))
        latents_2[:, 0] = rng.normal(0, 0.3, size=n_per_class)
        latents_2[:, 1] = rng.normal(5, 0.3, size=n_per_class)
        latents = np.vstack([latents_0, latents_1, latents_2])
        labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
        shuffle_idx = rng.permutation(n_samples)
        latents = latents[shuffle_idx]
        labels = labels[shuffle_idx]
        token_timestamps = np.linspace(0, n_samples * token_bin_ms / 1000, n_samples)
        behavioral_data = {"trial_target_id": labels}
        return _make_intermediates(
            latents, behavioral_data, token_timestamps, token_bin_size_ms=token_bin_ms
        )

    def test_regression_scalar_unchanged(self):
        """Scalar chunk_window_ms behaves identically to before."""
        intermediates = self._make_regression_data(token_bin_ms=50.0)
        metric = BehavioralRegressionMetric(
            target_field="cursor_pos",
            cv=3,
            train_drop_invalid=False,
            eval_drop_invalid=False,
            chunk_window_ms=50.0,
        )
        result = metric.compute(intermediates)
        assert result.raw_value > 0.5
        assert "chunk_window_sweep" not in result.metadata

    def test_regression_single_element_list_unchanged(self):
        """List with single value behaves same as scalar."""
        intermediates = self._make_regression_data(token_bin_ms=50.0)
        metric = BehavioralRegressionMetric(
            target_field="cursor_pos",
            cv=3,
            train_drop_invalid=False,
            eval_drop_invalid=False,
            chunk_window_ms=[50.0],
        )
        result = metric.compute(intermediates)
        assert result.raw_value > 0.5
        assert "chunk_window_sweep" not in result.metadata

    @pytest.mark.parametrize(
        "metric_class", [BehavioralRegressionMetric, BehavioralClassificationMetric]
    )
    def test_sweep_reports_smallest_eligible_chunk(self, metric_class):
        """Report by chunk size, even when a larger chunk scores better."""
        metric = metric_class(chunk_window_ms=[1000.0, 200.0])

        def compute(_):
            chunk = metric.chunk_window_ms
            return metric._make_result(
                raw_value={200.0: 0.4, 1000.0: 0.9}[chunk],
                metadata={"chunk_window_ms": chunk, "n_train_chunks": int(10000 / chunk)},
            )

        metric._compute_single_chunk_size = compute
        result = metric.compute({})
        assert result.raw_value == 0.4
        assert result.metadata["primary_chunk_window_ms"] == 200.0
        assert result.metadata["n_train_chunks"] == 50
        assert result.metadata["chunk_window_sweep"][1000.0]["raw_value"] == 0.9
        assert result.metadata["best_chunk_window_ms"] == 1000.0
        assert result.metadata["best_raw_value"] == 0.9
        assert metric.chunk_window_ms == 200.0

    @pytest.mark.parametrize(
        "metric_class", [BehavioralRegressionMetric, BehavioralClassificationMetric]
    )
    @pytest.mark.parametrize("failure", ["error", "nan"])
    def test_sweep_falls_back_to_1000(self, metric_class, failure):
        """A failed or non-finite 200 ms score leaves the valid 1000 ms result."""
        metric = metric_class(chunk_window_ms=[200.0, 1000.0])
        valid_score = -0.1 if metric_class is BehavioralRegressionMetric else 0.4

        def compute(_):
            chunk = metric.chunk_window_ms
            if chunk == 200.0 and failure == "error":
                raise ValueError("Chunk smaller than token bin")
            return metric._make_result(
                raw_value=float("nan") if chunk == 200.0 else valid_score,
                metadata={"chunk_window_ms": chunk},
            )

        metric._compute_single_chunk_size = compute
        result = metric.compute({})
        assert result.raw_value == valid_score  # A finite negative R² is still valid.
        assert result.metadata["primary_chunk_window_ms"] == 1000.0
        assert result.metadata["chunk_window_sweep"][200.0]["status"] == "error"
        assert metric.chunk_window_ms == 1000.0

    def test_regression_sweep_all_fail_raises(self):
        """No valid result is reported when both configured chunks fail."""
        intermediates = self._make_regression_data(token_bin_ms=50.0)
        metric = BehavioralRegressionMetric(
            target_field="cursor_pos",
            cv=3,
            train_drop_invalid=False,
            eval_drop_invalid=False,
            chunk_window_ms=[70.0, 130.0],
        )
        with pytest.raises(ValueError, match="No valid eligible chunk_window_ms"):
            metric.compute(intermediates)

    def test_classification_sweep_reports_smallest_chunk(self):
        """Real classification data also uses the smallest eligible chunk."""
        intermediates = self._make_classification_data(token_bin_ms=50.0)
        metric = BehavioralClassificationMetric(
            target_field="trial_target_id",
            grouping_field=None,
            cv=3,
            drop_invalid=False,
            chunk_window_ms=[200.0, 400.0],
        )
        result = metric.compute(intermediates)

        assert "chunk_window_sweep" in result.metadata
        sweep = result.metadata["chunk_window_sweep"]
        assert result.raw_value == sweep[200.0]["raw_value"]
        assert result.metadata["primary_chunk_window_ms"] == 200.0

    def test_classification_sweep_tolerates_failing_sizes(self):
        """Classification sweep handles failing chunk sizes gracefully."""
        intermediates = self._make_classification_data(token_bin_ms=50.0)
        metric = BehavioralClassificationMetric(
            target_field="trial_target_id",
            grouping_field=None,
            cv=3,
            drop_invalid=False,
            chunk_window_ms=[70.0, 200.0],
        )
        result = metric.compute(intermediates)

        sweep = result.metadata["chunk_window_sweep"]
        assert sweep[70.0]["status"] == "error"
        assert sweep[200.0]["status"] == "ok"

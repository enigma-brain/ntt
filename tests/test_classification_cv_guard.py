"""Tests for the cls_cv < 2 guard in BehavioralClassificationMetric._classify_1d_target.

`cls_cv = min(self.cv, len(y_train))` can drop to 1 on tiny folds. When
`cls_cv == 1`, RidgeClassifierCV raises a ValueError because it requires at least
2 folds, so folds where `cls_cv < 2` must be skipped (as the regression path does)
before creating the model.
"""

import numpy as np
import pytest

from ntt.evaluation.metrics.downstream_prediction import BehavioralClassificationMetric


def _make_intermediates(
    n_per_class: int = 10,
    n_classes: int = 2,
    latent_dim: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Build minimal intermediates for BehavioralClassificationMetric.

    Returns (intermediates, X, y) where X and y are the raw arrays (before
    wrapping) so tests can inspect or further manipulate them.

    Each class is repeated for a contiguous block of `n_per_class` samples.
    The `trial_id` field produces one group per class block.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_samples = n_per_class * n_classes
    X = rng.normal(size=(n_samples, latent_dim))
    # Alternate class labels in contiguous blocks
    y = np.repeat(np.arange(n_classes), n_per_class)
    # Each contiguous block is its own trial
    trial_id = np.repeat(np.arange(n_classes), n_per_class)
    token_timestamps = np.arange(n_samples, dtype=float)

    intermediates = {
        "latents": X,
        "behavioral_data_token_res": {
            "label": y,
            "trial_id": trial_id,
        },
        "token_timestamps": token_timestamps,
        "token_bin_size_ms": 20.0,
    }
    return intermediates, X, y


# ---------------------------------------------------------------------------
# The failing test: cls_cv < 2 is not guarded in the classification path
# ---------------------------------------------------------------------------


def test_classification_cv_guard_skips_tiny_fold(monkeypatch):
    """_classify_1d_target must skip folds where cls_cv < 2, not crash.

    Strategy: monkeypatch StratifiedGroupKFold.split to inject a fold whose
    training set has only 1 sample. With cv=5, cls_cv = min(5, 1) = 1.
    Without the guard, RidgeClassifierCV(cv=1) raises ValueError.
    With the guard (mirroring the regression path), the fold is silently
    skipped and the remaining valid fold(s) are used.
    """
    intermediates, X, y = _make_intermediates(n_per_class=100, n_classes=2)

    from sklearn.model_selection import StratifiedGroupKFold

    import ntt.evaluation.metrics.downstream_prediction as dp_module

    call_count = [0]

    def patched_split(self, X_inner, y_inner=None, groups=None):
        """Return two folds: first has 1 training sample, second is normal."""
        call_count[0] += 1
        n = len(X_inner)
        # Fold 0: train on just the very first sample (cls_cv will be 1)
        tiny_train = np.array([0])
        tiny_val = np.arange(1, n)
        # Fold 1: train on everything except the last sample (normal size)
        normal_train = np.arange(0, n - 1)
        normal_val = np.array([n - 1])
        return iter([(tiny_train, tiny_val), (normal_train, normal_val)])

    monkeypatch.setattr(StratifiedGroupKFold, "split", patched_split)

    metric = BehavioralClassificationMetric(
        target_field="label",
        grouping_field="trial_id",
        cv=5,  # cls_cv = min(5, 1) = 1 for the tiny fold → must be skipped
        n_splits=2,
        drop_invalid=False,
        chunk_window_ms=100,  # small enough for our 30-sample test data
    )

    # Without the guard: raises ValueError from RidgeClassifierCV(cv=1).
    # With the guard:   tiny fold is skipped, normal fold succeeds → result returned.
    result = metric.compute(intermediates)

    assert result is not None, "Expected a MetricResult, got None"
    assert 0.0 <= result.raw_value <= 1.0, f"Accuracy out of [0, 1]: {result.raw_value}"
    # Confirm our monkeypatch actually triggered (sanity check)
    assert call_count[0] >= 1, "monkeypatched split was never called"


def test_classification_cv_guard_all_folds_tiny(monkeypatch):
    """When every fold would hit cls_cv < 2, result should raise ValueError (no valid folds).

    This tests that the guard doesn't silently swallow *all* folds and produce a
    meaningless result — after skipping all tiny folds, the existing
    'no valid folds' error should fire.
    """
    intermediates, X, y = _make_intermediates(n_per_class=100, n_classes=2)

    from sklearn.model_selection import StratifiedGroupKFold

    def all_tiny_split(self, X_inner, y_inner=None, groups=None):
        """Return two folds both with only 1 training sample."""
        n = len(X_inner)
        fold_a = (np.array([0]), np.arange(1, n))
        fold_b = (np.array([1]), np.arange(0, 1))
        return iter([fold_a, fold_b])

    monkeypatch.setattr(StratifiedGroupKFold, "split", all_tiny_split)

    metric = BehavioralClassificationMetric(
        target_field="label",
        grouping_field="trial_id",
        cv=5,
        n_splits=2,
        drop_invalid=False,
        chunk_window_ms=100,
    )

    # With the guard, both folds are skipped → ValueError("no valid folds")
    with pytest.raises(ValueError, match="no valid folds"):
        metric.compute(intermediates)

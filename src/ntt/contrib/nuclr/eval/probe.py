"""Logistic regression probes on frozen embeddings.

Evaluates NuCLR neuron identity representations by classifying brain region
from per-neuron embeddings. Matches the original NuCLR evaluation protocol.

Three evaluation methods:
- ``l1out``: Leave-one-session-out cross-validation (primary metric).
- ``kfold``: Stratified K-fold cross-validation.
- ``zeroshot``: Train on one set, test on another.

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _session_to_group(session_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """Map session ids into ``n_groups`` buckets with a stable digest.

    Uses ``hashlib`` rather than Python's builtin ``hash`` (salted per process
    via ``PYTHONHASHSEED``) so the grouping — and the probe metrics computed
    from it — are reproducible across runs and processes.
    """
    return np.array(
        [
            int.from_bytes(hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest(), "big")
            % n_groups
            for s in session_ids
        ]
    )


def _fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    class_weight: str | None = "balanced",
) -> np.ndarray:
    """Fit logistic regression and return predictions on test set."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        solver="newton-cg",
        class_weight=class_weight,
    )
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute classification metrics."""
    return {
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
    }


def _run_l1out(
    embeddings: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray,
    class_weight: str | None,
    hash_sessions: bool = True,
) -> dict[str, float]:
    """Leave-one-session-out cross-validation.

    For IBL (many sessions), hashes session_ids into 5 pseudo-sessions.
    For Allen (fewer sessions), uses actual session IDs.
    """
    if hash_sessions:
        n_groups = 5
        groups = _session_to_group(session_ids, n_groups)
    else:
        groups = session_ids

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 session groups for L1Out")

    all_preds = []
    all_targets = []

    for test_group in unique_groups:
        train_mask = groups != test_group
        test_mask = groups == test_group

        preds = _fit_and_predict(
            embeddings[train_mask],
            labels[train_mask],
            embeddings[test_mask],
            class_weight,
        )
        all_preds.append(preds)
        all_targets.append(labels[test_mask])

    return _score(np.concatenate(all_targets), np.concatenate(all_preds))


def _grouped_splits(
    embeddings: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray | None,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build ``(train_idx, test_idx)`` folds for K-fold CV.

    When ``session_ids`` carries >= 2 distinct sessions, whole sessions are kept
    on one side of each split (``StratifiedGroupKFold``) so no session appears
    in both train and test. Otherwise (a single session) falls back to plain
    stratified K-fold and warns that the estimate is within-session, and so
    optimistic.
    """
    if session_ids is not None:
        n_sessions = len(np.unique(session_ids))
        if n_sessions >= 2:
            n_splits = min(n_folds, n_sessions)
            sgkf = StratifiedGroupKFold(n_splits=n_splits)
            return list(sgkf.split(embeddings, labels, groups=session_ids))

    logger.warning(
        "K-fold probe over a single session group: train and test share a "
        "session, so the estimate is within-session and optimistic."
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return list(skf.split(embeddings, labels))


def _run_kfold(
    embeddings: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray | None,
    class_weight: str | None,
    n_folds: int,
) -> dict[str, float]:
    """K-fold cross-validation, session-aware when possible (no leakage)."""
    all_preds = []
    all_targets = []

    for train_idx, test_idx in _grouped_splits(embeddings, labels, session_ids, n_folds):
        preds = _fit_and_predict(
            embeddings[train_idx],
            labels[train_idx],
            embeddings[test_idx],
            class_weight,
        )
        all_preds.append(preds)
        all_targets.append(labels[test_idx])

    return _score(np.concatenate(all_targets), np.concatenate(all_preds))


def run_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray | None = None,
    method: str = "l1out",
    class_weight: str | None = "balanced",
    n_folds: int = 5,
    hash_sessions: bool = True,
) -> dict[str, float]:
    """Run logistic regression probe on frozen embeddings.

    Args:
        embeddings: Shape ``(N, dim)`` float array of neuron embeddings.
        labels: Shape ``(N,)`` string array of brain region labels.
        session_ids: Shape ``(N,)`` array of session identifiers.
            Required for ``l1out`` method.
        method: Evaluation method: ``"l1out"`` | ``"kfold"``.
        class_weight: Passed to LogisticRegression. ``"balanced"`` adjusts
            for class imbalance.
        n_folds: Number of folds for ``kfold`` method.
        hash_sessions: If True, hash session IDs into 5 groups for L1Out
            (used for IBL with many sessions).

    Returns:
        Dict with keys: ``bacc``, ``f1_macro``, ``f1_micro``.
    """
    # Filter out NaN / invalid embeddings
    valid = np.isfinite(embeddings).all(axis=1)
    if not valid.all():
        n_invalid = (~valid).sum()
        logger.warning("Dropping %d units with non-finite embeddings", n_invalid)
        embeddings = embeddings[valid]
        labels = labels[valid]
        if session_ids is not None:
            session_ids = session_ids[valid]

    if len(embeddings) == 0:
        raise ValueError("No valid embeddings to evaluate")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes, got {len(unique_labels)}")

    logger.info(
        "Running %s probe: %d units, %d classes (%s)",
        method,
        len(embeddings),
        len(unique_labels),
        ", ".join(sorted(unique_labels)),
    )

    if method == "l1out":
        if session_ids is None:
            raise ValueError("session_ids required for l1out method")
        # Check if we have enough session groups before attempting L1Out
        if hash_sessions:
            groups = _session_to_group(session_ids, 5)
        else:
            groups = session_ids
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            logger.warning(
                "Only %d session group(s) for L1Out, falling back to kfold",
                n_groups,
            )
            return _run_kfold(embeddings, labels, session_ids, class_weight, n_folds)
        return _run_l1out(
            embeddings,
            labels,
            session_ids,
            class_weight,
            hash_sessions,
        )
    elif method == "kfold":
        return _run_kfold(embeddings, labels, session_ids, class_weight, n_folds)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'l1out' or 'kfold'.")

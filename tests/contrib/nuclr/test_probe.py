"""CPU tests for the NuCLR brain-region probe (eval/probe.py).

Covers reproducible session grouping (no per-process-salted hashing) and
session-aware K-fold cross-validation (no train/test session leakage).
"""

import hashlib
import logging

import numpy as np

from ntt.contrib.nuclr.eval.probe import _grouped_splits, _session_to_group, run_probe


def test_session_grouping_is_deterministic() -> None:
    """Grouping must be reproducible, i.e. not use Python's salted builtin hash.

    Expected values are recomputed here with the same stable digest the
    implementation uses; a builtin-``hash`` implementation would produce
    process-dependent buckets that would not match.
    """
    sessions = np.array(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    n_groups = 5

    groups = _session_to_group(sessions, n_groups)

    expected = [
        int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
        % n_groups
        for s in sessions
    ]
    assert groups.tolist() == expected
    # All buckets in range, and the mapping is idempotent.
    assert set(np.unique(groups).tolist()).issubset(set(range(n_groups)))
    assert _session_to_group(sessions, n_groups).tolist() == groups.tolist()


def test_kfold_splits_have_no_session_leakage() -> None:
    """With >=2 sessions, each fold keeps whole sessions on one side."""
    rng = np.random.default_rng(0)
    n_sessions, per = 4, 10
    sessions = np.repeat([f"s{i}" for i in range(n_sessions)], per)
    labels = np.tile(np.repeat(["A", "B"], per // 2), n_sessions)  # 5 A / 5 B per session
    embeddings = rng.standard_normal((n_sessions * per, 8))

    splits = _grouped_splits(embeddings, labels, sessions, n_folds=4)

    assert len(splits) >= 2
    for train_idx, test_idx in splits:
        train_sessions = set(sessions[train_idx].tolist())
        test_sessions = set(sessions[test_idx].tolist())
        assert train_sessions.isdisjoint(test_sessions), "session leaked across a fold"


def test_kfold_single_session_falls_back_and_warns(caplog) -> None:
    """A genuine single session cannot be split group-wise: fall back to plain
    K-fold and warn that the estimate is within-session (optimistic)."""
    rng = np.random.default_rng(1)
    sessions = np.array(["only"] * 20)
    labels = np.tile(["A", "B"], 10)
    embeddings = rng.standard_normal((20, 8))

    with caplog.at_level(logging.WARNING):
        splits = _grouped_splits(embeddings, labels, sessions, n_folds=5)

    assert len(splits) == 5
    assert "within-session" in caplog.text.lower() or "optimistic" in caplog.text.lower()


def test_run_probe_returns_metrics_for_both_methods() -> None:
    """run_probe threads session_ids through to both the kfold and l1out paths."""
    rng = np.random.default_rng(0)
    sessions = np.repeat([f"s{i}" for i in range(3)], 12)
    labels = np.tile(np.repeat(["A", "B"], 6), 3)
    emb = rng.standard_normal((36, 8))

    kfold = run_probe(emb, labels, sessions, method="kfold")
    l1out = run_probe(emb, labels, sessions, method="l1out", hash_sessions=False)
    for res in (kfold, l1out):
        assert set(res) == {"bacc", "f1_macro", "f1_micro"}
        assert all(0.0 <= v <= 1.0 for v in res.values())

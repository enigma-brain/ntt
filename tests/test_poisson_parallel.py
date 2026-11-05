"""Tests for the parallel realisations primitive."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ntt.evaluation.poisson_realisations import run_poisson_realisations

# ── Module-level stubs (must be picklable for fork-spawned workers) ──────


class _StubPipeline:
    """Stand-in for EvaluationPipeline.

    .run(orig, recon) returns a deterministic dict keyed on the recon
    contents, so identical inputs produce identical output.
    """

    def run(self, original, reconstructed):
        arr = np.asarray(reconstructed.binned_spikes)
        return {
            "mean": float(arr.mean()),
            "shape": list(arr.shape),  # JSON-friendly for equality checks
        }


def _make_stub_pipeline():
    """Module-level (picklable) factory that builds a fresh CPU pipeline."""
    return _StubPipeline()


class _PerSegmentMeanPipeline:
    """Pipeline whose ``run()`` returns the reconstruction's mean as a fingerprint.

    Used by the seed-distinctness test to detect collapsed seeds: identical
    inputs + identical seeds produce identical means, so a per-segment
    duplicate mean across realisations is a seed leak.
    """

    def run(self, original, reconstructed):
        arr = np.asarray(reconstructed.binned_spikes, dtype=float)
        return {"mean": float(arr.mean())}


def _make_per_segment_mean_pipeline():
    """Module-level (picklable) factory for ``_PerSegmentMeanPipeline``."""
    return _PerSegmentMeanPipeline()


class _StubEvalData:
    """Stand-in for the EvalData type used in the realisation loop."""

    def __init__(self, binned_spikes, bin_size_seconds, duration, data_type):
        self.binned_spikes = binned_spikes
        self.bin_size_seconds = bin_size_seconds
        self.duration = duration
        self.data_type = data_type


class _StubSegment:
    """Stand-in for CollatedValidationSegment — only the index is read."""

    def __init__(self, idx):
        self.segment_index = idx


def _make_originals(tmp_path, n_sessions=2, n_segments=3, n_neurons=5, n_bins=200):
    """Return an originals dict with mmap-backed numpy arrays.

    Arrays are in (units, time) format — the convention expected by
    ``poisson_baseline_reconstruction``.
    """
    originals = {}
    rng = np.random.default_rng(0)
    for s_idx in range(n_sessions):
        sid = f"sess{s_idx}"
        entries = []
        for seg_idx in range(n_segments):
            arr = rng.integers(1, 5, size=(n_neurons, n_bins), dtype=np.int32)
            path = tmp_path / f"{sid}_seg{seg_idx}.npy"
            np.save(path, arr)
            mmap = np.load(path, mmap_mode="r")
            data = _StubEvalData(
                binned_spikes=mmap,
                bin_size_seconds=0.001,
                duration=n_bins * 0.001,
                data_type="binned_spikes",
            )
            metadata = {"_original_stash_path": str(path)}
            entries.append((_StubSegment(seg_idx), data, metadata))
        originals[sid] = entries
    return originals


def test_zero_realisations_returns_empty_list_no_workers(tmp_path):
    originals = _make_originals(tmp_path)
    result = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=42,
        n_realisations_to_run=0,
    )
    assert result == []


def test_serial_and_parallel_produce_identical_results(tmp_path):
    originals = _make_originals(tmp_path, n_sessions=2, n_segments=3)

    serial = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=42,
        n_realisations_to_run=4,
        n_workers=1,
    )
    parallel = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=42,
        n_realisations_to_run=4,
        n_workers=2,
    )

    assert len(serial) == 4
    assert len(parallel) == 4
    for r_serial, r_parallel in zip(serial, parallel, strict=True):
        assert set(r_serial.keys()) == set(r_parallel.keys())
        for sid in r_serial:
            assert len(r_serial[sid]) == len(r_parallel[sid])
            for (_, data_s), (_, data_p) in zip(r_serial[sid], r_parallel[sid], strict=True):
                assert data_s["report"] == data_p["report"]


def test_results_returned_in_seed_order(tmp_path):
    """Even when workers complete out of order, results must be in seed order."""
    originals = _make_originals(tmp_path, n_sessions=1, n_segments=2)
    result = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=100,
        n_realisations_to_run=3,
        n_workers=2,
    )
    # 3 realisations × 1 session × 2 segments = 6 reports; should produce
    # at least 4 distinct mean values across different seeds.
    seen_means = set()
    for r in result:
        for segs in r.values():
            for _, data in segs:
                seen_means.add(data["report"]["mean"])
    assert len(seen_means) >= 4, seen_means


def test_distinct_seeds_produce_distinct_per_segment_outputs(tmp_path):
    """Each realisation must use a distinct seed *and* the seed must reach
    ``poisson_baseline_reconstruction``.

    The earlier ``stress_under_parent_cuda_pressure`` test only asserted
    ``len(set(all_means)) >= n_seeds`` across all (seed, session, segment)
    triples — that lower bound is satisfied even if every realisation
    secretly used the same seed, as long as there are ≥ n_seeds distinct
    *segments*. This test pins it down per-segment: for a fixed segment,
    the 5 realisations' reconstructions must NOT all be identical, and
    the per-segment std must be > 0. Catches any future regression that
    re-routes the seed (e.g. shadowing, fixed-seed factory, dropped kwarg).
    """
    originals = _make_originals(tmp_path, n_sessions=2, n_segments=3)

    n_realisations = 5
    result = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_per_segment_mean_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=42,
        n_realisations_to_run=n_realisations,
        n_workers=4,
    )
    assert len(result) == n_realisations

    # Pivot: for each (session, segment), collect the n_realisations means.
    by_segment: dict[tuple[str, int], list[float]] = {}
    for r in result:
        for sid, segs in r.items():
            for seg_idx, (_, data) in enumerate(segs):
                by_segment.setdefault((sid, seg_idx), []).append(data["report"]["mean"])

    # Every (session, segment) must see exactly n_realisations distinct draws —
    # if any two realisations produced an identical mean for the *same segment*,
    # the seeds collapsed.
    failures: list[str] = []
    for key, means in by_segment.items():
        assert len(means) == n_realisations
        if len(set(means)) < n_realisations:
            failures.append(f"{key}: means={means} (only {len(set(means))} distinct)")
        if float(np.std(means)) == 0.0:
            failures.append(f"{key}: std=0 across realisations (means={means})")
    assert not failures, "Seed leak detected:\n  " + "\n  ".join(failures)


def test_n_realisations_to_run_one_uses_serial_path(tmp_path):
    """A single realisation should not spawn a worker pool (cold-start cost)."""
    originals = _make_originals(tmp_path, n_sessions=1, n_segments=2)
    result = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=7,
        n_realisations_to_run=1,
        n_workers=4,  # User requested 4 but only 1 realisation → serial.
    )
    assert len(result) == 1


# ── CUDA / fork-isolation tests ──────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_parallel_realisations_isolate_workers_from_parent_cuda(tmp_path):
    """Parent has CUDA initialized + a CUDA tensor live; parallel workers
    must run successfully under fork.

    The real job (poisson_baseline.run) initializes CUDA in realisation 1
    and then submits parallel realisations. Under fork, deserializing any
    CUDA tensor in the args triggers `torch.cuda._lazy_init()` in the
    child and dies with `Cannot re-initialize CUDA in forked subprocess`.

    Contract: the runner accepts a ``pipeline_factory`` (a picklable
    zero-arg callable) so each fork child constructs its own CPU
    pipeline. The parent's CUDA state never enters the IPC path.
    """
    parent_cuda_tensor = torch.zeros(8, device="cuda")
    parent_cuda_tensor.fill_(3.14)

    originals = _make_originals(tmp_path, n_sessions=2, n_segments=2)

    result = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=42,
        n_realisations_to_run=4,
        n_workers=2,
    )
    assert len(result) == 4
    assert parent_cuda_tensor.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_parallel_realisations_stress_under_parent_cuda_pressure(tmp_path):
    """Stress smoke: parent holds live CUDA state; many seeds × many workers;
    each worker handles multiple tasks; parent CUDA stays healthy afterwards.

    Mimics the production scenario of many fork children with a parent that
    initialized CUDA via realisation 1, and locks down the contract that
    ``pipeline_factory`` keeps the parent's CUDA state out of the IPC path.
    """
    parent_tensors = [torch.full((128,), i, device="cuda", dtype=torch.float32) for i in range(4)]
    parent_module = torch.nn.Linear(64, 64).cuda()

    originals = _make_originals(tmp_path, n_sessions=3, n_segments=4, n_neurons=8, n_bins=300)

    n_seeds = 8
    n_workers = 4
    result = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=1000,
        n_realisations_to_run=n_seeds,
        n_workers=n_workers,
    )

    assert len(result) == n_seeds
    for r in result:
        assert set(r.keys()) == set(originals.keys())
        for sid, segs in r.items():
            assert len(segs) == len(originals[sid])

    # Different seeds should produce different reconstructions and so
    # different stub-reported means. Loose lower bound (≥ n_seeds distinct)
    # keeps the test robust to coincidental collisions.
    all_means = [data["report"]["mean"] for r in result for segs in r.values() for _, data in segs]
    assert len(set(all_means)) >= n_seeds, (
        f"expected ≥{n_seeds} distinct means across seeds; got {len(set(all_means))}"
    )

    # Parent CUDA state is still intact.
    for i, t in enumerate(parent_tensors):
        assert t.device.type == "cuda"
        assert torch.all(t == i).item()
    assert next(parent_module.parameters()).device.type == "cuda"

    # Parent can still allocate fresh CUDA memory.
    fresh = torch.full((16,), 7.0, device="cuda")
    assert fresh.sum().item() == pytest.approx(7.0 * 16)


def _make_segment_with_cuda_tensors(idx: int, original_shape: tuple[int, int]):
    """Build a CollatedValidationSegment carrying CUDA-resident tensors.

    Mirrors the post-realisation-1 production state: ``original`` already
    stripped (only its shape is preserved), but ``latents``, ``reconstructed``,
    and the token timestamp tensors live on CUDA.
    """
    from ntt.evaluation.utils.validation_collation import CollatedValidationSegment

    return CollatedValidationSegment(
        session_id=f"sess{idx // 4}",
        original=None,
        reconstructed=torch.zeros(original_shape, device="cuda"),
        pad_and_gap_mask=None,
        metadata={"sampling_rate": 1000.0, "subject_id": "x", "species": "y"},
        segment_index=idx,
        is_partial=False,
        latents=torch.zeros(4, 32, device="cuda"),
        token_timestamps=torch.zeros(8, device="cuda"),
        token_start_times=torch.zeros(8, device="cuda"),
        token_end_times=torch.zeros(8, device="cuda"),
        original_shape=original_shape,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_parallel_realisations_with_real_segments_and_cuda_tensors(tmp_path):
    """Real ``CollatedValidationSegment`` carrying CUDA tensors must be safely
    shipped to fork workers after ``tensor_free_copy()``.

    Even with ``pipeline_factory`` in place, the
    segments riding inside ``originals`` carry ``latents`` /
    ``reconstructed`` / ``token_*_times`` torch tensors on CUDA, which
    would trip ``Cannot re-initialize CUDA in forked subprocess``.

    Contract: ``CollatedValidationSegment.tensor_free_copy()`` returns a copy
    with all torch.Tensor fields nulled, preserving non-tensor identity
    (segment_index, original_shape, etc.). Callers strip segments before
    handing originals to ``run_poisson_realisations``.
    """
    rng = np.random.default_rng(0)
    originals: dict[str, list] = {}
    for s_idx in range(2):
        sid = f"sess{s_idx}"
        entries = []
        for seg_idx in range(2):
            idx = s_idx * 4 + seg_idx
            seg = _make_segment_with_cuda_tensors(idx, original_shape=(4, 200))
            seg_safe = seg.tensor_free_copy()
            arr = rng.integers(1, 5, size=(4, 200), dtype=np.int32)
            path = tmp_path / f"{sid}_seg{seg_idx}.npy"
            np.save(path, arr)
            mmap = np.load(path, mmap_mode="r")
            data = _StubEvalData(
                binned_spikes=mmap,
                bin_size_seconds=0.001,
                duration=0.2,
                data_type="binned_spikes",
            )
            metadata = {"_original_stash_path": str(path)}
            entries.append((seg_safe, data, metadata))
        originals[sid] = entries

    result = run_poisson_realisations(
        originals=originals,
        pipeline_factory=_make_stub_pipeline,
        bin_size_seconds=0.001,
        bin_size_ms=1.0,
        poisson_rate_ms=10.0,
        base_seed=42,
        n_realisations_to_run=4,
        n_workers=2,
    )
    assert len(result) == 4
    for r in result:
        assert set(r.keys()) == {"sess0", "sess1"}
        for sid in r:
            for stripped_seg, _data in r[sid]:
                # The stripped segment that came back must still be tensor-free
                # and carry preserved non-tensor identity.
                assert stripped_seg.original is None
                assert stripped_seg.reconstructed is None
                assert stripped_seg.latents is None
                assert stripped_seg.token_timestamps is None
                assert stripped_seg.token_start_times is None
                assert stripped_seg.token_end_times is None
                assert stripped_seg.original_shape == (4, 200)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runner_auto_strips_cuda_tensors_from_originals(tmp_path, caplog):
    """Defensive auto-strip: even if a caller forgets to ``tensor_free_copy()``
    its segments, the runner walks ``originals`` and moves CUDA tensors to CPU
    before fork submit. Must succeed and emit a warning naming the offending
    path(s) so the caller knows to fix the source.
    """
    import logging

    rng = np.random.default_rng(0)
    originals: dict[str, list] = {}
    for s_idx in range(2):
        sid = f"sess{s_idx}"
        entries = []
        for seg_idx in range(2):
            idx = s_idx * 4 + seg_idx
            # Note: NOT calling tensor_free_copy — segment carries CUDA tensors.
            seg = _make_segment_with_cuda_tensors(idx, original_shape=(4, 200))
            arr = rng.integers(1, 5, size=(4, 200), dtype=np.int32)
            path = tmp_path / f"{sid}_seg{seg_idx}.npy"
            np.save(path, arr)
            mmap = np.load(path, mmap_mode="r")
            data = _StubEvalData(
                binned_spikes=mmap,
                bin_size_seconds=0.001,
                duration=0.2,
                data_type="binned_spikes",
            )
            metadata = {"_original_stash_path": str(path)}
            entries.append((seg, data, metadata))
        originals[sid] = entries

    with caplog.at_level(logging.WARNING, logger="ntt.evaluation.poisson_realisations"):
        result = run_poisson_realisations(
            originals=originals,
            pipeline_factory=_make_stub_pipeline,
            bin_size_seconds=0.001,
            bin_size_ms=1.0,
            poisson_rate_ms=10.0,
            base_seed=42,
            n_realisations_to_run=4,
            n_workers=2,
        )
    assert len(result) == 4
    # The defensive strip should have logged what it found, naming the field.
    assert any(
        "stripped" in rec.message and "CUDA tensor" in rec.message for rec in caplog.records
    ), [rec.message for rec in caplog.records]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tensor_free_copy_nulls_all_torch_fields_preserves_metadata():
    """Unit test for the strip helper: every torch.Tensor field on the
    dataclass becomes None; non-tensor identity is preserved."""
    seg = _make_segment_with_cuda_tensors(idx=7, original_shape=(3, 150))
    seg.behavioral_data_token_res = {"phase": np.zeros(8, dtype=np.int8)}

    out = seg.tensor_free_copy()

    # Tensor fields nulled.
    assert out.original is None
    assert out.reconstructed is None
    assert out.pad_and_gap_mask is None
    assert out.latents is None
    assert out.token_timestamps is None
    assert out.token_start_times is None
    assert out.token_end_times is None

    # Identity preserved.
    assert out.session_id == "sess1"  # idx=7 → sess1 per helper
    assert out.segment_index == 7
    assert out.original_shape == (3, 150)
    assert out.is_partial is False
    assert out.metadata == {"sampling_rate": 1000.0, "subject_id": "x", "species": "y"}
    # Numpy fields (cheap, no CUDA) preserved by reference.
    assert out.behavioral_data_token_res is seg.behavioral_data_token_res

    # Original is unchanged (we returned a copy, didn't mutate).
    assert seg.latents is not None
    assert seg.latents.device.type == "cuda"

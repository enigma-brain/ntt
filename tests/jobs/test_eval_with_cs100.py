from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ntt.contrib.tokenizer.compressed_sensing import (
    CompressedSensingTokenizer,
    CompressedSensingTokenizerSpec,
)
from ntt.core.components.types import BinnedSpikesInput
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.utils.validation_collation import iter_collated_validation_segments
from ntt.jobs.eval import NTTEvalJob


def _make_eval_batch(
    *, time_bins: int = 1000, units: int = 2, sampling_rate: float = 1000.0
) -> BinnedSpikesInput:
    spike_matrix = torch.randint(0, 2, (1, time_bins, units), dtype=torch.float32)
    pad_and_gap_mask = torch.ones_like(spike_matrix, dtype=torch.int32)
    duration = torch.tensor([time_bins / sampling_rate], dtype=torch.float64)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_and_gap_mask,
        sampling_rate=torch.tensor([sampling_rate]),
        unit_ids=torch.arange(units, dtype=torch.int64).unsqueeze(0),
        unit_ids_pad_mask=torch.ones((1, units), dtype=torch.int64),
        duration_sec=duration,
        session_id=("sess0",),
        subject_id=("subject0",),
        species=("species0",),
        absolute_start=torch.tensor([0.0], dtype=torch.float64),
        absolute_end=duration.clone(),
        behavioral_data=None,
        behavioral_intervals=None,
    )


class _StopEval(Exception):
    pass


def test_main_eval_handles_cs100_via_chunked_tokenizer_fn():
    """1 s main eval keeps the full reconstruction length for a CS-100ms tokenizer."""
    spec = CompressedSensingTokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1.0,
        n_neurons=2,
        latent_dim=4,
        fista_iterations=2,
        random_seed=0,
    )
    tokenizer = CompressedSensingTokenizer(spec)
    batch = _make_eval_batch()
    dataloader = [batch]

    job = NTTEvalJob.model_construct(
        split="val",
        compare_poisson=False,
        poisson_reference=None,
        evaluate_all_neurons=False,
        data_config=None,
        eval_config=None,
        tokenizer_config=spec,
    )
    job._tokenizer = tokenizer
    job._spec = spec
    job._device = torch.device("cpu")
    job._eval_config = EvaluationConfig()
    job._datamodule = SimpleNamespace(
        window_length=1.0,
        val_dataloader=lambda: dataloader,
        test_dataloader=lambda: dataloader,
    )

    def _fake_eval_runner(
        passed_loader,
        tokenizer_fun,
        *,
        token_bin_size_ms,
        window_length_seconds,
        tokenizer_spec,
        **kwargs,
    ):
        assert passed_loader is dataloader
        assert window_length_seconds == pytest.approx(1.0)
        assert tokenizer_spec == spec

        segments = list(
            iter_collated_validation_segments(
                passed_loader,
                tokenizer_fun,
                token_bin_size_ms=token_bin_size_ms,
                window_length_seconds=window_length_seconds,
                tokenizer_spec=tokenizer_spec,
                device=kwargs.get("device"),
                max_windows=1,
            )
        )

        assert len(segments) == 1
        assert segments[0].reconstructed is not None
        assert segments[0].reconstructed.shape == (2, 1000)
        raise _StopEval

    # The fake datamodule has no units table to record a neuron selection from.
    with (
        patch(
            "ntt.jobs.eval.run_validation_evaluation_per_session", side_effect=_fake_eval_runner
        ),
        patch("ntt.jobs.eval.describe_neuron_selection", return_value=None),
    ):
        with pytest.raises(_StopEval):
            job.run()


def test_main_eval_skips_runtime_partial_window_shorter_than_chunk():
    """A session-boundary batch shorter than one tokenizer chunk is skipped, not fatal."""
    spec = CompressedSensingTokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1.0,
        n_neurons=2,
        latent_dim=4,
        fista_iterations=2,
        random_seed=0,
    )
    tokenizer = CompressedSensingTokenizer(spec)
    # 50 bins @ 1 ms = 0.05 s — SHORTER than the 0.1 s (100-bin) chunk → n_chunks==0.
    short_batch = _make_eval_batch(time_bins=50, units=2)

    @torch.no_grad()
    def _tok(b):
        return tokenizer.reconstruct_from_input(b, return_latents=True)

    segments = list(
        iter_collated_validation_segments(
            [short_batch],
            _tok,
            token_bin_size_ms=spec.token_bin_size_ms,
            window_length_seconds=0.1,
            tokenizer_spec=spec,
        )
    )
    assert segments == [], "expected the sub-chunk partial window to be skipped, not crash"


def test_calm_main_eval_produces_nonempty_reconstruction_at_1s():
    """calm produces a non-empty reconstruction over a genuine 1 s eval window.

    This mirrors the production standard-eval path (``eval.py:_raw_tokenize``),
    which calls ``tokenizer.reconstruct_from_input(batch, return_latents=True)``
    and feeds it through ``iter_collated_validation_segments``. calm has
    ``bin_size_ms=10.0`` (→ 100 Hz), so a real 1 s window is 100 bins at
    ``sampling_rate=100.0`` with ``duration_sec=1.0`` — the fixture is built
    fully consistent with that (no force-chunking, no latents-only dodge).
    """
    from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizer, CalmTokenizerSpec

    spec = CalmTokenizerSpec(window_length_seconds=1.0, bin_size_ms=10.0, n_neurons=2)
    tokenizer = CalmTokenizer(spec)
    # 100 bins @ 10 ms (sampling_rate=100 Hz) = a genuine 1 s window.
    batch = _make_eval_batch(sampling_rate=100.0, time_bins=100, units=2)

    @torch.no_grad()
    def _tok(b):
        # Production tokenizer_fn shape (see eval.py:_raw_tokenize).
        return tokenizer.reconstruct_from_input(b, return_latents=True)

    segments = list(
        iter_collated_validation_segments(
            [batch],
            _tok,
            token_bin_size_ms=spec.token_bin_size_ms,
            window_length_seconds=1.0,
            tokenizer_spec=spec,
        )
    )

    assert len(segments) == 1, f"expected 1 non-empty segment, got {len(segments)}"
    reconstructed = segments[0].reconstructed
    assert reconstructed is not None, "production reconstruct path must yield a reconstruction"
    # 100 bins @ 10 ms over 2 units → (units, time_bins).
    assert reconstructed.shape == (
        2,
        100,
    ), f"expected a (2, 100) reconstruction for a 1 s window, got {tuple(reconstructed.shape)}"


def test_fixed_tokenizer_incompatible_window_fails_loud():
    from ntt.contrib.tokenizer.fsqffn.fsqffn import FSQFFNTokenizerSpec
    from ntt.evaluation.utils.window_contract import (
        IncompatibleEvalWindowError,
        assert_tokenizer_accepts_eval_window,
    )

    spec = FSQFFNTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0, n_neurons=2)
    with pytest.raises(IncompatibleEvalWindowError, match="fsqffn"):
        assert_tokenizer_accepts_eval_window(spec, 1.5)


def test_apply_eval_window_sets_window_and_validates():
    """apply_eval_window is the single source of truth: it stamps the eval window
    onto data_config (calm passes) and fails loud for an incompatible fixed window."""
    from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec
    from ntt.contrib.tokenizer.fsqffn.fsqffn import FSQFFNTokenizerSpec
    from ntt.evaluation.utils.window_contract import (
        IncompatibleEvalWindowError,
        apply_eval_window,
    )

    calm_spec = CalmTokenizerSpec(window_length_seconds=2.0, bin_size_ms=10.0, n_neurons=2)
    data_config = SimpleNamespace(window_length_seconds=2.0)
    apply_eval_window(data_config, calm_spec, 1.0)
    assert data_config.window_length_seconds == 1.0

    fsq_spec = FSQFFNTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0, n_neurons=2)
    bad_config = SimpleNamespace(window_length_seconds=1.0)
    with pytest.raises(IncompatibleEvalWindowError, match="fsqffn"):
        apply_eval_window(bad_config, fsq_spec, 1.5)


def test_plot_segment_chunks_a_rigid_tokenizer_shorter_than_the_eval_window():
    """The padded datamodule loads the eval window (1 s), the CS tokenizer takes 0.1 s."""
    from ntt.evaluation import plotting

    spec = CompressedSensingTokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1.0,
        n_neurons=2,
        latent_dim=4,
        fista_iterations=2,
        random_seed=0,
    )
    tokenizer = CompressedSensingTokenizer(spec)
    datamodule = SimpleNamespace(
        val_dataset=object(), config=SimpleNamespace(window_length_seconds=1.0)
    )
    captured = {}
    real_iter = plotting.iter_collated_validation_segments

    def _spy(loader, fn, **kwargs):
        captured["max_windows"] = kwargs["max_windows"]
        return real_iter(loader, fn, **kwargs)

    with (
        patch.object(plotting, "_build_filtered_dataloader", return_value=[_make_eval_batch()]),
        patch.object(plotting, "iter_collated_validation_segments", _spy),
    ):
        segment, _ = plotting.get_validation_segment(
            tokenizer, datamodule, seconds=3.0, device="cpu"
        )

    assert captured["max_windows"] == 3  # 3 s of 1 s windows, not 30 of 0.1 s
    assert segment.reconstructed is not None
    assert segment.reconstructed.shape == (2, 1000)

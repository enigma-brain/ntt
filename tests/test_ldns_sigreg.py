from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import ValidationError
from torch import nn

from ntt.contrib.loss.sigreg import SIGReg
from ntt.contrib.tokenizer.ldns import LDNSTokenizer, LDNSTokenizerSpec
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.model.tokenizer.validation import make_dummy_input


def _sum_aux(aux_loss):
    """Sum dict aux_loss values to a scalar for test comparison."""
    if isinstance(aux_loss, dict):
        return sum(aux_loss.values())
    return aux_loss


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def _load_tokenizer_config(config_name: str) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[
                f"tokenizer@job.tokenizer_config=contrib/{config_name}",
                "loss@job.loss_config=contrib/poisson_nll",
            ],
        )
    tokenizer_cfg = cfg.job.tokenizer_config
    return cast(dict[str, Any], OmegaConf.to_container(tokenizer_cfg, resolve=True))


def _make_ldns_spec(**overrides: Any) -> LDNSTokenizerSpec:
    base = {
        "n_neurons": 8,
        "window_length_seconds": 0.32,
        "bin_size_ms": 10.0,
        "hidden_dim": 16,
        "latent_dim": 4,
        "num_encoder_blocks": 1,
        "num_decoder_blocks": 0,
        "num_lin_per_mlp": 2,
        "bidirectional": False,
        "latent_beta": 0.1,
        "latent_td_beta": 0.2,
        "latent_td_k": 2,
        "mask_prob": 0.0,
        "neuron_dropout_prob": 0.0,
        "validate_from_data": False,
    }
    base.update(overrides)
    return LDNSTokenizerSpec(**base)


def _manual_latent_loss(z: torch.Tensor, spec: LDNSTokenizerSpec) -> torch.Tensor:
    numel = z.numel()
    total = spec.latent_beta * (z**2).sum() / numel
    td = torch.zeros((), device=z.device)
    for i in range(1, spec.latent_td_k + 1):
        if z.shape[-1] > i:
            td = td + ((z[:, :, :-i] - z[:, :, i:]) ** 2 * (1 / (1 + i))).sum() / numel
    total = total + spec.latent_td_beta * td
    return total


class _RecordingSigReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_proj: torch.Tensor | None = None

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        self.last_proj = proj.detach().clone()
        return proj.sum()


def test_ldns_sigreg_config_round_trip() -> None:
    base_spec = LDNSTokenizerSpec.model_validate(_load_tokenizer_config("ldns"))
    sigreg_spec = LDNSTokenizerSpec.model_validate(_load_tokenizer_config("ldns_sigreg"))
    alignment_spec = LDNSTokenizerSpec.model_validate(
        _load_tokenizer_config("ldns_view_alignment")
    )

    assert base_spec.use_sigreg is False
    assert base_spec.views == 0
    assert sigreg_spec.use_sigreg is True
    assert sigreg_spec.views == 0
    assert sigreg_spec.sigreg_beta == pytest.approx(1e-2)
    assert sigreg_spec.sigreg_knots == 17
    assert sigreg_spec.sigreg_sketch_dim == 256
    assert sigreg_spec.sigreg_seed == 42
    assert alignment_spec.use_sigreg is True
    assert alignment_spec.sigreg_beta == pytest.approx(1e-2)
    assert alignment_spec.sigreg_seed == 42
    assert alignment_spec.views == 1
    assert alignment_spec.view_alignment_mask_prob == pytest.approx(0.20)
    assert alignment_spec.view_alignment_neuron_dropout_prob == pytest.approx(0.0)


def test_sigreg_disabled_matches_existing_latent_loss() -> None:
    torch.manual_seed(0)
    spec = _make_ldns_spec(use_sigreg=False)
    tokenizer = cast(Any, LDNSTokenizer(spec))
    batch = make_dummy_input(spec)
    assert isinstance(batch, BinnedSpikesInput)

    tokenizer.train()
    output = tokenizer(batch)
    spike_matrix = batch.spike_matrix.float().transpose(-1, -2)
    z = tokenizer.count_wrapper.encode(spike_matrix)
    expected = _manual_latent_loss(z, spec)

    assert output.aux_loss is not None
    assert tokenizer._latest_sigreg_loss is None
    assert tokenizer._latest_sigreg_full_loss is None
    assert torch.allclose(sum(output.aux_loss.values()), expected)


def test_sigreg_adds_aux_loss_when_enabled() -> None:
    torch.manual_seed(0)
    base_spec = _make_ldns_spec(use_sigreg=True, sigreg_beta=0.0, views=0, sigreg_seed=123)
    tokenizer_no_weight = cast(Any, LDNSTokenizer(base_spec))

    torch.manual_seed(0)
    weighted_spec = _make_ldns_spec(use_sigreg=True, sigreg_beta=0.5, views=0, sigreg_seed=123)
    tokenizer_with_weight = cast(Any, LDNSTokenizer(weighted_spec))
    batch = make_dummy_input(base_spec)
    assert isinstance(batch, BinnedSpikesInput)

    tokenizer_no_weight.train()
    tokenizer_with_weight.train()
    assert tokenizer_no_weight.sigreg is not None
    no_weight_sigreg_state = tokenizer_no_weight.sigreg._rng.get_state()
    no_weight_loss = _sum_aux(tokenizer_no_weight(batch).aux_loss)
    tokenizer_no_weight.sigreg._rng.set_state(no_weight_sigreg_state)
    spike_matrix = batch.spike_matrix.float().transpose(-1, -2)
    z = tokenizer_with_weight.count_wrapper.encode(spike_matrix)
    assert tokenizer_with_weight.sigreg is not None
    sigreg_state = tokenizer_with_weight.sigreg._rng.get_state()
    expected_sigreg = tokenizer_with_weight._compute_sigreg_loss(z)
    tokenizer_with_weight.sigreg._rng.set_state(sigreg_state)
    weighted_loss = _sum_aux(tokenizer_with_weight(batch).aux_loss)

    assert no_weight_loss is not None
    assert weighted_loss is not None
    assert torch.isfinite(weighted_loss)
    expected = _manual_latent_loss(z, weighted_spec) + weighted_spec.sigreg_beta * expected_sigreg
    assert torch.allclose(weighted_loss, expected)
    assert tokenizer_with_weight._latest_sigreg_loss is not None
    assert tokenizer_with_weight._latest_sigreg_full_loss is not None
    assert torch.allclose(tokenizer_with_weight._latest_sigreg_loss, expected_sigreg)
    assert torch.allclose(
        tokenizer_with_weight._latest_sigreg_full_loss,
        expected_sigreg * (z.shape[0] * z.shape[-1]),
    )
    assert weighted_loss.item() > no_weight_loss.item()


def test_sigreg_uses_batch_and_sequence_positions_as_samples() -> None:
    spec = _make_ldns_spec(use_sigreg=True)
    tokenizer = cast(Any, LDNSTokenizer(spec))
    recorder = _RecordingSigReg()
    tokenizer.sigreg = recorder
    z = torch.arange(2 * spec.latent_dim * 3, dtype=torch.float32).reshape(2, spec.latent_dim, 3)

    loss = tokenizer._compute_sigreg_loss(z)

    expected = z.permute(0, 2, 1).reshape(-1, spec.latent_dim)
    assert recorder.last_proj is not None
    assert recorder.last_proj.shape == (z.shape[0] * z.shape[-1], z.shape[1])
    assert torch.equal(recorder.last_proj, expected)
    assert torch.equal(loss, expected.sum())


def test_sigreg_seed_makes_random_features_reproducible() -> None:
    proj = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    sigreg_a = SIGReg(knots=5, t_max=2.0, sketch_dim=8, seed=7)
    sigreg_b = SIGReg(knots=5, t_max=2.0, sketch_dim=8, seed=7)

    torch.manual_seed(999)
    loss_a = sigreg_a(proj)
    torch.manual_seed(123)
    loss_b = sigreg_b(proj)

    assert torch.allclose(loss_a, loss_b)


def test_sigreg_is_skipped_in_eval_mode() -> None:
    torch.manual_seed(0)
    spec = _make_ldns_spec(use_sigreg=True, sigreg_beta=0.5, views=0)
    tokenizer = cast(Any, LDNSTokenizer(spec))
    batch = make_dummy_input(spec)
    assert isinstance(batch, BinnedSpikesInput)

    tokenizer.eval()
    output = tokenizer(batch)
    spike_matrix = batch.spike_matrix.float().transpose(-1, -2)
    z = tokenizer.count_wrapper.encode(spike_matrix)
    expected = _manual_latent_loss(z, spec)

    assert output.aux_loss is not None
    assert torch.allclose(sum(output.aux_loss.values()), expected)


def test_sigreg_is_skipped_in_stage_two_adapter_mode() -> None:
    torch.manual_seed(0)
    base_spec = _make_ldns_spec(
        enable_spike_adapter=True,
        freeze_autoencoder=True,
        spike_history_length=4,
        use_sigreg=False,
    )
    tokenizer_control = LDNSTokenizer(base_spec)

    torch.manual_seed(0)
    sigreg_spec = _make_ldns_spec(
        enable_spike_adapter=True,
        freeze_autoencoder=True,
        spike_history_length=4,
        use_sigreg=True,
        sigreg_beta=1.0,
        views=0,
    )
    tokenizer_sigreg = LDNSTokenizer(sigreg_spec)
    batch = make_dummy_input(base_spec)

    tokenizer_control.train()
    tokenizer_sigreg.train()
    control_loss = _sum_aux(tokenizer_control(batch).aux_loss)
    sigreg_loss = _sum_aux(tokenizer_sigreg(batch).aux_loss)

    assert control_loss is not None
    assert sigreg_loss is not None
    assert torch.allclose(control_loss, sigreg_loss)
    assert tokenizer_sigreg._latest_sigreg_loss is None
    assert tokenizer_sigreg._latest_sigreg_full_loss is None


def test_view_alignment_requires_non_zero_corruption() -> None:
    with pytest.raises(ValidationError, match="requires at least one non-zero corruption"):
        _make_ldns_spec(
            use_sigreg=True,
            views=1,
            mask_prob=0.0,
            neuron_dropout_prob=0.0,
            view_alignment_mask_prob=0.0,
            view_alignment_neuron_dropout_prob=0.0,
        )


def test_view_alignment_adds_aux_loss_when_enabled() -> None:
    torch.manual_seed(0)
    base_spec = _make_ldns_spec(
        use_sigreg=True,
        views=2,
        sigreg_beta=1.0,
        sigreg_seed=123,
        mask_prob=0.0,
        neuron_dropout_prob=0.0,
        view_alignment_mask_prob=0.25,
    )
    tokenizer_no_weight = cast(Any, LDNSTokenizer(base_spec))

    torch.manual_seed(0)
    weighted_spec = _make_ldns_spec(
        use_sigreg=True,
        views=2,
        sigreg_beta=0.5,
        sigreg_seed=123,
        mask_prob=0.0,
        neuron_dropout_prob=0.0,
        view_alignment_mask_prob=0.25,
    )
    tokenizer_with_weight = cast(Any, LDNSTokenizer(weighted_spec))
    batch = make_dummy_input(base_spec)
    assert isinstance(batch, BinnedSpikesInput)
    batch = batch._replace(
        spike_matrix=(
            torch.arange(1, batch.spike_matrix.numel() + 1, dtype=torch.int32).reshape_as(
                batch.spike_matrix
            )
            % 5
        )
    )

    tokenizer_no_weight.train()
    tokenizer_with_weight.train()
    assert tokenizer_no_weight.sigreg is not None
    no_weight_sigreg_state = tokenizer_no_weight.sigreg._rng.get_state()
    no_weight_loss = _sum_aux(tokenizer_no_weight(batch).aux_loss)
    tokenizer_no_weight.sigreg._rng.set_state(no_weight_sigreg_state)
    spike_matrix = batch.spike_matrix.float().transpose(-1, -2)
    z_no_weight = tokenizer_no_weight.count_wrapper.encode(spike_matrix)
    z = tokenizer_with_weight.count_wrapper.encode(spike_matrix)
    assert tokenizer_with_weight.sigreg is not None
    expected_no_weight_sigreg = tokenizer_no_weight._compute_sigreg_loss(z_no_weight)
    tokenizer_no_weight.sigreg._rng.set_state(no_weight_sigreg_state)
    weighted_sigreg_state = tokenizer_with_weight.sigreg._rng.get_state()
    expected_sigreg = tokenizer_with_weight._compute_sigreg_loss(z)
    view_alignment_rng_state = torch.random.get_rng_state()
    expected_alignment = tokenizer_with_weight._compute_view_alignment_aux_loss(
        spike_matrix,
        z,
        views=weighted_spec.views,
    )
    tokenizer_with_weight.sigreg._rng.set_state(weighted_sigreg_state)
    torch.random.set_rng_state(view_alignment_rng_state)
    weighted_loss = _sum_aux(tokenizer_with_weight(batch).aux_loss)

    assert no_weight_loss is not None
    assert weighted_loss is not None
    assert torch.isfinite(weighted_loss)
    assert expected_alignment.item() > 0
    expected_no_weight = (
        _manual_latent_loss(z_no_weight, base_spec)
        + base_spec.sigreg_beta * expected_no_weight_sigreg
    )
    expected = (
        _manual_latent_loss(z, weighted_spec)
        + weighted_spec.sigreg_beta * expected_sigreg
        + (1.0 - weighted_spec.sigreg_beta) * expected_alignment
    )
    assert torch.allclose(no_weight_loss, expected_no_weight)
    assert torch.allclose(weighted_loss, expected)


def test_view_alignment_uses_independent_corruption_with_same_shape() -> None:
    torch.manual_seed(0)
    spec = _make_ldns_spec(
        views=1,
        mask_prob=0.5,
        neuron_dropout_prob=0.0,
        view_alignment_mask_prob=0.5,
    )
    tokenizer = cast(Any, LDNSTokenizer(spec))
    spike_matrix = torch.arange(1, 1 + 2 * spec.n_neurons * 32, dtype=torch.float32).reshape(
        2, spec.n_neurons, 32
    )

    primary_view, _neuron_mask = tokenizer._apply_input_corruptions(
        spike_matrix,
        mask_prob=spec.mask_prob,
        neuron_dropout_prob=spec.neuron_dropout_prob,
    )
    second_view = tokenizer._make_view_alignment_input(spike_matrix)

    assert second_view.shape == spike_matrix.shape
    assert not torch.equal(second_view, spike_matrix)
    assert not torch.equal(second_view, primary_view)


def test_view_alignment_is_skipped_when_sigreg_disabled() -> None:
    torch.manual_seed(0)
    spec = _make_ldns_spec(
        use_sigreg=False,
        views=1,
        sigreg_beta=0.5,
        mask_prob=0.0,
        neuron_dropout_prob=0.0,
        view_alignment_mask_prob=0.25,
    )
    tokenizer = cast(Any, LDNSTokenizer(spec))
    batch = make_dummy_input(spec)
    assert isinstance(batch, BinnedSpikesInput)

    tokenizer.train()
    output = tokenizer(batch)
    spike_matrix = batch.spike_matrix.float().transpose(-1, -2)
    z = tokenizer.count_wrapper.encode(spike_matrix)
    expected = _manual_latent_loss(z, spec)

    assert output.aux_loss is not None
    assert torch.allclose(sum(output.aux_loss.values()), expected)


def test_view_alignment_is_skipped_in_eval_mode() -> None:
    torch.manual_seed(0)
    spec = _make_ldns_spec(
        use_sigreg=True,
        views=1,
        sigreg_beta=0.5,
        mask_prob=0.0,
        neuron_dropout_prob=0.0,
        view_alignment_mask_prob=0.25,
    )
    tokenizer = cast(Any, LDNSTokenizer(spec))
    batch = make_dummy_input(spec)
    assert isinstance(batch, BinnedSpikesInput)

    tokenizer.eval()
    output = tokenizer(batch)
    spike_matrix = batch.spike_matrix.float().transpose(-1, -2)
    z = tokenizer.count_wrapper.encode(spike_matrix)
    expected = _manual_latent_loss(z, spec)

    assert output.aux_loss is not None
    assert torch.allclose(sum(output.aux_loss.values()), expected)


def test_view_alignment_is_skipped_in_stage_two_adapter_mode() -> None:
    torch.manual_seed(0)
    base_spec = _make_ldns_spec(
        enable_spike_adapter=True,
        freeze_autoencoder=True,
        spike_history_length=4,
        views=0,
    )
    tokenizer_control = LDNSTokenizer(base_spec)

    torch.manual_seed(0)
    alignment_spec = _make_ldns_spec(
        enable_spike_adapter=True,
        freeze_autoencoder=True,
        spike_history_length=4,
        use_sigreg=True,
        views=1,
        sigreg_beta=0.0,
        mask_prob=0.0,
        neuron_dropout_prob=0.0,
        view_alignment_mask_prob=0.25,
    )
    tokenizer_alignment = LDNSTokenizer(alignment_spec)
    batch = make_dummy_input(base_spec)

    tokenizer_control.train()
    tokenizer_alignment.train()
    control_loss = _sum_aux(tokenizer_control(batch).aux_loss)
    alignment_loss = _sum_aux(tokenizer_alignment(batch).aux_loss)

    assert control_loss is not None
    assert alignment_loss is not None
    assert torch.allclose(control_loss, alignment_loss)


def test_neuron_dropout_does_not_mutate_pad_and_gap_mask() -> None:
    """forward() must never modify batch.pad_and_gap_mask in-place."""
    torch.manual_seed(0)
    spec = _make_ldns_spec(neuron_dropout_prob=0.5)
    tokenizer = LDNSTokenizer(spec)
    tokenizer.train()

    batch = make_dummy_input(spec)
    assert isinstance(batch, BinnedSpikesInput)
    assert batch.pad_and_gap_mask is not None

    original_mask = batch.pad_and_gap_mask.clone()
    tokenizer(batch)

    assert torch.equal(batch.pad_and_gap_mask, original_mask), (
        "forward() must not mutate batch.pad_and_gap_mask in-place"
    )


def test_get_effective_pad_mask_with_neuron_dropout() -> None:
    """With neuron_dropout_prob > 0 in training, get_effective_pad_mask()
    returns a mask that is a subset of the original."""
    torch.manual_seed(42)
    spec = _make_ldns_spec(neuron_dropout_prob=0.5)
    tokenizer = LDNSTokenizer(spec)
    tokenizer.train()

    batch = make_dummy_input(spec)
    assert isinstance(batch, BinnedSpikesInput)
    assert batch.pad_and_gap_mask is not None

    tokenizer(batch)
    effective = tokenizer.get_effective_pad_mask()

    assert effective is not None, "expected non-None effective mask with neuron dropout"
    # Effective mask must be a subset: everywhere effective is True, original is True.
    original = batch.pad_and_gap_mask.bool()
    assert (effective.bool() & ~original).sum() == 0, (
        "effective mask has True where original is False"
    )
    # With 50% dropout, at least some neurons should have been dropped.
    assert effective.sum() < original.sum(), (
        "expected effective mask to have fewer True values than original"
    )


def test_get_effective_pad_mask_none_without_dropout() -> None:
    """With neuron_dropout_prob == 0, get_effective_pad_mask() returns None."""
    spec = _make_ldns_spec(neuron_dropout_prob=0.0)
    tokenizer = LDNSTokenizer(spec)
    tokenizer.train()

    batch = make_dummy_input(spec)
    tokenizer(batch)

    assert tokenizer.get_effective_pad_mask() is None


def test_get_effective_pad_mask_none_in_eval() -> None:
    """In eval mode, get_effective_pad_mask() returns None."""
    spec = _make_ldns_spec(neuron_dropout_prob=0.5)
    tokenizer = LDNSTokenizer(spec)
    tokenizer.eval()

    batch = make_dummy_input(spec)
    tokenizer(batch)

    assert tokenizer.get_effective_pad_mask() is None


def test_base_tokenizer_get_effective_pad_mask_returns_none() -> None:
    """BaseTokenizer.get_effective_pad_mask() returns None by default."""
    # LDNSTokenizer inherits from BaseTokenizer; before forward() is
    # called, the default should be None.
    spec = _make_ldns_spec()
    tokenizer = LDNSTokenizer(spec)
    # Call the base class method directly to confirm the contract.
    from ntt.core.model.tokenizer.base import BaseTokenizer

    assert BaseTokenizer.get_effective_pad_mask(tokenizer) is None


def test_view_alignment_requires_sigreg_beta_in_unit_interval() -> None:
    with pytest.raises(ValidationError, match="sigreg_beta` to be in \\[0, 1\\]"):
        _make_ldns_spec(
            use_sigreg=True,
            views=1,
            sigreg_beta=1.5,
            view_alignment_mask_prob=0.25,
        )

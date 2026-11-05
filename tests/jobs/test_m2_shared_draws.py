"""latent_robustness (M2) uses shared random draws for tokenizers that sample spikes.

M2 compares a clean reconstruction with reconstructions of noisy latents. When
the decoder samples (calm with ``reconstruct_sample`` bernoulli/poisson, the
Poisson baseline), independent draws on each side make the score measure
sampling noise: at zero latent noise the ACG similarity of calm was 0.62, not 1.
M2 therefore decodes expected values and samples the clean and every noisy
reconstruction of a window from one uniform field keyed on (seed, session,
window start): spike iff u < p (Bernoulli), inverse-CDF counts (Poisson).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ntt.core.components.types import DataType, ReconstructionOutput
from ntt.evaluation.metrics.robustness_latent_stats import LatentStats
from ntt.jobs.eval_robustness_latent import NTTEvalRobustnessLatentJob
from tests.jobs.test_eval_robustness_latent import (
    _FakeExistingEvalDM,
    _make_run_job,
    _make_window_segment,
    _StubPipeline,
    _zero_noise,
)

Job = NTTEvalRobustnessLatentJob


# ── Sampling primitives ──────────────────────────────────────────────────────


def test_shared_uniforms_are_keyed_on_window_identity() -> None:
    def uniforms(seed=0, session_id="sessA", window_start=1.5):
        return Job._m2_shared_uniforms(
            (3, 50), seed=seed, session_id=session_id, window_start=window_start
        )

    base = uniforms()

    assert torch.equal(base, uniforms())  # reproducible
    assert bool(((base >= 0) & (base < 1)).all())
    assert not torch.equal(base, uniforms(seed=1))
    assert not torch.equal(base, uniforms(session_id="sessB"))
    assert not torch.equal(base, uniforms(window_start=2.5))


def test_bernoulli_shared_draw_spikes_where_uniform_is_below_probability() -> None:
    p = torch.tensor([[0.0, 0.3, 0.3, 0.9, 1.0]])
    u = torch.tensor([[0.0, 0.2, 0.4, 0.95, 0.999]])

    sample = Job._sample_with_shared_uniforms(p, u, kind="bernoulli")

    assert sample.tolist() == [[0.0, 1.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("rate", [0.05, 0.5, 3.0])
def test_poisson_shared_draw_has_poisson_moments(rate: float) -> None:
    u = torch.rand(400_000, generator=torch.Generator().manual_seed(0)).reshape(1, -1)

    counts = Job._sample_with_shared_uniforms(torch.full_like(u, rate), u, kind="poisson")

    assert torch.equal(counts, counts.round())
    assert float(counts.mean()) == pytest.approx(rate, rel=0.03)
    assert float(counts.var()) == pytest.approx(rate, rel=0.03)


@pytest.mark.parametrize("rate", [0.01, 0.5, 2.0, 5.0, 30.0, 200.0, 500.0])
def test_poisson_shared_draw_is_the_inverse_cdf(rate: float) -> None:
    from scipy.stats import poisson

    u = torch.linspace(0.0, 0.999999, 5001, dtype=torch.float64).reshape(1, -1)
    rates = torch.full_like(u, rate)
    rates[0, ::7] = 0.0  # mixed rates in one tensor

    counts = Job._sample_with_shared_uniforms(rates, u, kind="poisson")

    # ppf(u) is the smallest k with CDF(k) >= u; the sampler uses CDF(k) > u.
    expected = np.where(
        rates.numpy() > 0, poisson.ppf(np.nextafter(u.numpy(), 1.0), rates.numpy()), 0.0
    )
    np.testing.assert_array_equal(counts.numpy(), expected)


@pytest.mark.parametrize("kind", ["bernoulli", "poisson"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_shared_draw_rejects_non_finite_expected_values(kind: str, bad: float) -> None:
    """A diverged decoder must fail loud, not read as a silent window."""
    expected = torch.tensor([[0.3, bad, 0.2]])
    u = torch.full((1, 3), 0.99, dtype=torch.float64)

    with pytest.raises(ValueError, match="non-finite"):
        Job._sample_with_shared_uniforms(expected, u, kind=kind)  # type: ignore[arg-type]


@pytest.mark.parametrize("rate", [500.5, 800.0, 5_000.0, 1e6])
def test_poisson_shared_draw_approximates_the_inverse_cdf_at_large_rates(rate: float) -> None:
    """Past exp(-rate) underflow the quantile is approximated, never off by more than 1."""
    from scipy.stats import poisson

    # From float64 torch.rand's resolution; u = 0 maps to that resolution too.
    u = torch.linspace(2**-53, 1.0 - 1e-12, 20001, dtype=torch.float64).reshape(1, -1)

    counts = Job._sample_with_shared_uniforms(torch.full_like(u, rate), u, kind="poisson")

    expected = poisson.ppf(np.nextafter(u.numpy(), 1.0), rate)
    diff = counts.numpy() - expected
    assert np.abs(diff).max() <= 1
    assert np.mean(diff == 0) > 0.99


def test_poisson_shared_draw_mixes_exact_and_approximate_rates() -> None:
    u = torch.rand(1, 50_000, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    rates = torch.full_like(u, 2.0)
    rates[0, ::2] = 900.0

    counts = Job._sample_with_shared_uniforms(rates, u, kind="poisson")

    assert float(counts[0, ::2].mean()) == pytest.approx(900.0, rel=0.005)
    assert float(counts[0, ::2].var()) == pytest.approx(900.0, rel=0.05)
    assert float(counts[0, 1::2].mean()) == pytest.approx(2.0, rel=0.03)


def test_poisson_shared_draw_is_monotone_at_large_rates() -> None:
    u = torch.rand(1, 10_000, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    u[0, :2] = torch.tensor([0.0, 1.0 - 2**-53])  # extreme uniforms stay finite

    low = Job._sample_with_shared_uniforms(torch.full_like(u, 600.0), u, kind="poisson")
    high = Job._sample_with_shared_uniforms(torch.full_like(u, 610.0), u, kind="poisson")

    assert bool(torch.isfinite(low).all())
    assert bool((high >= low).all())
    assert bool((high > low).any())


def test_cap_cdf_table_reproduces_the_exact_walk_at_the_cap() -> None:
    """The floor above the cap is the walk's own count at the cap, bit for bit."""
    from ntt.jobs.eval_robustness_latent import (
        _M2_EXACT_POISSON_MAX_RATE,
        _poisson_cdf_at_exact_cap,
    )

    u = torch.rand(1, 10_000, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    u[0, :2] = torch.tensor([0.0, 1.0 - 2**-53])
    cdf = torch.tensor(_poisson_cdf_at_exact_cap(), dtype=torch.float64)

    walk = Job._sample_with_shared_uniforms(
        torch.full_like(u, _M2_EXACT_POISSON_MAX_RATE), u, kind="poisson"
    )
    lookup = torch.searchsorted(cdf, u, right=True).clamp(max=len(cdf) - 1)

    assert torch.equal(walk, lookup.to(torch.float32))


@pytest.mark.parametrize(("low_rate", "high_rate"), [(500.0, 500.0001), (499.9, 500.5)])
def test_poisson_shared_draw_is_monotone_across_the_exact_cap(
    low_rate: float, high_rate: float
) -> None:
    """Exact below the cap, approximate above: the count must not drop at the switch."""
    u = torch.rand(1, 20_000, generator=torch.Generator().manual_seed(0), dtype=torch.float64)

    low = Job._sample_with_shared_uniforms(torch.full_like(u, low_rate), u, kind="poisson")
    high = Job._sample_with_shared_uniforms(torch.full_like(u, high_rate), u, kind="poisson")

    assert bool((high >= low).all())


def test_poisson_shared_draw_is_monotone_in_rate() -> None:
    """Common random numbers: a higher rate never yields fewer spikes for the same u."""
    u = torch.rand(1, 10_000, generator=torch.Generator().manual_seed(1))

    low = Job._sample_with_shared_uniforms(torch.full_like(u, 0.4), u, kind="poisson")
    high = Job._sample_with_shared_uniforms(torch.full_like(u, 0.6), u, kind="poisson")

    assert bool((high >= low).all())
    assert bool((high > low).any())


# ── End to end: zero latent noise reproduces the clean reconstruction ────────


class _CalmLikeBernoulliTokenizer:
    """Decodes latents to spike probabilities; ``reconstruct`` Bernoulli-samples them."""

    def __init__(self, spec):
        self.spec = spec

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.clamp(0.0, 1.0)

    def reconstruct(self, latents: torch.Tensor) -> ReconstructionOutput:
        return ReconstructionOutput(
            reconstruction=torch.bernoulli(self.decode(latents)),
            data_type=DataType.SPIKES,
            bin_size_seconds=0.001,
        )


class PoissonBaselineTokenizer:  # M2 recognises the Poisson baseline by class name
    def __init__(self, spec):
        self.spec = spec

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.clamp(min=0.0)

    def reconstruct(self, latents: torch.Tensor) -> ReconstructionOutput:
        return ReconstructionOutput(
            reconstruction=torch.poisson(self.decode(latents)),
            data_type=DataType.SPIKES,
            bin_size_seconds=0.001,
        )


def _run_zero_noise(tmp_path: Path, monkeypatch, tokenizer_cls, **spec_fields) -> list[dict]:
    import json

    job = _make_run_job(tmp_path)
    for name, value in spec_fields.items():
        setattr(job._spec, name, value)
    job._spec.output_type = DataType.SPIKES
    job._tokenizer = tokenizer_cls(job._spec)
    # Clean reconstructions as the collector stores them for sampling tokenizers:
    # expected values (probabilities / expected counts), sampled later in M2.
    expected = [[0.5, 0.2, 0.9], [0.1, 0.7, 0.4]]
    segments = [
        _make_window_segment("sess0", 0, start_seconds=0.0, window_values_tu=expected),
        _make_window_segment("sess0", 1, start_seconds=1.0, window_values_tu=expected),
    ]

    monkeypatch.setattr("ntt.jobs.eval_robustness_latent.EvaluationPipeline", _StubPipeline)
    monkeypatch.setattr(Job, "_draw_noise_from_generator", staticmethod(_zero_noise))
    monkeypatch.setattr(
        job,
        "_build_eval_loader_and_tokenizer_fn",
        lambda *a, **k: ("unused", lambda batch: {"reconstruction": batch}),
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
            sigma_joint=1.0, sigma_per_dim=np.ones((3,), dtype=np.float32)
        ),
    )

    job.run()

    with (tmp_path / "results.jsonl").open() as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize(
    "tokenizer_cls,spec_fields",
    [
        (_CalmLikeBernoulliTokenizer, {"KIND": "calm", "reconstruct_sample": "bernoulli"}),
        (PoissonBaselineTokenizer, {"KIND": "poisson_baseline"}),
    ],
    ids=["calm_bernoulli", "poisson_baseline"],
)
def test_zero_latent_noise_reproduces_the_clean_sample(
    tmp_path: Path, monkeypatch, tokenizer_cls, spec_fields
) -> None:
    rows = _run_zero_noise(tmp_path, monkeypatch, tokenizer_cls, **spec_fields)

    mse = [row for row in rows if row["metric"] == "mse_match_miss"]
    assert mse
    assert all(row["raw_value"] == pytest.approx(0.0) for row in mse)
    # Reconstructions are samples, not the stored expected values (which would also
    # give mse 0): the stub metric averages integer counts over 2 windows x 6 cells,
    # so it is a multiple of 1/12, whereas the expected values average to 2.8/6.
    session_means = [
        float(row["raw_value"]) for row in rows if row["metric"] == "dummy_session_metric"
    ]
    assert session_means
    assert all(abs(v * 12 - round(v * 12)) < 1e-5 for v in session_means)  # float32


def _recorded_sampling(tmp_path: Path) -> tuple[object, object]:
    import json

    report = json.loads((tmp_path / "eval_report.json").read_text())
    run_meta = json.loads((tmp_path / "run_meta.json").read_text())
    return report["config"].get("m2_sampling", "missing"), run_meta.get("m2_sampling", "missing")


def test_reports_record_shared_draws(tmp_path: Path, monkeypatch) -> None:
    """Old (independent-draw) and new M2 results must be distinguishable."""
    _run_zero_noise(
        tmp_path,
        monkeypatch,
        _CalmLikeBernoulliTokenizer,
        KIND="calm",
        reconstruct_sample="bernoulli",
    )

    assert _recorded_sampling(tmp_path) == ("shared_uniforms:bernoulli",) * 2


def test_reports_record_deterministic_decoding(tmp_path: Path, monkeypatch) -> None:
    from tests.jobs.test_eval_robustness_latent import _FakeLatentTokenizer

    _run_zero_noise(tmp_path, monkeypatch, _FakeLatentTokenizer, KIND="wanneural")

    assert _recorded_sampling(tmp_path) == (None, None)


def test_unhandled_sampling_tokenizer_is_flagged(tmp_path: Path, monkeypatch, caplog) -> None:
    """A sampler M2 does not take over (e.g. wanx with bernoulli) is logged and recorded."""
    import logging

    from tests.jobs.test_eval_robustness_latent import _FakeLatentTokenizer

    with caplog.at_level(logging.WARNING):
        _run_zero_noise(
            tmp_path,
            monkeypatch,
            _FakeLatentTokenizer,
            KIND="wanneural",
            reconstruct_sample="bernoulli",
        )

    assert "independent draws" in caplog.text
    assert _recorded_sampling(tmp_path) == ("independent",) * 2


# ── Real tokenizers ──────────────────────────────────────────────────────────


def test_expected_values_are_what_real_calm_samples_from() -> None:
    """Under one torch seed, calm's own Bernoulli reconstruct equals sampling the
    expected-value view, so M2's shared draws sample calm's own distribution."""
    from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizer
    from ntt.jobs.eval_robustness_latent import _ExpectedValueTokenizer
    from tests.test_calm_tokenizer import _make_rates_input, _small_spec

    spec = _small_spec(
        bin_size_ms=1.0,
        reconstruct_sample="bernoulli",
        window_length_seconds=0.02,
        patch_length=4,
        output_activation="softplus",
    )
    tokenizer = CalmTokenizer(spec).eval()
    with torch.no_grad():
        latents = tokenizer.tokens_to_latents(tokenizer.encode(_make_rates_input(2, 20, 8)))
        expected = _ExpectedValueTokenizer(tokenizer, "bernoulli").reconstruct(latents)
        torch.manual_seed(0)
        sampled_by_calm = tokenizer.reconstruct(latents).reconstruction
        torch.manual_seed(0)
        sampled_from_expected = torch.bernoulli(expected.reconstruction)

    assert Job._m2_sampling_kind(tokenizer) == "bernoulli"
    assert bool(((expected.reconstruction >= 0) & (expected.reconstruction <= 1)).all())
    assert torch.equal(sampled_by_calm, sampled_from_expected)


def test_deterministic_and_poisson_baseline_tokenizers_are_classified() -> None:
    from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizer
    from ntt.contrib.tokenizer.cosmos import CosmosTokenizer2D
    from ntt.contrib.tokenizer.poisson_baseline import (
        PoissonBaselineTokenizer as RealPoissonBaseline,
    )
    from ntt.contrib.tokenizer.poisson_baseline import PoissonBaselineTokenizerSpec
    from tests.test_calm_tokenizer import _small_spec
    from tests.test_cosmos_tokenizer import _make_spec_2d

    poisson = RealPoissonBaseline(
        PoissonBaselineTokenizerSpec(
            window_length_seconds=1.0, bin_size_ms=100.0, rate_estimation_bin_ms=100.0
        )
    )
    assert Job._m2_sampling_kind(poisson) == "poisson"
    assert Job._m2_sampling_kind(CalmTokenizer(_small_spec(reconstruct_sample="none"))) is None
    assert Job._m2_sampling_kind(CosmosTokenizer2D(_make_spec_2d())) is None


# ── The clean pass yields expected values for sampling tokenizers ────────────


class _CalmLikeEncoder(_CalmLikeBernoulliTokenizer):
    def encode_with_timestamps(self, batch):
        return batch.latents, {"token_timestamps": torch.zeros(batch.latents.shape[:2])}


def test_clean_pass_returns_expected_values_for_sampling_tokenizers(tmp_path: Path) -> None:
    job = _make_run_job(tmp_path)
    job._spec.KIND = "calm"
    job._spec.reconstruct_sample = "bernoulli"
    job._spec.output_type = DataType.SPIKES
    job._tokenizer = _CalmLikeEncoder(job._spec)
    job._datamodule = _FakeExistingEvalDM(window_length_seconds=1.0, loader_label="loader")
    latents = torch.tensor([[[0.2, 0.8], [0.6, 0.4]]])
    batch = SimpleNamespace(
        latents=latents,
        session_id=["sess0"],
        absolute_start=torch.tensor([0.0], dtype=torch.float64),
    )

    _loader, tokenizer_fn = job._build_eval_loader_and_tokenizer_fn(eval_window_length_seconds=1.0)
    out = tokenizer_fn(batch)

    torch.testing.assert_close(out["reconstruction"], latents)
    assert out["reconstruction_data_type"] == DataType.SPIKES


def test_clean_pass_runs_on_real_calm_token_indices(tmp_path: Path) -> None:
    """``encode_with_timestamps`` already returns codebook vectors; converting them
    again as token indices crashed every calm M2 run (einops: 4 dims, expected 3)."""
    from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizer
    from ntt.jobs.eval_robustness_latent import _ExpectedValueTokenizer
    from tests.test_calm_tokenizer import _make_rates_input, _small_spec

    spec = _small_spec(
        bin_size_ms=1.0,
        reconstruct_sample="bernoulli",
        window_length_seconds=0.02,
        patch_length=4,
        output_activation="softplus",
    )
    tokenizer = CalmTokenizer(spec).eval()
    job = _make_run_job(tmp_path)
    job._spec = spec
    job._tokenizer = tokenizer
    job._datamodule = _FakeExistingEvalDM(window_length_seconds=0.02, loader_label="loader")
    batch = _make_rates_input(2, 20, 8)

    _loader, tokenizer_fn = job._build_eval_loader_and_tokenizer_fn(
        eval_window_length_seconds=0.02
    )
    out = tokenizer_fn(batch)

    with torch.no_grad():
        latents = tokenizer.tokens_to_latents(tokenizer.encode(batch))
        expected = _ExpectedValueTokenizer(tokenizer, "bernoulli").reconstruct(latents)
    torch.testing.assert_close(out["latents"], latents)
    torch.testing.assert_close(out["reconstruction"], expected.reconstruction)
    assert out["reconstruction_data_type"] == DataType.SPIKES
    assert out["token_timestamps"].shape == latents.shape[:2]

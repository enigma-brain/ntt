"""Tokenizers with ``input_type=RATES`` must be scored on the same scale as spike tokenizers.

The data pipeline's RATES mode carries spike counts per bin, not Hz (see
``RatesInput``). An eval path that stores those counts as ``EvalData.rates_hz``
shrinks every count by the bin size (1000x at 1 ms): mutual information rounds to
all-NaN and the absolute errors (firing rate, Fano factor, binned rate difference,
smoothed squared error) look ~1000x better than they are.
"""

import numpy as np
import pytest
import torch

from ntt.core.components.types import DataType
from ntt.evaluation import EvaluationPipeline
from ntt.evaluation.core.types import EvalData

BIN_SECONDS = 0.001


def _spike_pair(
    seed: int = 0, n_units: int = 6, n_bins: int = 6000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary (units, time) spike trains and a noisy copy, both integer-valued."""
    rng = np.random.default_rng(seed)
    original = (rng.random((n_units, n_bins)) < 0.03).astype(np.float32)
    flips = rng.random((n_units, n_bins)) < 0.01
    reconstructed = np.where(flips, 1.0 - original, original).astype(np.float32)
    return torch.from_numpy(original), torch.from_numpy(reconstructed)


def _run(
    original: torch.Tensor, reconstructed: torch.Tensor, orig_type: DataType, recon_type: DataType
):
    return EvaluationPipeline().run(
        EvalData.from_tensor(original, orig_type, bin_size_seconds=BIN_SECONDS),
        EvalData.from_tensor(reconstructed, recon_type, bin_size_seconds=BIN_SECONDS),
    )


def _value(report, name: str) -> float:
    return float(np.nanmean(np.asarray(report.results[name].raw_value, dtype=np.float64)))


def test_from_tensor_rates_stores_per_bin_counts() -> None:
    arr = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)

    data = EvalData.from_tensor(arr, data_type=DataType.RATES, bin_size_seconds=BIN_SECONDS)

    assert data.data_type == DataType.RATES
    assert data.rates_hz is None
    assert data.binned_counts is not None
    np.testing.assert_array_equal(data.binned_counts, arr.numpy())


@pytest.mark.parametrize(
    "metric",
    [
        "firing_rate_error",
        "fano_factor_error",
        "binned_rate_difference",
        "smoothed_squared_error",
        "spike_count_correlation",
        "mutual_information",
    ],
)
def test_rates_labelled_counts_score_like_spikes(metric: str) -> None:
    original, reconstructed = _spike_pair()

    spikes = _run(original, reconstructed, DataType.SPIKES, DataType.SPIKES)
    rates = _run(original, reconstructed, DataType.RATES, DataType.RATES)

    assert metric in spikes.results, spikes.skipped_metrics.get(metric)
    assert metric in rates.results, rates.skipped_metrics.get(metric)
    assert np.isfinite(_value(rates, metric))
    np.testing.assert_allclose(_value(rates, metric), _value(spikes, metric), rtol=1e-5)


def _counts_and_hz(
    seed: int = 1, n_units: int = 4, n_bins: int = 400
) -> tuple[EvalData, EvalData]:
    """The same per-bin counts as ``binned_counts`` and as Hz-valued ``rates_hz``."""
    counts = np.random.default_rng(seed).poisson(0.05, size=(n_units, n_bins)).astype(np.float32)
    as_counts = EvalData(
        binned_counts=counts, bin_size_seconds=BIN_SECONDS, data_type=DataType.RATES
    )
    as_hz = EvalData(
        rates_hz=counts / BIN_SECONDS, bin_size_seconds=BIN_SECONDS, data_type=DataType.RATES
    )
    return as_counts, as_hz


@pytest.mark.parametrize("key", ["original_windows", "original_windows_per_unit"])
def test_original_windows_are_counts_for_hz_rates(key: str) -> None:
    from ntt.evaluation.core.intermediates import IntermediatesCache
    from ntt.evaluation.metrics.base import IntermediateRequest

    as_counts, as_hz = _counts_and_hz()
    latents = np.zeros((4, 3, 2), dtype=np.float32)  # 4 windows -> 100 bins each
    request = IntermediateRequest.from_dict(key, {})

    expected = IntermediatesCache(as_counts, None, latents=latents).get(request)
    actual = IntermediatesCache(as_hz, None, latents=latents).get(request)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_reconstruction_windows_are_counts_for_hz_rates() -> None:
    from ntt.evaluation.core.intermediates import IntermediatesCache

    as_counts, as_hz = _counts_and_hz()
    latents = np.zeros((4, 3, 2), dtype=np.float32)
    cache = IntermediatesCache(as_counts, as_hz, latents=latents)

    expected = cache._compute_original_windows_per_unit_for(as_counts)
    actual = cache._compute_original_windows_per_unit_for(as_hz)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_windowed_structure_original_window_is_counts_for_hz_rates() -> None:
    from types import SimpleNamespace

    from ntt.evaluation.metrics.structure_preservation import _extract_original_window_per_unit

    as_counts, as_hz = _counts_and_hz()

    expected = _extract_original_window_per_unit(SimpleNamespace(original=as_counts))
    actual = _extract_original_window_per_unit(SimpleNamespace(original=as_hz))

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize("recon_type", [DataType.SPIKES, DataType.RATES])
def test_integer_counts_labelled_rates_match_spikes_on_every_metric(recon_type: DataType) -> None:
    """A rate-input tokenizer whose original is observed 0/1 counts at 1 ms (calm) is
    scored exactly like a spike-input tokenizer, spike-level metrics included."""
    original, reconstructed = _spike_pair(seed=3)

    spikes = _run(original, reconstructed, DataType.SPIKES, DataType.SPIKES)
    mixed = _run(original, reconstructed, DataType.RATES, recon_type)

    assert set(mixed.results) == set(spikes.results), (
        sorted(set(spikes.results) ^ set(mixed.results)),
        mixed.skipped_metrics,
    )
    for name in spikes.results:
        np.testing.assert_allclose(
            np.asarray(mixed.results[name].raw_value, dtype=np.float64),
            np.asarray(spikes.results[name].raw_value, dtype=np.float64),
            rtol=1e-5,
            atol=1e-12,
            equal_nan=True,
            err_msg=name,
        )


def test_non_integer_rates_do_not_convert_to_spike_times() -> None:
    """Expected counts (PCA-style outputs) are not spikes: spike matching stays skipped."""
    original, reconstructed = _spike_pair(seed=4)
    expected_counts = reconstructed * 0.7 + 0.01

    report = _run(original, expected_counts, DataType.SPIKES, DataType.RATES)

    assert "spike_detection_f1" not in report.results
    assert "spike_detection_f1" in report.skipped_metrics


def test_stochastic_spike_times_from_counts_match_hz_rates() -> None:
    """With stochastic conversion on, expected counts are Poisson-sampled on the count scale."""
    from ntt.evaluation.core.converters import DataConverter

    expected = np.random.default_rng(6).random((3, 500)).astype(np.float32) * 0.3
    as_counts = EvalData(binned_counts=expected, bin_size_seconds=BIN_SECONDS)
    as_hz = EvalData(rates_hz=expected / BIN_SECONDS, bin_size_seconds=BIN_SECONDS)

    def spike_times(data: EvalData) -> list[np.ndarray]:
        converter = DataConverter(allow_stochastic_conversion=True, stochastic_seed=0)
        return converter.convert(data, DataType.SPIKE_TIMES).spike_times  # type: ignore[return-value]

    for got, want in zip(spike_times(as_counts), spike_times(as_hz), strict=True):
        np.testing.assert_allclose(got, want)


def test_stochastic_spike_times_from_rates_that_also_carry_spike_times() -> None:
    """RATES data with spike_times alongside still Poisson-samples from its counts."""
    from ntt.evaluation.core.converters import DataConverter

    expected = np.random.default_rng(6).random((3, 500)).astype(np.float32) * 0.3
    only_counts = EvalData(binned_counts=expected, bin_size_seconds=BIN_SECONDS)
    with_spike_times = EvalData(
        binned_counts=expected,
        spike_times=[np.array([0.0])] * 3,
        bin_size_seconds=BIN_SECONDS,
        data_type=DataType.RATES,
    )

    def spike_times(data: EvalData) -> list[np.ndarray]:
        converter = DataConverter(allow_stochastic_conversion=True, stochastic_seed=0)
        return converter.convert(data, DataType.SPIKE_TIMES).spike_times  # type: ignore[return-value]

    for got, want in zip(spike_times(with_spike_times), spike_times(only_counts), strict=True):
        np.testing.assert_allclose(got, want)


def test_rebinning_counts_matches_hz_rates() -> None:
    from ntt.evaluation.core.converters import DataConverter

    as_counts, as_hz = _counts_and_hz()
    converter = DataConverter()

    rebinned_counts = converter._rebin_rates(as_counts, 0.005)
    rebinned_hz = converter._rebin_rates(as_hz, 0.005)

    assert rebinned_counts.bin_size_seconds == pytest.approx(0.005)
    np.testing.assert_allclose(
        rebinned_counts.counts_per_bin(), rebinned_hz.counts_per_bin(), rtol=1e-5
    )


def test_coarse_integer_counts_do_not_convert_to_spike_times() -> None:
    original, reconstructed = _spike_pair(seed=5)
    coarse = EvaluationPipeline().run(
        EvalData.from_tensor(original, DataType.RATES, bin_size_seconds=0.01),
        EvalData.from_tensor(reconstructed, DataType.RATES, bin_size_seconds=0.01),
    )

    assert "spike_detection_f1" not in coarse.results


def test_negative_counts_are_not_observed_counts() -> None:
    counts = np.array([[0.0, 1.0, -1.0, 2.0]], dtype=np.float32)

    assert not EvalData(binned_counts=counts, bin_size_seconds=BIN_SECONDS).has_integer_counts()
    assert EvalData(
        binned_counts=np.abs(counts), bin_size_seconds=BIN_SECONDS
    ).has_integer_counts()


# ── The reconstruction's declared type reaches the eval ──────────────────────
#
# ``reconstruct()`` declares what it returns (``ReconstructionOutput.data_type``):
# calm Bernoulli-samples SPIKES from RATES input, cosmos Poisson-samples SPIKES
# although its spec says ``output_type=RATES``, PCA returns expected counts (RATES).


def test_reconstruct_from_input_reports_reconstruction_data_type() -> None:
    from tests.test_pca_baseline import _fitted_tokenizer, _make_input

    tok = _fitted_tokenizer(n_components=4)
    with torch.no_grad():
        x = _make_input(B=2, N=32, absolute_start=torch.tensor([0.0, 1.0], dtype=torch.float64))
        out = tok.reconstruct_from_input(x, return_latents=True)

    assert out["reconstruction_data_type"] == DataType.RATES


def test_perceiver_reconstruct_from_input_reports_reconstruction_data_type(tmp_path) -> None:
    """The perceiver overrides reconstruct_from_input; it declares RATES (ldns_perceiver)."""
    from tests.test_perceiver_tokenizer_extraction import (
        N_UNITS_PER_SESSION,
        _make_batched_input,
        _make_tokenizer,
        _write_embeddings_file,
    )

    tokenizer = _make_tokenizer(_write_embeddings_file(tmp_path)).eval()
    batch = _make_batched_input(batch_size=1, n_units=N_UNITS_PER_SESSION, n_bins=8)
    with torch.no_grad():
        out = tokenizer.reconstruct_from_input(batch, return_latents=True)

    assert out["reconstruction_data_type"] == tokenizer.spec.output_type == DataType.RATES


# ── Rate-output tokenizers return expected counts ────────────────────────────
#
# A tokenizer reconstructs what its ``output_type`` declares (as PCA already does):
# RATES -> expected counts per bin, deterministic; SPIKES -> sampled counts.
# cosmos and LDNS must not Poisson-sample in eval mode regardless of output_type.


def _cosmos(output_type: DataType):
    from ntt.contrib.tokenizer.cosmos import CosmosTokenizer2D
    from tests.test_cosmos_tokenizer import _make_batch, _make_spec_2d

    torch.manual_seed(0)
    spec = _make_spec_2d(output_type=output_type)
    return CosmosTokenizer2D(spec).eval(), _make_batch(spec)


def _ldns(output_type: DataType):
    from ntt.contrib.tokenizer.ldns import LDNSTokenizer
    from ntt.core.model.tokenizer.validation import make_dummy_input
    from tests.test_ldns_sigreg import _make_ldns_spec

    torch.manual_seed(0)
    spec = _make_ldns_spec(output_type=output_type)
    return LDNSTokenizer(spec).eval(), make_dummy_input(spec, batch_size=2)


@pytest.mark.parametrize("make", [_cosmos, _ldns], ids=["cosmos", "ldns"])
def test_rate_output_tokenizer_reconstructs_expected_counts(make) -> None:
    tokenizer, batch = make(DataType.RATES)

    with torch.no_grad():
        first = tokenizer.reconstruct_from_input(batch)
        second = tokenizer.reconstruct_from_input(batch)

    assert first.data_type == DataType.RATES
    assert torch.equal(first.reconstruction, second.reconstruction)  # no sampling noise
    assert not torch.equal(first.reconstruction, first.reconstruction.round())


@pytest.mark.parametrize("make", [_cosmos, _ldns], ids=["cosmos", "ldns"])
def test_spike_output_tokenizer_reconstructs_sampled_counts(make) -> None:
    tokenizer, batch = make(DataType.SPIKES)

    with torch.no_grad():
        out = tokenizer.reconstruct_from_input(batch)

    assert out.data_type == DataType.SPIKES
    assert torch.equal(out.reconstruction.float(), out.reconstruction.float().round())


def _collated_report_types(tokenizer_fn, **kwargs) -> set[tuple[DataType | None, DataType | None]]:
    from ntt.evaluation.config import EvaluationConfig
    from ntt.evaluation.utils.validation_collation import run_validation_evaluation_per_session
    from tests.test_neuron_cap import _make_loader

    spikes = (torch.rand(5, 3000, generator=torch.Generator().manual_seed(0)) < 0.05).float()
    reports = run_validation_evaluation_per_session(
        _make_loader(spikes, sampling_rate=1000.0),
        tokenizer_fn,
        data_type=DataType.SPIKES,
        bin_size_seconds=BIN_SECONDS,
        eval_config=EvaluationConfig(metrics={"fre": {"metric_class": "firing_rate_error"}}),
        progress=False,
        **kwargs,
    )
    return {
        (data["report"].original_data_type, data["report"].reconstructed_data_type)
        for segments in reports.values()
        for _segment, data in segments
    }


def test_collation_labels_reconstruction_with_its_declared_type() -> None:
    def tokenizer_fn(batch):
        return {
            "reconstruction": batch.spike_matrix.float() * 0.5,
            "latents": None,
            "reconstruction_data_type": DataType.RATES,
        }

    assert _collated_report_types(tokenizer_fn) == {(DataType.SPIKES, DataType.RATES)}


def test_collation_falls_back_to_given_reconstruction_type() -> None:
    def tokenizer_fn(batch):
        return {"reconstruction": batch.spike_matrix.float() * 0.5, "latents": None}

    types = _collated_report_types(tokenizer_fn, reconstructed_data_type=DataType.RATES)

    assert types == {(DataType.SPIKES, DataType.RATES)}


def test_collation_without_declared_type_keeps_input_type() -> None:
    def tokenizer_fn(batch):
        return {"reconstruction": batch.spike_matrix.float(), "latents": None}

    assert _collated_report_types(tokenizer_fn) == {(DataType.SPIKES, DataType.SPIKES)}


def test_collation_rejects_batches_declaring_different_types() -> None:
    calls = iter([DataType.SPIKES, DataType.RATES, DataType.RATES, DataType.RATES])

    def tokenizer_fn(batch):
        return {
            "reconstruction": batch.spike_matrix.float(),
            "latents": None,
            "reconstruction_data_type": next(calls),
        }

    with pytest.raises(ValueError, match="reconstruction_data_type"):
        _collated_report_types(tokenizer_fn)


def test_concatenated_tokenizer_outputs_keep_reconstruction_data_type() -> None:
    from ntt.evaluation.utils.neuron_covering import _concat_results

    parts = [
        {
            "reconstruction": torch.zeros(1, 4, 3),
            "latents": None,
            "reconstruction_data_type": DataType.RATES,
        }
        for _ in range(2)
    ]

    assert _concat_results(parts)["reconstruction_data_type"] == DataType.RATES


def test_oom_safe_forward_keeps_declared_type_of_reconstruction_output() -> None:
    from ntt.core.components.types import ReconstructionOutput
    from ntt.evaluation.utils.neuron_covering import oom_safe_forward
    from tests.test_neuron_cap import _make_loader

    spikes = (torch.rand(3, 2000, generator=torch.Generator().manual_seed(0)) < 0.05).float()
    batch = next(iter(_make_loader(spikes, sampling_rate=1000.0, n_windows=2)))

    def tokenizer_fn(b):
        return ReconstructionOutput(
            reconstruction=b.spike_matrix.float(),
            data_type=DataType.SPIKES,
            bin_size_seconds=BIN_SECONDS,
        )

    result, _ = oom_safe_forward(tokenizer_fn, batch, batch_size=2)

    assert result["reconstruction_data_type"] == DataType.SPIKES


@pytest.mark.parametrize("n_units", [3, 7])  # one group, and a covering of several groups
def test_neuron_covering_keeps_reconstruction_data_type(n_units: int) -> None:
    from ntt.evaluation.utils.neuron_covering import NeuronCoveringWrapper
    from tests.test_neuron_cap import _make_loader

    def tokenizer_fn(batch):
        return {
            "reconstruction": batch.spike_matrix.float() * 0.5,
            "latents": None,
            "reconstruction_data_type": DataType.RATES,
        }

    spikes = (torch.rand(n_units, 2000, generator=torch.Generator().manual_seed(0)) < 0.05).float()
    batch = next(iter(_make_loader(spikes, sampling_rate=1000.0, n_windows=2)))

    out = NeuronCoveringWrapper(tokenizer_fn=tokenizer_fn, n_neurons=4, seed=0)(batch)

    assert out["reconstruction_data_type"] == DataType.RATES


def test_replica_driver_labels_reconstruction_with_its_declared_type() -> None:
    """M1 (input robustness) builds its EvalData through the replica driver."""
    from torch.utils.data import DataLoader

    from ntt.core.data.datamodule import collate
    from ntt.core.data.replica.sampler import ReplicaWindowIndex, WindowReplicaBatchSampler
    from ntt.evaluation.config import EvaluationConfig
    from ntt.evaluation.utils.replica_collation import (
        run_validation_evaluation_per_session_replicas,
    )
    from tests.evaluation.utils.test_replica_collation import (
        _identity_tokenizer_with_latents,
        _IdentityReplicaDataset,
        _SeqSourceSampler,
    )

    def tokenizer_fn(batch):
        out = _identity_tokenizer_with_latents(batch)
        out["reconstruction"] = out["reconstruction"].float() * 0.5
        out["reconstruction_data_type"] = DataType.RATES
        return out

    loader = DataLoader(
        dataset=_IdentityReplicaDataset(n_units=3, time_bins_per_window=100),
        batch_sampler=WindowReplicaBatchSampler(
            base_sampler=_SeqSourceSampler(
                [ReplicaWindowIndex(recording_id="sess0", start=0.0, end=0.1, replica_id=0)]
            ),
            n_replicas=2,
            max_windows=None,
        ),
        collate_fn=collate,
    )

    reports = run_validation_evaluation_per_session_replicas(
        loader,
        tokenizer_fn,
        n_replicas=2,
        data_type=DataType.SPIKES,
        bin_size_seconds=BIN_SECONDS,
        eval_config=EvaluationConfig(metrics={"fre": {"metric_class": "firing_rate_error"}}),
        device=torch.device("cpu"),
        progress=False,
    )

    types = {
        (data["report"].original_data_type, data["report"].reconstructed_data_type)
        for segments in reports.values()
        for _segment, data in segments
    }
    assert types == {(DataType.SPIKES, DataType.RATES)}


def _m2_report_types(
    tmp_path, monkeypatch, *, output_type: DataType, declared: DataType | None
) -> tuple[str, str]:
    """Run the M2 job harness of ``tests/jobs`` and return the report's data types.

    M2 compares a clean reconstruction with noisy ones, so both sides are
    reconstructions and carry the reconstruction's type, not ``spec.input_type``.
    """
    import json

    from ntt.evaluation.metrics.robustness_latent_stats import LatentStats
    from ntt.jobs.eval_robustness_latent import NTTEvalRobustnessLatentJob
    from tests.jobs.test_eval_robustness_latent import (
        _make_run_job,
        _make_window_segment,
        _StubPipeline,
        _zero_noise,
    )

    job = _make_run_job(tmp_path)  # spec.input_type == RATES
    job._spec.output_type = output_type
    segments = [
        _make_window_segment(
            "sess0", 0, start_seconds=0.0, window_values_tu=[[1.0, 0.0], [0.0, 1.0]]
        ),
        _make_window_segment(
            "sess0", 1, start_seconds=1.0, window_values_tu=[[0.0, 1.0], [1.0, 0.0]]
        ),
    ]

    def tokenizer_fn(batch):
        out = {"reconstruction": batch}
        if declared is not None:
            out["reconstruction_data_type"] = declared
        return out

    def collect(**kwargs):
        kwargs["tokenizer_fn"](torch.zeros(1, 2, 2))  # the collector tokenizes every window
        return segments, ["sess0"], {"sess0": [0, 1]}

    monkeypatch.setattr("ntt.jobs.eval_robustness_latent.EvaluationPipeline", _StubPipeline)
    monkeypatch.setattr(
        NTTEvalRobustnessLatentJob, "_draw_noise_from_generator", staticmethod(_zero_noise)
    )
    monkeypatch.setattr(
        job, "_build_eval_loader_and_tokenizer_fn", lambda *a, **k: ("unused", tokenizer_fn)
    )
    monkeypatch.setattr(job, "_collect_selected_window_segments", collect)
    monkeypatch.setattr(
        job,
        "_compute_sigma",
        lambda segments, seed: LatentStats(
            sigma_joint=1.0, sigma_per_dim=np.ones((2,), dtype=np.float32)
        ),
    )

    # Every EvalData M2 builds (clean, truncated clean, noisy) is a reconstruction.
    built_types: set[str] = set()
    from_tensor = EvalData.from_tensor

    def recording_from_tensor(*args, **kwargs):
        data = from_tensor(*args, **kwargs)
        built_types.add(DataType(data.data_type).value)
        return data

    monkeypatch.setattr(EvalData, "from_tensor", recording_from_tensor)

    job.run()

    report = json.loads((tmp_path / "eval_report.json").read_text())
    assert built_types == {report["reconstructed_data_type"]}
    return report["original_data_type"], report["reconstructed_data_type"]


def test_m2_labels_reconstructions_with_spec_output_type(tmp_path, monkeypatch) -> None:
    types = _m2_report_types(tmp_path, monkeypatch, output_type=DataType.SPIKES, declared=None)

    assert types == (DataType.SPIKES.value, DataType.SPIKES.value)


def test_m2_labels_reconstructions_with_declared_type(tmp_path, monkeypatch) -> None:
    """cosmos: ``spec.output_type`` is RATES but ``reconstruct()`` returns Poisson spikes."""
    types = _m2_report_types(
        tmp_path, monkeypatch, output_type=DataType.RATES, declared=DataType.SPIKES
    )

    assert types == (DataType.SPIKES.value, DataType.SPIKES.value)

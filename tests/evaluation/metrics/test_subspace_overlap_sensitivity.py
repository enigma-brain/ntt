import numpy as np

from ntt.evaluation.metrics.population import (
    _concordance_correlation,
    _covariance_from_vectors,
    _subspace_overlap_from_covariances,
)


def _low_rank_population(rank, n_units, n_samples, noise, seed):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((n_samples, rank))
    loadings = rng.standard_normal((rank, n_units))
    clean = latents @ loadings
    recon = clean + noise * rng.standard_normal((n_samples, n_units))
    return clean, recon


def _overlap(noise, measure, seed=1):
    clean, recon = _low_rank_population(rank=50, n_units=80, n_samples=600, noise=noise, seed=seed)
    cov_clean = _covariance_from_vectors(clean)
    cov_recon = _covariance_from_vectors(recon)
    assert cov_clean is not None
    assert cov_recon is not None
    return _subspace_overlap_from_covariances(
        cov_clean, cov_recon, max_components=20, measure=measure
    )


def test_chordal_desaturates_where_mean_cosine_saturates():
    # Effective rank (~50) far exceeds k=20 — the real-data regime. mean-cosine saturates
    # across all reconstruction qualities; chordal must spread them out.
    noises = [0.05, 0.2, 0.8, 1.5]
    chordal = [_overlap(n, "chordal")[0] for n in noises]
    meancos = [_overlap(n, "mean_cosine")[0] for n in noises]
    for vals in (chordal, meancos):
        assert all(0.0 <= v <= 1.0 for v in vals)  # higher-better in [0,1]
        assert vals[0] > vals[-1]  # cleaner reconstruction scores higher
    chordal_range = max(chordal) - min(chordal)
    meancos_range = max(meancos) - min(meancos)
    assert chordal_range > 5 * meancos_range, (chordal_range, meancos_range)


def test_measure_metadata_exposes_both_and_selects_correctly():
    o_mc, meta = _overlap(0.2, "mean_cosine")
    assert {"subspace_overlap_mean_cosine", "subspace_overlap_chordal", "subspace_k"} <= set(meta)
    assert np.isclose(o_mc, meta["subspace_overlap_mean_cosine"])
    o_ch, _ = _overlap(0.2, "chordal")
    assert np.isclose(o_ch, meta["subspace_overlap_chordal"])


def test_metadata_exposes_variance_explained():
    # variance_explained_* must be present for the chordal method: some
    # downstream consumers read these keys, and they are an ingredient for any future
    # variance-aware component selection.
    _, meta = _overlap(0.2, "chordal")
    assert {"variance_explained_original", "variance_explained_reconstructed"} <= set(meta)
    k = meta["subspace_k"]
    assert len(meta["variance_explained_original"]) == k
    assert len(meta["variance_explained_reconstructed"]) == k


def test_subspace_overlap_compute_returns_full_metadata():
    # End-to-end through the metric class (not just the helper): the public result
    # must carry the new metadata keys and the tightened [0, 1] value range.
    from ntt.evaluation.metrics.population import SubspaceOverlap

    clean, recon = _low_rank_population(rank=50, n_units=80, n_samples=600, noise=0.2, seed=3)
    result = SubspaceOverlap().compute(
        {
            "population_vectors_original": clean,
            "population_vectors_reconstructed": recon,
        }
    )
    expected = {
        "principal_angles",
        "subspace_k",
        "subspace_overlap_mean_cosine",
        "subspace_overlap_chordal",
        "variance_explained_original",
        "variance_explained_reconstructed",
    }
    assert expected <= set(result.metadata)
    assert 0.0 <= result.raw_value <= 1.0
    assert result.value_range == (0.0, 1.0)


def test_population_coupling_warns_above_5ms_and_dedupes():
    import warnings as _warnings

    import pytest

    from ntt.evaluation.metrics.population import (
        _WARNED_COUPLING_BIN_SIZES,
        PopulationCouplingPreservation,
    )

    # Start from a clean slate so prior instantiations elsewhere don't mask the warning.
    _WARNED_COUPLING_BIN_SIZES.discard(10.0)
    with pytest.warns(UserWarning, match="population_coupling"):
        PopulationCouplingPreservation(population_bin_size_ms=10.0)

    # Second instantiation at the same coarse bin must not re-warn (warnings are deduplicated).
    with _warnings.catch_warnings(record=True) as recorded:
        _warnings.simplefilter("always")
        PopulationCouplingPreservation(population_bin_size_ms=10.0)
    assert not [w for w in recorded if "population_coupling" in str(w.message)]


def test_population_coupling_default_1ms_does_not_warn():
    import warnings as _warnings

    from ntt.evaluation.metrics.population import PopulationCouplingPreservation

    with _warnings.catch_warnings(record=True) as recorded:
        _warnings.simplefilter("always")
        PopulationCouplingPreservation()  # default population_bin_size_ms = 1.0
    assert not [w for w in recorded if "population_coupling" in str(w.message)]


def test_unknown_measure_raises():
    import pytest

    clean, recon = _low_rank_population(rank=6, n_units=40, n_samples=300, noise=0.1, seed=0)
    cov_clean = _covariance_from_vectors(clean)
    cov_recon = _covariance_from_vectors(recon)
    assert cov_clean is not None
    assert cov_recon is not None
    with pytest.raises(ValueError):
        _subspace_overlap_from_covariances(
            cov_clean, cov_recon, max_components=20, measure="bogus"
        )


def test_ccc_penalizes_scale_compression_where_pearson_does_not():
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = 0.1 * x + 0.45  # perfectly correlated but compressed toward the mean
    assert np.isclose(np.corrcoef(x, y)[0, 1], 1.0)  # Pearson blind to compression
    assert _concordance_correlation(x, y) < 0.6  # CCC penalizes it

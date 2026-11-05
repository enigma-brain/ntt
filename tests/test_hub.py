"""Tests for ``ntt.hub``: export a tokenizer to config.json + safetensors and load it back."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ntt.contrib.tokenizer.compressed_sensing import CompressedSensingTokenizerSpec
from ntt.contrib.tokenizer.perceiver.perceiver_tokenizer import (
    PerceiverTokenizerSpec,
    _load_embeddings,
    save_embeddings_safetensors,
)
from ntt.contrib.tokenizer.simple_vae import SimpleVAETokenizerSpec
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.model.checkpoint_io import save_live_tokenizer_as_checkpoint
from ntt.core.model.tokenizer import create_tokenizer
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec
from ntt.core.model.tokenizer.validation import make_dummy_input
from ntt.hub import export_tokenizer, load_pretrained


def _cs_spec(**overrides) -> CompressedSensingTokenizerSpec:
    """A tiny compressed-sensing spec cheap enough for test use."""
    defaults = {
        "n_neurons": 4,
        "window_length_seconds": 0.5,
        "bin_size_ms": 10.0,
        "compress_mode": "per_neuron",
        "latent_dim": 8,
        "trainable": False,
        "random_seed": 123,
        "fista_iterations": 5,
    }
    defaults.update(overrides)
    return CompressedSensingTokenizerSpec(**defaults)


def _vae_spec() -> SimpleVAETokenizerSpec:
    return SimpleVAETokenizerSpec(latent_dim=4, hidden_dims=(8,))


SPECS = {"cs": _cs_spec, "simple_vae": _vae_spec}


def _make_ckpt(spec: TokenizerSpec, tmp_path: Path) -> Path:
    return save_live_tokenizer_as_checkpoint(create_tokenizer(spec), spec, tmp_path / "m.ckpt")


def _poisson_batch(spec: TokenizerSpec) -> BinnedSpikesInput:
    batch = make_dummy_input(spec, batch_size=2)
    assert isinstance(batch, BinnedSpikesInput)
    generator = torch.Generator().manual_seed(0)
    counts = torch.poisson(torch.full(batch.spike_matrix.shape, 0.2), generator=generator)
    return batch._replace(spike_matrix=counts.to(batch.spike_matrix.dtype))


def _assert_identical(
    original: BaseTokenizer,
    loaded: BaseTokenizer,
    spec: TokenizerSpec,
    batch: BinnedSpikesInput | None = None,
):
    """Same state dict, same encode output and same forward reconstruction on a seeded batch."""
    assert original.state_dict().keys() == loaded.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key

    batch = batch if batch is not None else _poisson_batch(spec)
    # simple_vae samples its latent: seed identically before each call.
    with torch.no_grad():
        torch.manual_seed(1)
        encoded = original.encode(batch)
        torch.manual_seed(1)
        reloaded_encoded = loaded.encode(batch)
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(reloaded_encoded, torch.Tensor)
        assert torch.equal(encoded, reloaded_encoded)
        torch.manual_seed(1)
        expected = original(batch)
        torch.manual_seed(1)
        actual = loaded(batch)
    assert torch.equal(expected.reconstruction, actual.reconstruction)


@pytest.mark.parametrize("kind", ["cs", "simple_vae"])
def test_roundtrip_identical_outputs(kind: str, tmp_path: Path):
    spec = SPECS[kind]()
    original = create_tokenizer(spec).eval()
    ckpt = save_live_tokenizer_as_checkpoint(original, spec, tmp_path / "m.ckpt")
    out = export_tokenizer(ckpt, tmp_path / "hub")

    loaded = load_pretrained(out)

    _assert_identical(original, loaded, spec)


def test_simple_vae_has_parameters():
    assert sum(p.numel() for p in create_tokenizer(_vae_spec()).parameters()) > 0


def test_export_writes_only_json_and_safetensors(tmp_path: Path):
    out = export_tokenizer(_make_ckpt(_cs_spec(), tmp_path), tmp_path / "hub")

    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.safetensors"]
    assert json.loads((out / "config.json").read_text())["format_version"] == 1


def test_export_rejects_absolute_path_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ckpt = _make_ckpt(_cs_spec(), tmp_path)
    original_dump = TokenizerSpec.model_dump

    def dump_with_path(self, *args, **kwargs):
        dumped = original_dump(self, *args, **kwargs)
        dumped["pretrained_weights_path"] = "/some/where/weights.pt"
        return dumped

    monkeypatch.setattr(TokenizerSpec, "model_dump", dump_with_path)

    with pytest.raises(ValueError, match="pretrained_weights_path"):
        export_tokenizer(ckpt, tmp_path / "hub")


def test_export_copies_declared_files_and_load_resolves_them(tmp_path: Path):
    # `hydra_target` is a real optional str field on every spec; it stands in for a side file.
    ckpt = _make_ckpt(_cs_spec(), tmp_path)
    side_file = tmp_path / "side.safetensors"
    side_file.write_text("{}")

    out = export_tokenizer(ckpt, tmp_path / "hub", files={"hydra_target": side_file})

    config = json.loads((out / "config.json").read_text())
    assert config["files"] == {"hydra_target": "side.safetensors"}
    assert config["spec"]["hydra_target"] == "side.safetensors"
    assert str(tmp_path) not in json.dumps(config)
    assert (out / "side.safetensors").read_text() == "{}"

    loaded = load_pretrained(out)

    assert loaded.spec.hydra_target == str((out / "side.safetensors").absolute())


def test_export_revalidates_old_spec(tmp_path: Path):
    spec = _cs_spec()
    original = create_tokenizer(spec).eval()
    ckpt = save_live_tokenizer_as_checkpoint(original, spec, tmp_path / "m.ckpt")
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec_dict = saved["hyper_parameters"]["tokenizer_spec"].model_dump()
    del spec_dict["binarize_spikes"]
    saved["hyper_parameters"]["tokenizer_spec"] = spec_dict
    torch.save(saved, ckpt)

    out = export_tokenizer(ckpt, tmp_path / "hub")
    loaded = load_pretrained(out)

    _assert_identical(original, loaded, spec)
    assert (
        loaded.spec.binarize_spikes
        == CompressedSensingTokenizerSpec.model_fields["binarize_spikes"].default
    )


def test_export_spec_with_listconfig_nested_in_a_dict_field(tmp_path: Path):
    """A ``vqpt_kwargs`` dict holding a ListConfig (as unpickled from a checkpoint) exports."""
    from omegaconf import ListConfig

    from ntt.contrib.tokenizer.cosmos.cosmos_2d import CosmosTokenizer2DSpec

    spec = CosmosTokenizer2DSpec(
        n_neurons=16,
        window_length_seconds=0.064,
        bottleneck_type="vqpt",
        token_bin_size_ms=16.0,
        vqpt_type="fsq",
        vqpt_kwargs={"levels": [8, 5, 5, 5]},
        channels=8,
        channels_mult=(1, 1),
        num_res_blocks=1,
        z_channels=8,
        latent_channels=4,
        neuron_compression=4,
        temporal_compression=16,
    )
    original = create_tokenizer(spec).eval()
    ckpt = save_live_tokenizer_as_checkpoint(original, spec, tmp_path / "m.ckpt")
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved["hyper_parameters"]["tokenizer_spec"].vqpt_kwargs["levels"] = ListConfig([8, 5, 5, 5])
    torch.save(saved, ckpt)

    out = export_tokenizer(ckpt, tmp_path / "hub")
    loaded = load_pretrained(out)

    config = json.loads((out / "config.json").read_text())
    assert config["spec"]["vqpt_kwargs"] == {"levels": [8, 5, 5, 5]}
    assert loaded.spec == spec
    _assert_identical(original, loaded, spec)


def test_load_pretrained_rejects_unknown_format_version(tmp_path: Path):
    out = export_tokenizer(_make_ckpt(_cs_spec(), tmp_path), tmp_path / "hub")
    config = json.loads((out / "config.json").read_text())
    config["format_version"] = 2
    (out / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="format_version"):
        load_pretrained(out)


# --- Perceiver: NuCLR embeddings as safetensors ---

_SESSIONS = {"sess_a": 3, "sess_b": 4, "sess_c": 2}
_D_EMBED = 8


def _save_tables(tmp_path: Path) -> tuple[Path, Path]:
    """The same tiny table (ids shaped like the real file) as ``.pt`` and ``.safetensors``."""
    unit_ids = np.array([f"{s}/{i}" for s, n in _SESSIONS.items() for i in range(n)])
    session_ids = np.array([s for s, n in _SESSIONS.items() for _ in range(n)])
    embs = torch.randn(len(unit_ids), _D_EMBED, generator=torch.Generator().manual_seed(0))
    pt_path = tmp_path / "embs.pt"
    torch.save({"embs": embs, "unit_ids": unit_ids, "session_ids": session_ids}, pt_path)
    st_path = save_embeddings_safetensors(
        tmp_path / "embs.safetensors", embs, unit_ids.tolist(), session_ids.tolist()
    )
    return pt_path, st_path


def _perceiver_spec(embeddings_path: str) -> PerceiverTokenizerSpec:
    return PerceiverTokenizerSpec(
        n_neurons=4,
        window_length_seconds=0.5,
        bin_size_ms=50.0,
        hidden_dim=8,
        latent_tokens=2,
        d_embed=_D_EMBED,
        num_heads=2,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=1,
        use_s4=False,
        grad_checkpointing=False,
        embeddings_path=embeddings_path,
    )


def _perceiver_batch(session_id: str) -> BinnedSpikesInput:
    """A Poisson batch on a real table session (the dummy batch's session is special-cased)."""
    batch_size, n_bins, n_units = 2, 10, 3
    generator = torch.Generator().manual_seed(0)
    counts = torch.poisson(torch.full((batch_size, n_bins, n_units), 0.5), generator=generator)
    return BinnedSpikesInput(
        spike_matrix=counts.to(torch.int32),
        pad_and_gap_mask=torch.ones((batch_size, n_bins, n_units), dtype=torch.int32),
        sampling_rate=20.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1),
        unit_ids_pad_mask=torch.ones((batch_size, n_units), dtype=torch.int32),
        duration_sec=torch.full((batch_size,), 0.5),
        session_id=(session_id,) * batch_size,
        subject_id=("subj",) * batch_size,
        species=("mouse",) * batch_size,
        absolute_start=torch.zeros((batch_size,), dtype=torch.float64),
        absolute_end=torch.full((batch_size,), 0.5, dtype=torch.float64),
    )


def _export_perceiver(tmp_path: Path) -> tuple[PerceiverTokenizerSpec, BaseTokenizer, Path, Path]:
    _, st_path = _save_tables(tmp_path)
    spec = _perceiver_spec(str(st_path))
    # load_pretrained returns an eval-mode model; training mode would apply random neuron masking.
    original = create_tokenizer(spec).eval()
    ckpt = save_live_tokenizer_as_checkpoint(original, spec, tmp_path / "m.ckpt")
    out = export_tokenizer(ckpt, tmp_path / "hub", files={"embeddings_path": st_path})
    return spec, original, st_path, out


def test_safetensors_embeddings_match_pt(tmp_path: Path):
    pt_path, st_path = _save_tables(tmp_path)
    pt_embs, pt_offset = _load_embeddings(str(pt_path))
    st_embs, st_offset = _load_embeddings(str(st_path))
    assert torch.equal(pt_embs, st_embs)
    assert pt_offset == st_offset


def test_perceiver_export_roundtrip(tmp_path: Path):
    spec, original, st_path, out = _export_perceiver(tmp_path)
    config = json.loads((out / "config.json").read_text())
    assert config["files"] == {"embeddings_path": st_path.name}
    assert str(tmp_path) not in json.dumps(config)

    loaded = load_pretrained(out)
    _assert_identical(original, loaded, spec, batch=_perceiver_batch("sess_b"))


def test_perceiver_unknown_session_raises(tmp_path: Path):
    _, _, _, out = _export_perceiver(tmp_path)
    loaded = load_pretrained(out)
    with pytest.raises(KeyError, match="sess_dropped"), torch.no_grad():
        loaded.encode(_perceiver_batch("sess_dropped"))


# ── hf:// URIs ──────────────────────────────────────────────────────────────
_HF_URI = "hf://the-enigma-project/ntt-tokenizers/conv-vae-100to1"


@pytest.fixture
def hf_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fake the Hub: ``snapshot_download`` returns a tmp dir holding an exported model folder."""
    from unittest import mock

    spec = _cs_spec()
    original = create_tokenizer(spec).eval()
    ckpt = save_live_tokenizer_as_checkpoint(original, spec, tmp_path / "src.ckpt")
    snapshot = tmp_path / "snapshot"
    export_tokenizer(ckpt, snapshot / "conv-vae-100to1")
    monkeypatch.setenv("NTT_HUB_CACHE", str(tmp_path / "cache"))
    with mock.patch("ntt.hub.snapshot_download", return_value=str(snapshot)) as download:
        yield download, original, spec


def test_parse_hf_uri():
    from ntt.hub import parse_hf_uri

    repo = "the-enigma-project/ntt-tokenizers"
    assert parse_hf_uri(_HF_URI) == (repo, "conv-vae-100to1", None)
    assert parse_hf_uri(_HF_URI + "@v1.0") == (repo, "conv-vae-100to1", "v1.0")
    assert parse_hf_uri(f"hf://{repo}/a/b") == (repo, "a/b", None)
    for bad in ("hf://a/b", "hf:/a/b/c", "hf://a/b/c@"):
        with pytest.raises(ValueError, match="hf://<owner>/<repo>/<name>"):
            parse_hf_uri(bad)


def test_materialize_checkpoint_loads_with_lightning_loader(hf_hub, tmp_path: Path):
    from ntt.core.model import load_tokenizer_from_checkpoint
    from ntt.hub import materialize_checkpoint

    download, original, spec = hf_hub
    cache = tmp_path / "cache"

    ckpt = materialize_checkpoint(_HF_URI + "@v1.0", cache_dir=cache)

    download.assert_called_once()
    assert download.call_args.kwargs["allow_patterns"] == ["conv-vae-100to1/*"]
    assert download.call_args.kwargs["revision"] == "v1.0"
    assert ckpt == cache / "the-enigma-project/ntt-tokenizers/v1.0/conv-vae-100to1/model.ckpt"
    loaded = load_tokenizer_from_checkpoint(ckpt, weights_only=False)
    _assert_identical(original, loaded, spec)


def test_materialize_checkpoint_default_cache_from_env(hf_hub, tmp_path: Path):
    from ntt.hub import materialize_checkpoint

    ckpt = materialize_checkpoint(_HF_URI)

    assert (
        ckpt
        == tmp_path / "cache/the-enigma-project/ntt-tokenizers/main/conv-vae-100to1/model.ckpt"
    )
    assert ckpt.is_file()


def test_materialize_checkpoint_writes_through_a_temp_file(hf_hub, tmp_path: Path):
    """The final path only ever appears complete (``os.replace`` of a sibling temp file)."""
    from unittest import mock

    from ntt import hub

    final = tmp_path / "cache/the-enigma-project/ntt-tokenizers/main/conv-vae-100to1/model.ckpt"
    real_save = hub.save_live_tokenizer_as_checkpoint

    def save_elsewhere(tokenizer, spec, path):
        assert Path(path) != final and Path(path).parent == final.parent
        assert not final.exists()
        return real_save(tokenizer, spec, path)

    with mock.patch.object(hub, "save_live_tokenizer_as_checkpoint", side_effect=save_elsewhere):
        ckpt = hub.materialize_checkpoint(_HF_URI)

    assert ckpt == final and ckpt.is_file()
    assert [p.name for p in final.parent.iterdir()] == ["model.ckpt"]


def test_materialize_checkpoint_leaves_no_partial_file_on_failure(hf_hub, tmp_path: Path):
    from unittest import mock

    from ntt import hub

    def save_partially(tokenizer, spec, path):
        Path(path).write_bytes(b"truncated")
        raise OSError("disk full")

    with (
        mock.patch.object(hub, "save_live_tokenizer_as_checkpoint", side_effect=save_partially),
        pytest.raises(OSError, match="disk full"),
    ):
        hub.materialize_checkpoint(_HF_URI)

    model_dir = tmp_path / "cache/the-enigma-project/ntt-tokenizers/main/conv-vae-100to1"
    assert list(model_dir.iterdir()) == []


def test_load_pretrained_unknown_model_name(hf_hub):
    download, _, _ = hf_hub

    with pytest.raises(
        FileNotFoundError,
        match="no model 'no-such-model' in the-enigma-project/ntt-tokenizers@v1.0",
    ):
        load_pretrained("no-such-model", revision="v1.0")
    download.assert_called_once()


def test_load_pretrained_local_dir_without_config(hf_hub, tmp_path: Path):
    download, _, _ = hf_hub
    empty = tmp_path / "not-an-export"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="holds no config.json"):
        load_pretrained(empty)
    download.assert_not_called()


def test_export_refuses_a_non_empty_out_dir(tmp_path: Path):
    ckpt = _make_ckpt(_cs_spec(), tmp_path)
    out = tmp_path / "hub"
    out.mkdir()
    (out / "stale.safetensors").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="not empty"):
        export_tokenizer(ckpt, out)
    assert [p.name for p in out.iterdir()] == ["stale.safetensors"]


def test_export_into_an_existing_empty_dir(tmp_path: Path):
    out = tmp_path / "hub"
    out.mkdir()

    export_tokenizer(_make_ckpt(_cs_spec(), tmp_path), out)

    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.safetensors"]


def test_load_pretrained_rejects_a_disallowed_target_without_running_it(tmp_path: Path):
    """A ``_target_`` in config.json reaches ``hydra.utils.instantiate``: only Dists pass."""
    out = export_tokenizer(_make_ckpt(_cs_spec(), tmp_path), tmp_path / "hub")
    marker = tmp_path / "pwned"
    config = json.loads((out / "config.json").read_text())
    config["spec"]["window_length_seconds"] = {"_target_": "os.mkdir", "path": str(marker)}
    (out / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"spec\.window_length_seconds\._target_ = 'os\.mkdir'"):
        load_pretrained(out)
    assert not marker.exists()


def test_dist_targets_are_allowed():
    """The Dist classes' own dumps (the real specs' only ``_target_``) pass the allow-list."""
    from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec
    from ntt.core.components.dist import GaussianDist, TriangleDist, UniformDist
    from ntt.hub import _reject_unknown_targets

    windows = [
        UniformDist(low=0.1, high=2.0),
        TriangleDist(min_units=1, mode_units=2, max_units=4),
        GaussianDist(mean=1.0, std=0.1, clip_low=0.5, clip_high=1.5),
    ]
    for window in windows:
        spec = CalmTokenizerSpec(
            window_length_seconds=window, n_neurons=128, bin_size_ms=1.0, patch_length=100
        )
        _reject_unknown_targets(spec.model_dump(mode="json"), "spec")


def _hf_cache_layout(folder: Path, cache: Path) -> Path:
    """Mirror ``folder`` like the HF cache: each file a symlink to an extension-less blob."""
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / "0123abcd" / folder.name
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for i, path in enumerate(sorted(folder.iterdir())):
        blob = blobs / f"blob{i:04d}"
        blob.write_bytes(path.read_bytes())
        (snapshot / path.name).symlink_to(blob)
    return snapshot


def test_perceiver_loads_from_hf_cache_layout(tmp_path: Path):
    spec, original, _, out = _export_perceiver(tmp_path)
    snapshot = _hf_cache_layout(out, tmp_path / "cache")
    loaded = load_pretrained(snapshot)
    _assert_identical(original, loaded, spec, batch=_perceiver_batch("sess_b"))


@pytest.mark.parametrize(
    "entry",
    ["../embs.safetensors", "/abs/embs.safetensors", "sub/embs.safetensors", "embs.pt", "", 5],
)
def test_load_pretrained_rejects_unsafe_files_entry(tmp_path: Path, entry: str | int):
    _, _, _, out = _export_perceiver(tmp_path)
    config = json.loads((out / "config.json").read_text())
    config["files"] = {"embeddings_path": entry}
    (out / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="files"):
        load_pretrained(out)


def test_export_rejects_a_side_file_load_would_reject(tmp_path: Path):
    ckpt = _make_ckpt(_cs_spec(), tmp_path)
    side_file = tmp_path / "side.pt"
    side_file.write_bytes(b"")

    with pytest.raises(ValueError, match="safetensors"):
        export_tokenizer(ckpt, tmp_path / "hub", files={"hydra_target": side_file})
    assert not (tmp_path / "hub").exists()


# ── hf:// paths to files and folders (e.g. Poisson references) ─────────────
@pytest.fixture
def hub_files(tmp_path: Path):
    """Fake the Hub: ``snapshot_download`` returns a tmp dir with a file and a folder."""
    from unittest import mock

    snapshot = tmp_path / "snapshot"
    (snapshot / "poisson_references" / "m1").mkdir(parents=True)
    (snapshot / "poisson_references" / "recon.json").write_text("{}")
    (snapshot / "poisson_references" / "m1" / "results_per_session.csv").write_text("")
    with mock.patch("ntt.hub.snapshot_download", return_value=str(snapshot)) as download:
        yield download, snapshot


def test_download_hub_path_returns_a_file_or_folder_in_the_snapshot(hub_files):
    from ntt.hub import download_hub_path

    download, snapshot = hub_files
    repo = "the-enigma-project/ntt-tokenizers"
    path = download_hub_path(f"hf://{repo}/poisson_references/recon.json@v1.0")
    assert path == snapshot / "poisson_references" / "recon.json"
    assert download.call_args.args == (repo,)
    assert download.call_args.kwargs["revision"] == "v1.0"
    assert set(download.call_args.kwargs["allow_patterns"]) == {
        "poisson_references/recon.json",
        "poisson_references/recon.json/*",
    }
    assert download_hub_path(f"hf://{repo}/poisson_references/m1").is_dir()


def test_download_hub_path_missing_entry_names_the_uri(hub_files):
    from ntt.hub import download_hub_path

    with pytest.raises(FileNotFoundError, match="poisson_references/nope.json"):
        download_hub_path("hf://the-enigma-project/ntt-tokenizers/poisson_references/nope.json")


def test_resolve_artifact_path_downloads_hf_uris(hub_files):
    from ntt.evaluation.utils.paths import resolve_artifact_path

    _, snapshot = hub_files
    uri = "hf://the-enigma-project/ntt-tokenizers/poisson_references/recon.json"
    assert resolve_artifact_path(uri) == snapshot / "poisson_references" / "recon.json"
    assert resolve_artifact_path("/local/file.json") == Path("/local/file.json")


def test_download_hub_path_accepts_a_trailing_slash(hub_files):
    from ntt.hub import download_hub_path

    download, snapshot = hub_files
    path = download_hub_path("hf://the-enigma-project/ntt-tokenizers/poisson_references/m1/")
    assert path == snapshot / "poisson_references" / "m1"
    patterns = download.call_args.kwargs["allow_patterns"]
    assert set(patterns) == {"poisson_references/m1", "poisson_references/m1/*"}


@pytest.mark.parametrize("revision", ["", "@v1"])
def test_robustness_reference_on_the_hub_folder_is_accepted(hub_files, revision: str):
    from ntt.jobs._poisson_reference import resolve_poisson_reference_paths

    _, snapshot = hub_files
    uri = f"hf://the-enigma-project/ntt-tokenizers/poisson_references/m1{revision}"
    csv_path, _ = resolve_poisson_reference_paths(uri, mode="M1")
    assert csv_path == snapshot / "poisson_references" / "m1" / "results_per_session.csv"


@pytest.mark.parametrize("suffix", ["", "/", "@v1", "/@v1"])
def test_robustness_reference_on_the_hub_must_be_a_folder(hub_files, suffix: str):
    """A results_per_session.csv alone would leave its sibling run_meta.json behind."""
    from ntt.jobs._poisson_reference import resolve_poisson_reference_paths

    download, _ = hub_files
    uri = (
        "hf://the-enigma-project/ntt-tokenizers/poisson_references/m1/"
        f"results_per_session.csv{suffix}"
    )
    with pytest.raises(ValueError, match="results folder"):
        resolve_poisson_reference_paths(uri, mode="M1")
    download.assert_not_called()

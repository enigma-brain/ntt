"""Publish NTT tokenizers as ``config.json`` + ``model.safetensors`` folders and load them back.

One folder per model. ``config.json`` holds the format version, the spec (JSON dump) and a
``files`` map (spec field -> basename of a ``.safetensors`` side file stored next to it, e.g.
embeddings).
No pickles are involved, so a folder is safe to host on the Hugging Face Hub.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import safetensors.torch
from huggingface_hub import snapshot_download
from pydantic import TypeAdapter

from ntt.core.model.checkpoint_io import save_live_tokenizer_as_checkpoint
from ntt.core.model.tokenizer import (
    BaseTokenizer,
    create_tokenizer,
    load_tokenizer_from_checkpoint_no_lightning,
)

log = logging.getLogger(__name__)

DEFAULT_REPO_ID: str = "the-enigma-project/ntt-tokenizers"
HUB_FORMAT_VERSION: int = 1

_CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "model.safetensors"

# The only Hydra ``_target_`` values a config.json may hold: the spec validator instantiates them
# (``TokenizerSpec._instantiate_hydra_dicts``), so anything else would run arbitrary code on load.
# ``PytorchDist`` is left out on purpose: it imports a module named in its own fields.
_ALLOWED_TARGETS = frozenset(
    f"ntt.core.components.dist.{name}" for name in ("UniformDist", "TriangleDist", "GaussianDist")
)


def _reject_unknown_targets(value: Any, path: str) -> None:
    """Raise if ``value`` holds a ``_target_`` outside ``_ALLOWED_TARGETS`` (at any depth)."""
    if isinstance(value, dict):
        target = value.get("_target_")
        if target is not None and target not in _ALLOWED_TARGETS:
            raise ValueError(
                f"config.json: {path}._target_ = {target!r} is not allowed; "
                f"loading instantiates only {sorted(_ALLOWED_TARGETS)}."
            )
        for key, item in value.items():
            _reject_unknown_targets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unknown_targets(item, f"{path}[{index}]")


def _find_local_references(value: Any, path: str) -> str | None:
    """Return the dotted path of the first string in ``value`` that is a local path or URL."""
    if isinstance(value, str):
        if value.startswith("/") or "://" in value:
            return path
    elif isinstance(value, dict):
        for key, item in value.items():
            found = _find_local_references(item, f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_local_references(item, f"{path}[{index}]")
            if found:
                return found
    return None


def _check_side_file_name(field: str, name: Any) -> None:
    """Raise unless ``name`` is a plain ``*.safetensors`` file name (no directory part).

    Only safetensors on purpose: other formats may be read with pickle (e.g. the perceiver's
    ``.pt`` embeddings), and a directory part could point outside the model folder.
    """
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
        raise ValueError(
            f"'files' entry {field!r}={name!r} must be a plain .safetensors file name "
            "(perceiver embeddings: convert with save_embeddings_safetensors)"
        )


def export_tokenizer(
    ckpt_path: str | Path,
    out_dir: str | Path,
    *,
    files: dict[str, str | Path] | None = None,
) -> Path:
    """Export the tokenizer of a checkpoint to ``out_dir/{config.json,model.safetensors}``.

    Args:
        ckpt_path: Lightning checkpoint holding the tokenizer spec and weights.
        out_dir: Destination folder; created if missing, and must be empty if it exists.
        files: Spec field -> local ``.safetensors`` file. Each file is copied into ``out_dir``
            under its basename and the spec field is replaced by that basename.

    Raises:
        FileExistsError: If ``out_dir`` exists and is not empty.
        ValueError: If a spec field not listed in ``files`` holds an absolute path or URL, or
            a ``files`` entry is not a ``.safetensors`` file.
    """
    files = files or {}
    for field, source in files.items():
        _check_side_file_name(field, Path(source).name)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        # Stale files from an earlier export would be uploaded alongside the new ones.
        raise FileExistsError(f"Export directory {out_dir} is not empty; use a new or empty one.")
    tokenizer = load_tokenizer_from_checkpoint_no_lightning(ckpt_path, allow_spec_update=True)
    spec_dump = tokenizer.spec.model_dump(mode="json")

    for field, value in spec_dump.items():
        if field in files:
            continue
        offending = _find_local_references(value, field)
        if offending:
            raise ValueError(
                f"Spec field '{offending}' holds an absolute path or URL ({value!r}); "
                "pass it through `files` or clear it before exporting."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    basenames: dict[str, str] = {}
    for field, source in files.items():
        basenames[field] = Path(source).name
        shutil.copyfile(source, out_dir / basenames[field])
        spec_dump[field] = basenames[field]

    config = {"format_version": HUB_FORMAT_VERSION, "spec": spec_dump, "files": basenames}
    (out_dir / _CONFIG_NAME).write_text(json.dumps(config, indent=2))
    safetensors.torch.save_file(tokenizer.state_dict(), str(out_dir / _WEIGHTS_NAME))
    return out_dir


def load_pretrained(
    name_or_path: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    device: str = "cpu",
) -> BaseTokenizer:
    """Load a tokenizer from a local export folder or from a model folder on the Hub.

    An existing local directory is used as is and must contain ``config.json``; otherwise
    ``name_or_path`` is the model folder inside ``repo_id`` (at ``revision``).

    Raises:
        FileNotFoundError: If the local directory has no ``config.json``, or ``repo_id`` at
            ``revision`` has no model folder ``name_or_path``.
        ValueError: If ``config.json`` has another format version, a disallowed ``_target_``,
            or a ``files`` entry that is not a plain ``.safetensors`` file name.
    """
    local_dir = Path(name_or_path)
    if local_dir.is_dir():
        if not (local_dir / _CONFIG_NAME).is_file():
            raise FileNotFoundError(
                f"{local_dir} is a directory but holds no {_CONFIG_NAME}; "
                "pass an export folder or the name of a model on the Hub."
            )
    else:
        name = str(name_or_path)
        snapshot = snapshot_download(repo_id, revision=revision, allow_patterns=[f"{name}/*"])
        local_dir = Path(snapshot) / name
        if not (local_dir / _CONFIG_NAME).is_file():
            raise FileNotFoundError(f"no model '{name}' in {repo_id}@{revision or 'main'}")

    config = json.loads((local_dir / _CONFIG_NAME).read_text())
    if config.get("format_version") != HUB_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format_version {config.get('format_version')!r} in {local_dir}; "
            f"this ntt reads version {HUB_FORMAT_VERSION}."
        )

    spec_dict = dict(config["spec"])
    _reject_unknown_targets(spec_dict, "spec")
    # Side files must resolve before the spec is validated: some tokenizers read them in __init__.
    for field, basename in config.get("files", {}).items():
        _check_side_file_name(field, basename)
        # absolute(), not resolve(): the HF cache stores files as symlinks to extension-less
        # blobs, and readers dispatch on the file name.
        spec_dict[field] = str((local_dir / basename).absolute())

    import ntt.contrib.tokenizer  # noqa: F401  # isort: skip  (registers contrib kinds)
    from ntt.contrib import TokenizerSpecKind  # isort: skip

    spec = TypeAdapter(TokenizerSpecKind).validate_python(spec_dict)
    tokenizer = create_tokenizer(spec)
    state_dict = safetensors.torch.load_file(str(local_dir / _WEIGHTS_NAME))
    tokenizer.load_state_dict(state_dict, strict=True)
    return tokenizer.to(device).eval()


_HF_URI_PATTERN = re.compile(r"^hf://([^/@]+)/([^/@]+)/([^@]+?)(?:@([^@/]+))?$")


def parse_hf_uri(uri: str) -> tuple[str, str, str | None]:
    """Split ``hf://<owner>/<repo>/<name>[@<revision>]`` into ``(owner/repo, name, revision)``.

    ``name`` may contain ``/``; ``revision`` may not (the first ``/`` after the repo starts the
    name), so refs such as ``refs/pr/3`` are not expressible here. Use a branch, tag or commit
    sha without ``/``, or call ``load_pretrained(name, repo_id=..., revision=...)`` directly.

    Raises:
        ValueError: If ``uri`` does not have that form.
    """
    match = _HF_URI_PATTERN.match(uri)
    if match is None:
        raise ValueError(
            f"Malformed Hub URI {uri!r}; expected hf://<owner>/<repo>/<name>[@<revision>]"
        )
    owner, repo, name, revision = match.groups()
    return f"{owner}/{repo}", name, revision


def download_hub_path(uri: str) -> Path:
    """Download the file or folder an ``hf://<owner>/<repo>/<path>[@<revision>]`` URI names.

    Returns its path in the Hugging Face cache (``$HF_HOME``), e.g. a Poisson reference
    ``poisson_baseline.json`` to pass as ``job.poisson_reference``.

    Raises:
        FileNotFoundError: If the repo revision has no such file or folder.
    """
    repo_id, name, revision = parse_hf_uri(uri)
    name = name.rstrip("/")
    snapshot = snapshot_download(repo_id, revision=revision, allow_patterns=[name, f"{name}/*"])
    # The snapshot folder is named after the resolved Hub commit; log it, so a run on an
    # unpinned URI (the repo's default branch) records which version it read.
    log.info(f"{uri}: Hub commit {Path(snapshot).name}")
    # absolute(), not resolve(): in the HF cache, files are symlinks to extension-less blobs.
    path = Path(snapshot).absolute() / name
    if not path.exists():
        raise FileNotFoundError(f"{uri}: no such file or folder in {repo_id}")
    return path


def materialize_checkpoint(uri: str, cache_dir: str | Path | None = None) -> Path:
    """Turn an ``hf://`` URI into a local Lightning ``.ckpt`` that the eval loaders accept.

    The file is ``<cache>/<repo_id>/<revision or "main">/<name>/model.ckpt`` and is rewritten on
    every call, atomically (temp file in the same directory, then ``os.replace``), so jobs
    sharing the cache never read a partial file. ``cache_dir`` defaults to
    ``$NTT_HUB_CACHE``, else ``~/.cache/ntt/hub``. The download itself goes to the Hugging
    Face cache (``$HF_HOME``).
    """
    repo_id, name, revision = parse_hf_uri(uri)
    if cache_dir is None:
        cache_dir = os.environ.get("NTT_HUB_CACHE") or Path.home() / ".cache" / "ntt" / "hub"
    tokenizer = load_pretrained(name, repo_id=repo_id, revision=revision)
    out_path = Path(cache_dir) / repo_id / (revision or "main") / name / "model.ckpt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per writer; created by torch.save, so it gets the usual umask permissions.
    tmp_path = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_live_tokenizer_as_checkpoint(tokenizer, tokenizer.spec, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path

"""Downstream-eval launcher job.

Runs ``ntt-downstream``'s extrinsic evaluations (behaviour **decoding** via
POYO; stimulus **encoding** via Cowley V4 / Wingert) on a *frozen* ntt
tokenizer checkpoint, exposed through the single ``ntt`` entrypoint::

    uv run ntt paths.run_name=smoke job=downstream_decoding \\
        job.mode=single-run job.ckpt=/abs/model.ckpt job.session=<sess> \\
        job.data_root=/abs/processed

The task is fixed by the chosen config (``job=downstream_decoding`` /
``job=downstream_encoding``); both compose this same job class.

Design contract (see docs/downstream_eval.md):

* ``ntt`` does **not** depend on / import ``ntt_downstream``. This job only
  builds a command line + environment and shells into the vendored submodule
  (``downstream``), which carries its own venv with the heavy
  ``torch+cu126`` / ``flash-attn`` stack. That breaks the dependency cycle
  (ntt-downstream already depends on ntt) and keeps ntt's base env light.
* The submodule runs against **this checkout's ntt**, not the ntt its lock pins
  (an overlay). The launcher owns the submodule's env
  (:meth:`NTTDownstreamJob._ensure_env`): the downstream lock minus ntt, plus
  this checkout installed editable with its dependencies. The console scripts
  therefore run with ``--no-sync``, never ``--frozen`` (which would re-sync the
  env to the lock and swap the host ntt back out).
* This job shells into the submodule's four versioned ``ntt-downstream-*``
  console scripts (``decode`` / ``encode`` / ``calibrate`` / ``benchmark``; see
  ``downstream/docs/cli.md``) via that overlaid venv. The launcher
  forwards promoted ``--flags`` plus raw ``key=value`` Hydra passthrough; no
  bash, no dependency on a mutable ``.sh`` env-var contract.
* Bucket A: ``NTT_SPLITS_DIR`` is pointed at the *host* repo's selections so any
  ``${ntt_split:...}`` override (multi-session / tier configs) resolves against
  this checkout.
* Bucket B: decoding session cohorts live in this repo under
  ``config/data/selections/decoding/`` and are flattened here (single-brainset).
  For calibrate and benchmark they are written into a sweep-config YAML
  (engine-dest keys) that the ``ntt-downstream-{calibrate,benchmark}`` console
  scripts load via ``--config-dir``.

This module must stay free of torch / lightning / ntt_downstream imports.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
import tomllib
import typing as T
from pathlib import Path

import pydantic as pdt
import rootutils
from omegaconf import OmegaConf

from ntt.jobs.base import BaseJob, Locals
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Env vars stripped from the child so the inner bare ``uv run`` (the
# orchestrators call it themselves) resolves the *submodule's* overlaid venv and
# the scripts ``cd`` into the right repo. Stripping these is load-bearing, not
# cosmetic: a leaked VIRTUAL_ENV / UV_PROJECT_ENVIRONMENT would make the child
# reuse ntt's venv (no torch / flash-attn), and an inherited SLURM_SUBMIT_DIR
# would make the orchestrators ``cd`` to the host repo instead of the submodule.
_STRIP_ENV = (
    "VIRTUAL_ENV",
    "UV_PROJECT",
    "UV_PROJECT_ENVIRONMENT",
    "UV_FROZEN",
    "UV_NO_SYNC",
    "UV_WORKING_DIRECTORY",
    "PYTHONPATH",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "NTT_HYDRA_RUN_DIR",
    "SLURM_SUBMIT_DIR",
)


# Written into ``<sub>/.venv`` after an overlay build: a hash of the inputs the
# env was built from (see ``_env_fingerprint``). With ``--no-sync`` nothing else
# notices a submodule bump or a host dependency change, so a missing or stale
# stamp triggers a rebuild (auto) or an error (verify). Hand builds have none:
# build with ``python -m ntt.jobs.downstream_env`` instead.
_OVERLAY_STAMP = "ntt-overlay.sha256"
# The command that builds + stamps the env outside a launch.
_BUILD_HELPER = "uv run python -m ntt.jobs.downstream_env"
# The overlay may legitimately move tracking libs (mlflow, wandb, …) to the
# host's floors; a move in one of these changes numerics, so it is a warning.
_CORE_PACKAGES = frozenset(
    {
        "torch", "numpy", "lightning", "temporaldata", "torch-brain", "brainsets",
        # tokenizer numerics
        "vector-quantize-pytorch", "einops", "flash-attn",
    }
)  # fmt: skip
# Lock directory next to ``<sub>/.venv`` that serialises overlay builds from one
# shared checkout. ``mkdir`` is atomic on network filesystems (unlike flock, which depends on
# mount options). A cold build takes ~1–3 min; a lock older than _LOCK_STALE_S
# is a crashed build (SIGKILL skips the release) and is broken.
_BUILD_LOCK = ".venv.ntt-overlay.lock"
_LOCK_TIMEOUT_S = 40 * 60
_LOCK_STALE_S = 30 * 60
_LOCK_POLL_S = 10
_BUMP_HINT = (
    "Don't patch the env in place: bump downstream's ntt pin + relock + release, "
    "then bump the submodule here."
)
# One finding of `uv pip check` (uv 0.10). Anything else it prints, apart from
# its "Checked …" / "Found …" headers, is treated as a failure (fail closed).
_PIP_CHECK_FINDING = re.compile(
    r"The package `(?P<pkg>[^`]+)` requires `(?P<req>[^`]+)`, "
    r"but (?:(?P<missing>it's not installed)|`[^`]+` is installed)"
)
_ENV_RECORD = "downstream_env.json"
_PKG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _norm(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


@contextlib.contextmanager
def _build_lock(path: Path) -> T.Iterator[None]:
    """Hold the ``mkdir`` lock ``path`` (see ``_BUILD_LOCK``) for an overlay build.

    Waits up to ``_LOCK_TIMEOUT_S`` for another holder, breaks a lock older than
    ``_LOCK_STALE_S`` (best effort: two waiters breaking the same stale lock at
    once can still race), and always releases on exit.
    """
    deadline = time.monotonic() + _LOCK_TIMEOUT_S
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            pass
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:  # released meanwhile: try again
            continue
        try:
            owner = (path / "owner").read_text().strip()
        except OSError:  # not written yet (or the holder died right after mkdir)
            owner = "unknown"
        if age > _LOCK_STALE_S:
            log.warning(
                "Breaking a stale overlay build lock %s (%s, %.0fs old).", path, owner, age
            )
            shutil.rmtree(path, ignore_errors=True)
            continue
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out after {_LOCK_TIMEOUT_S}s waiting for another downstream env build "
                f"(lock {path}, held by {owner}). If no build is running, remove the lock "
                "directory and relaunch."
            )
        log.info("Waiting for another downstream env build (lock %s, held by %s).", path, owner)
        time.sleep(_LOCK_POLL_S)
    try:
        (path / "owner").write_text(f"{socket.gethostname()} pid={os.getpid()}\n")
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _hub_ckpt_name(uri: str) -> tuple[str, str]:
    """``(name, identity)`` of an ``hf://`` checkpoint; identity is ``repo/name@revision``."""
    from ntt.hub import parse_hf_uri

    repo_id, name, revision = parse_hf_uri(uri)
    return name, f"{repo_id}/{name}@{revision or 'main'}"


def _ckpt_root_name(name: str, identity: str) -> str:
    """``<name>-<hash>``, ``<hash>`` being 8 hex of sha1 over the identity."""
    digest = hashlib.sha1(identity.encode()).hexdigest()[:8]
    return f"{name.replace('/', '--')}-{digest}"


def hub_benchmark_tag(uri: str) -> str:
    """The folder under the submodule's ``experiments_benchmark/`` that decoding benchmark mode
    writes for ``job.ckpt=<uri>`` (an ``hf://`` checkpoint): one per model and revision."""
    return f"ntt_downstream_{_ckpt_root_name(*_hub_ckpt_name(uri))}"


def _is_uri(value: str) -> bool:
    """True for non-local checkpoint references (mlflow-artifacts:/, mlflow:/, …)."""
    return "://" in value or value.startswith("mlflow")


class NTTDownstreamJob(
    BaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
):
    """Launch an ntt-downstream decoding / encoding eval as a subprocess.

    Subclasses the lightweight :class:`~ntt.jobs.base.BaseJob` (not
    ``NTTBaseJob``) so it never instantiates a tokenizer / datamodule — all
    heavy work happens in the submodule's environment.
    """

    KIND: T.Literal["downstream"] = "downstream"

    # --- what to run -----------------------------------------------------
    task: T.Literal["decoding", "encoding"] = "decoding"
    # decoding only; ignored for encoding
    mode: T.Literal["calibrate", "benchmark", "single-run"] = "single-run"

    # --- the user's frozen tokenizer ------------------------------------
    # local path, mlflow-artifacts:/ URI or hf://<owner>/<repo>/<name>[@rev] (converted
    # to a local .ckpt, see ntt.hub); required for single-run, calibrate, and encoding.
    # decoding/benchmark takes ONE tokenizer here (hf:// or local path; ntt builds a
    # single-tokenizer ``ckpt_root`` under results/downstream/benchmark_roots/) or a
    # cohort via ``ckpt_root``.
    ckpt: str | None = None
    # decoding/benchmark: a directory of tokenizers the contract discovers as
    # ``<ckpt_root>/<name>/checkpoints/model.ckpt`` (a symlink into that layout is
    # fine). Mutually exclusive with ``ckpt``.
    ckpt_root: str | None = None

    # --- data / cohort selection ----------------------------------------
    data_root: str | None = None
    # decoding/single-run: a single recording id
    session: str | None = None
    # decoding cohorts — names of ntt selections (relative to
    # config/data/selections/, e.g. "decoding/ntt_decoding_calib_eval").
    train_selection: str | None = None  # calibrate base-train -> TRAIN_SESSIONS
    eval_selection: str | None = None  # calibrate -> OTHER_SESSIONS / benchmark -> SESSIONS

    # --- decoding training knobs (None -> the downstream default) -------
    # Applies to single-run + benchmark; calibrate is step-driven (epochs /
    # model_cfg are ignored there — use job.extra for per-cell train.py overrides).
    epochs: int | None = None
    batch_size: int | None = None
    seed: int | None = None
    num_workers: int | None = None
    model_cfg: str | None = None  # POYO config: poyo_1.3M | poyo_1.3M_no_crossattn | poyo_11.8M
    unit_embedding_mode: str | None = None  # none | add | concat
    sentinels: bool | None = None

    # --- encoding -------------------------------------------------------
    # Either a bare Hydra config name under the submodule's configs/encoding/,
    # OR a path (containing "/" or ending .yaml/.yml) to an EXTERNAL Hydra config
    # — absolute, or relative to the ntt repo root — emitted as --config-dir
    # <dir> --config-name <stem>. Hydra ADDS the dir to its search path, so an
    # external overlay can compose the internal tree (defaults: [/cowley_v4_base,
    # _self_]). See config/downstream/encoding/example_overlay.yaml.
    recipe: str = "cowley_v4_base"
    # encoding loss mode. None -> "mse": predict the tokenizer's latents (the
    # generic tokenizer eval, works with any compress_t tokenizer). Override for
    # generative evals (flow_matching / diffusion / energy / mar / cross_entropy).
    # The recipe controls data + backbone + model architecture; this controls
    # the loss so the frozen tokenizer is always actually used.
    loss_mode: str | None = None

    # --- shared knobs ---------------------------------------------------
    out: str | None = None  # host-side output root (absolute is enforced)
    # space-separated GPU indices; None -> auto-discover. Only reaches
    # calibrate/benchmark (via the sweep config); ignored for mode=single-run.
    gpus: str | None = None
    # MLflow logging of a calibrate/benchmark sweep: "auto" (on iff MLFLOW_TRACKING_URI is
    # set) | true | false; None -> the submodule's default ("auto"). single-run logs to MLflow
    # only with job.extra=["mlflow.enabled=true"].
    mlflow: T.Literal["auto"] | bool | None = None
    extra: list[str] = pdt.Field(default_factory=list)  # raw passthrough overrides
    dry_run: bool = False  # print the resolved command + env, do not execute

    # --- advanced / overrides -------------------------------------------
    root_dir: str | None = None  # host repo root; auto-discovered when None
    submodule_dir: str = "downstream"
    # The submodule's env (see _ensure_env). auto: check it, build the overlay if
    # the check fails. verify: check only and raise — for fan-outs from one
    # shared checkout (the check is read-only; only a build would race).
    downstream_env: T.Literal["auto", "verify"] = "auto"

    _DEFAULT_SELECTIONS: T.ClassVar[dict[str, str]] = {
        "train": "decoding/ntt_decoding_calib_base_train",
        "calib_eval": "decoding/ntt_decoding_calib_eval",
        "benchmark_eval": "decoding/ntt_decoding_benchmark_eval",
    }

    # Fields each task's builder actually reads. Setting one on the *other* task
    # is a silent no-op without the guard below (the builder never looks at it).
    # (`gpus`/`data_root`/`seed` are deliberately *not* listed: gpus/data_root
    # ship in the encoding config and are tolerated there, and `seed` is a
    # real global seed on *both* tasks (encode calls lightning.seed_everything)
    # — over-rejecting them would break the real config.)
    _DECODING_ONLY_FIELDS: T.ClassVar[frozenset[str]] = frozenset(
        {
            "mode", "ckpt_root", "session", "train_selection", "eval_selection",
            "epochs", "batch_size", "num_workers", "model_cfg",
            "unit_embedding_mode", "sentinels", "mlflow",
        }
    )  # fmt: skip
    _ENCODING_ONLY_FIELDS: T.ClassVar[frozenset[str]] = frozenset({"recipe", "loss_mode"})

    @pdt.field_validator("gpus", "session", mode="before")
    @classmethod
    def _coerce_numeric_override_to_str(cls, v: object) -> object:
        """Allow numeric CLI overrides for the few fields that legitimately take them.

        ``job.gpus=0`` (a GPU index) and numeric recording ids arrive from Hydra
        as ints; coerce just these. Every other string field keeps strict
        validation, so a fat-fingered numeric (e.g. ``job.model_cfg=11``) is
        rejected at load time instead of silently flowing into the child command.
        """
        if isinstance(v, bool):  # bool is an int subclass — never a valid value here
            return v
        return str(v) if isinstance(v, (int, float)) else v

    def _overridden(self, names: frozenset[str]) -> list[str]:
        """Of ``names``, those the user set to a *non-default* value.

        Requires both ``model_fields_set`` membership AND a value differing from
        the field default, so the guard stays idempotent under a
        ``model_dump()`` -> reconstruct round-trip (a dumped default does not
        count as a user override). The cost — an explicit override *to* the
        default value is not flagged — is harmless, since such an override is a
        no-op anyway.
        """
        fields = type(self).model_fields
        return sorted(
            n
            for n in (names & self.model_fields_set)
            if getattr(self, n) != fields[n].get_default(call_default_factory=True)
        )

    @pdt.model_validator(mode="after")
    def _reject_misscoped_fields(self) -> T.Self:
        """Fail loudly when a field is set on the wrong task, or both ckpts given.

        The single class backs both ``job=downstream_decoding`` and
        ``job=downstream_encoding``; without this, a field meaningful only to the
        other task is accepted and silently dropped. The shipped (task-scoped)
        configs leave the other task's fields at their defaults, so this fires
        only on hand-set CLI overrides — see :meth:`_overridden`.
        """
        if self.task == "encoding":
            bad = self._overridden(self._DECODING_ONLY_FIELDS)
            if bad:
                raise ValueError(
                    f"job=downstream_encoding does not use decoding-only field(s): "
                    f"{', '.join(bad)}. Use job=downstream_decoding for those, or drop them."
                )
        else:  # decoding
            bad = self._overridden(self._ENCODING_ONLY_FIELDS)
            if bad:
                raise ValueError(
                    f"job=downstream_decoding does not use encoding-only field(s): "
                    f"{', '.join(bad)}. Use job=downstream_encoding for those, or drop them."
                )
        if self.ckpt is not None and self.ckpt_root is not None:
            raise ValueError(
                "job.ckpt and job.ckpt_root are mutually exclusive — pass exactly one "
                "(benchmark takes a directory of tokenizers via ckpt_root, or one "
                "tokenizer via ckpt; the other modes take a single ckpt)."
            )
        for name in ("data_root", "out", "ckpt_root"):
            value = getattr(self, name)
            if isinstance(value, str) and value.startswith("hf://"):
                raise ValueError(
                    f"job.{name}={value!r}: hf:// paths are accepted only for job.ckpt "
                    "(a single tokenizer checkpoint)."
                )
        return self

    # ── paths ───────────────────────────────────────────────────────────
    def _host_root(self) -> Path:
        if self.root_dir is not None:
            return Path(self.root_dir).expanduser().resolve()
        return Path(rootutils.find_root(search_from=__file__, indicator=".project-root"))

    def _abspath(self, value: str | None) -> str | None:
        """Resolve a local path to absolute; pass URIs through untouched."""
        if value is None:
            return None
        if _is_uri(value):
            return value
        return str(Path(value).expanduser().resolve())

    def _ckpt_path(self) -> str | None:
        """``job.ckpt`` for the child: an ``hf://`` URI becomes a local ``.ckpt``.

        Under ``dry_run`` nothing is downloaded and the URI is passed through as is.
        """
        if self.ckpt is None or not self.ckpt.startswith("hf://"):
            return self._abspath(self.ckpt)
        if self.dry_run:
            log.info("[dry-run] would download %s and convert it to a local .ckpt", self.ckpt)
            return self.ckpt
        from ntt.hub import materialize_checkpoint

        return str(materialize_checkpoint(self.ckpt))

    def _selection_sessions(self, host_root: Path, name: str) -> list[str]:
        """Flatten a single-brainset ntt SelectionConfig YAML to its session list."""
        path = host_root / "config" / "data" / "selections" / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"Selection {name!r} not found at {path}. "
                "Available decoding cohorts live under config/data/selections/decoding/."
            )
        doc = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        selection = doc.get("selection") if isinstance(doc, dict) else None
        if not isinstance(selection, list) or not selection:
            raise ValueError(f"{path}: expected a non-empty 'selection' list.")
        if len(selection) != 1:
            raise ValueError(
                f"{path}: downstream decoding cohorts must be single-brainset "
                f"(got {len(selection)} brainsets). The flagship scripts resolve "
                "${data_root}/${brainset}/${session}.h5 against one brainset."
            )
        sessions = [str(s) for s in (selection[0].get("sessions") or [])]
        if not sessions:
            raise ValueError(f"{path}: selection[0] has no 'sessions'.")
        return sessions

    def _write_sweep_config(self, path: Path, mapping: dict[str, object]) -> None:
        """Write the engine-dest sweep YAML the contract CLI loads via --config-dir.

        Written under dry_run too, so the printed command is re-runnable as-is.
        Keys are validated by the submodule launcher (unknown dest -> exit 2);
        omitted keys fall back to the engine's own defaults.

        The file is keyed solely by the caller-chosen ``path``, so concurrent
        runs writing the same ``path`` clobber each other's sweep YAML — callers
        must pick distinct paths for parallel sweeps.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(mapping), path)

    # ── environment ─────────────────────────────────────────────────────
    @staticmethod
    def _load_dotenv(path: Path, env: dict[str, str]) -> None:
        """Merge ``KEY=VALUE`` lines from a .env file (existing keys win).

        Policy is *user env wins* (a key already in ``env`` is not overwritten),
        which differs from the .sh orchestrators' ``set -a; source .env`` (file
        wins). It only matters for MLflow creds on the direct-python paths; an
        explicitly-exported value is intentionally respected over the .env.
        """
        if not path.is_file():
            return
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in env:
                env[key] = val

    def _clean_env(self, host_root: Path, *, source_dotenv: Path | None) -> dict[str, str]:
        env = dict(os.environ)
        stripped = [k for k in _STRIP_ENV if k in env]
        for k in stripped:
            env.pop(k, None)
        if stripped:
            log.debug("Stripped from child env: %s", ", ".join(stripped))
        # Bucket A: host repo is the source of truth for any ${ntt_split} override.
        env["NTT_SPLITS_DIR"] = str(host_root / "config" / "data" / "selections")
        # The .sh orchestrators self-source ${REPO_ROOT}/.env, but the direct
        # python paths (single-run, encoding) do not — pull MLflow creds etc. in.
        if source_dotenv is not None:
            self._load_dotenv(source_dotenv, env)
        return env

    def _cli(self, submodule: Path, verb: str) -> list[str]:
        """The contract invocation: the submodule's console script in its overlaid venv.

        ``--no-sync``, not ``--frozen``: the launcher owns the env
        (:meth:`_ensure_env`); ``--frozen`` would re-sync it to the lock and put
        the lock's ntt back in place of this checkout's.
        """
        return ["uv", "run", "--project", str(submodule), "--no-sync", f"ntt-downstream-{verb}"]

    # ── the submodule env: downstream lock + this checkout's ntt ────────────
    @staticmethod
    def _overlay_commands(submodule: Path, host_root: Path) -> list[list[str]]:
        """The two build steps, both run with ``cwd=submodule``.

        1. The downstream lock minus ntt (the lock pins ntt as a git source;
           ``--no-install-package`` means it is never fetched, step 2 installs
           this checkout instead).
        2. This checkout, editable, WITH its dependencies, so its floors win
           over the lock's (ntt main routinely requires newer tracking libs).
           ``--resolution lowest-direct`` + the installed packages as
           preferences: only packages below a host floor move, and only to that
           floor. A host dependency the lock lacks would get its OLDEST release,
           so :meth:`_check_env` refuses one. ``--no-sources``: the host's
           ``[tool.uv.sources]`` (git revs for its own dev env, one an unpinned
           branch) must not override the lock's versions. The cwd matters: uv
           pip reads the submodule's ``[tool.uv] override-dependencies``
           (einops<0.7, no ``pytorch-brain`` next to git ``torch_brain``) from
           the cwd's ``pyproject.toml``.
        """
        python = submodule / ".venv" / "bin" / "python"
        return [
            ["uv", "sync", "--project", str(submodule), "--frozen",
             "--no-install-package", "ntt", "--python", "3.13"],
            ["uv", "pip", "install", "--python", str(python), "--no-sources",
             "--resolution", "lowest-direct", "-e", str(host_root)],
        ]  # fmt: skip

    def _overlay_hint(self, submodule: Path, host_root: Path) -> str:
        """The hand-run build commands (for the dry-run log and debugging; no stamp)."""
        cmds = self._overlay_commands(submodule, host_root)
        return "\n".join(
            [f"    cd {shlex.quote(str(submodule))}", *(f"    {shlex.join(c)}" for c in cmds)]
        )

    @staticmethod
    def _build_hint(submodule: Path, host_root: Path) -> str:
        """The one command that builds, stamps and checks the env (``ntt.jobs.downstream_env``)."""
        return (
            f"    cd {shlex.quote(str(host_root))} && {_BUILD_HELPER} "
            f"--root-dir {shlex.quote(str(host_root))} "
            f"--submodule-dir {shlex.quote(str(submodule))}"
        )

    @staticmethod
    def _env_fingerprint(submodule: Path, host_root: Path) -> str:
        """Hash of what the overlay is built from (host deps + downstream lock)."""
        digest = hashlib.sha256()
        for path in (
            host_root / "pyproject.toml",
            submodule / "pyproject.toml",
            submodule / "uv.lock",
        ):
            digest.update(path.read_bytes() if path.is_file() else b"")
        return digest.hexdigest()

    @staticmethod
    def _overridden_packages(submodule: Path) -> tuple[set[str], set[str]]:
        """The submodule's ``[tool.uv] override-dependencies``: ``(all, removed)``.

        ``removed`` are the overrides with an environment marker
        (``pytorch-brain; python_version > '99'``): they take a package out of
        the resolution rather than constraining its version.
        """
        doc = tomllib.loads((submodule / "pyproject.toml").read_text())
        reqs = doc.get("tool", {}).get("uv", {}).get("override-dependencies", [])
        overridden, removed = set(), set()
        for req in reqs:
            if m := _PKG_NAME.match(req):
                overridden.add(_norm(m.group(0)))
                if ";" in req:
                    removed.add(_norm(m.group(0)))
        return overridden, removed

    def _pip_check(self, python: Path, submodule: Path, env: dict[str, str]) -> str | None:
        """``uv pip check``, minus what the submodule's overrides deliberately cause.

        The overrides are incompatibilities by design, so a bare ``uv pip check``
        never passes, not even on the plain locked env. Two kinds of finding are
        tolerated, nothing else:

        * a package other than ntt requires an overridden package (today:
          vector-quantize-pytorch's einops>=0.8 vs the einops<0.7 override) —
          downstream's lock already ships that pair;
        * ntt requires a package an override removes, and it is not installed
          (today: ``pytorch-brain``, replaced by git ``torch_brain``).

        A finding where ntt requires a version an override prevents (e.g. the
        host raises its einops floor to 0.8) is a real conflict: the override
        silently wins over the host. Unparseable output also fails (closed).
        """
        proc = subprocess.run(  # noqa: S603
            ["uv", "pip", "check", "--python", str(python)],
            cwd=submodule, env=env, capture_output=True, text=True,
        )  # fmt: skip
        if proc.returncode == 0:
            return None
        output = (proc.stdout + proc.stderr).strip()
        overridden, removed = self._overridden_packages(submodule)
        real, findings = [], 0
        for line in output.splitlines():
            if line.startswith(("Checked ", "Found ")):
                continue
            m = _PIP_CHECK_FINDING.fullmatch(line.strip())
            pkg = _norm(m["pkg"]) if m else None
            req = _PKG_NAME.match(m["req"]) if m else None
            if m is None or req is None:
                real.append(line)  # unparseable: fail closed
                continue
            findings += 1
            required = _norm(req.group(0))
            if pkg != "ntt" and required in overridden:
                continue
            if pkg == "ntt" and required in removed and m["missing"]:
                continue
            real.append(line)
        if not real and not findings:  # failed without telling us why
            return output or f"uv pip check exited {proc.returncode}"
        return "\n".join(real) or None

    def _check_env(
        self, submodule: Path, host_root: Path, env: dict[str, str]
    ) -> tuple[str | None, dict[str, list[str | None]], str | None]:
        """Read-only check of the overlay: ``(ntt.__file__, changes vs lock, problem)``."""
        python = submodule / ".venv" / "bin" / "python"
        if not python.is_file():
            return None, {}, f"{python} does not exist"
        proc = subprocess.run(  # noqa: S603
            [str(python), "-I", "-c", "import ntt; print(ntt.__file__)"],
            cwd=submodule, env=env, capture_output=True, text=True,
        )  # fmt: skip
        if proc.returncode != 0:
            return None, {}, f"`import ntt` fails in {python}:\n{proc.stderr.strip()}"
        ntt_file = proc.stdout.strip()
        host_src = (host_root / "src").resolve()
        if not Path(ntt_file).resolve().is_relative_to(host_src):
            return ntt_file, {}, f"ntt resolves to {ntt_file}, not under {host_src}"
        stamp = submodule / ".venv" / _OVERLAY_STAMP
        if not stamp.is_file():
            # Built by hand: we can't tell what from.
            return ntt_file, {}, f"no overlay stamp ({stamp})"
        if stamp.read_text().strip() != self._env_fingerprint(submodule, host_root):
            stale = (
                "the env was built from a different host pyproject.toml or downstream lock "
                f"({stamp} is stale)"
            )
            return ntt_file, {}, stale
        bad = self._pip_check(python, submodule, env)
        if bad:
            return ntt_file, {}, f"`uv pip check` reports:\n{bad}"
        changes = self._env_changes(python, submodule, env)
        new = [f"{n} {v}" for n, (was, v) in changes.items() if was is None and n != "ntt"]
        if new:
            # lowest-direct resolves a package nothing pins to its OLDEST release.
            refused = (
                f"new host dependencies not in downstream's lock: {', '.join(new)}. The "
                "overlay would install their oldest release, so it refuses them."
            )
            return ntt_file, changes, refused
        return ntt_file, changes, None

    def _env_changes(
        self, python: Path, submodule: Path, env: dict[str, str]
    ) -> dict[str, list[str | None]]:
        """Installed packages that differ from the submodule lock: name -> [locked, installed]."""
        lock = tomllib.loads((submodule / "uv.lock").read_text())
        locked: dict[str, set[str]] = {}
        for pkg in lock.get("package", []):
            locked.setdefault(_norm(pkg["name"]), set()).add(str(pkg.get("version")))
        proc = subprocess.run(  # noqa: S603
            ["uv", "pip", "list", "--python", str(python), "--format", "json"],
            cwd=submodule, env=env, capture_output=True, text=True, check=True,
        )  # fmt: skip
        changes: dict[str, list[str | None]] = {}
        for item in json.loads(proc.stdout):
            name, version = _norm(item["name"]), item["version"]
            if version not in locked.get(name, set()):
                was = " | ".join(sorted(locked[name])) if name in locked else None
                changes[name] = [was, version]
        return dict(sorted(changes.items()))

    def _build_overlay(self, submodule: Path, host_root: Path, env: dict[str, str]) -> None:
        """Run :meth:`_overlay_commands`, then write the stamp. Raises with uv's stderr."""
        for cmd in self._overlay_commands(submodule, host_root):
            log.info("$ %s", shlex.join(cmd))
            proc = subprocess.run(  # noqa: S603
                cmd, cwd=submodule, env=env, capture_output=True, text=True,
            )  # fmt: skip
            log.debug("%s", proc.stderr.strip())
            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
                raise RuntimeError(
                    f"Building the downstream env overlay failed (exit {proc.returncode}): "
                    f"{shlex.join(cmd)}\n{tail}\n"
                    "If uv found no solution, this host ntt's requirements conflict with the "
                    f"downstream lock. {_BUMP_HINT} (The build also needs network access to "
                    "PyPI and GitHub.)"
                )
        (submodule / ".venv" / _OVERLAY_STAMP).write_text(
            self._env_fingerprint(submodule, host_root)
        )

    def _ensure_env(self, submodule: Path, host_root: Path) -> dict[str, T.Any] | None:
        """Make sure the submodule's venv runs this checkout's ntt.

        Checks (read-only, the steady state): the venv imports ntt from
        ``host_root/src``, its overlay stamp exists and is current, ``uv pip
        check`` passes (:meth:`_pip_check`), and no host dependency is missing
        from the lock. If not, ``auto`` builds the overlay under a lock
        (:func:`_build_lock`; re-checking first, since a concurrent job may have
        just built it) and re-checks; ``verify`` raises. Returns the provenance
        record (the resolved ``ntt.__file__`` and the package changes versus the
        lock), or None under ``dry_run``, which makes no subprocess calls.
        """
        if self.dry_run:
            log.info(
                "[dry-run] downstream_env=%s: would check that %s imports ntt from %s and, "
                "if not, build the overlay:\n%s",
                self.downstream_env, submodule / ".venv", host_root / "src",
                self._overlay_hint(submodule, host_root),
            )  # fmt: skip
            return None
        env = self._clean_env(host_root, source_dotenv=None)
        ntt_file, changes, problem = self._check_env(submodule, host_root, env)
        built = False
        if problem is not None:
            if self.downstream_env == "verify":
                raise RuntimeError(
                    f"job.downstream_env=verify and the downstream env is not ready: {problem}\n"
                    "Build it once with job.downstream_env=auto, or run:\n"
                    + self._build_hint(submodule, host_root)
                )
            with _build_lock(submodule / _BUILD_LOCK):
                # Another job may have built it while we waited for the lock.
                ntt_file, changes, problem = self._check_env(submodule, host_root, env)
                if problem is not None:
                    log.info("Building the downstream env overlay (%s).", problem)
                    self._build_overlay(submodule, host_root, env)
                    built = True
                    ntt_file, changes, problem = self._check_env(submodule, host_root, env)
            if problem is not None:
                raise RuntimeError(
                    f"The downstream env is still not right after the overlay build: {problem}\n"
                    "This host ntt's requirements conflict with (or go beyond) the downstream "
                    f"lock in a way the overlay may not change. {_BUMP_HINT}"
                )
            if not built:
                log.info("Downstream env OK (built by a concurrent job).")
        else:
            log.info("Downstream env OK (no build needed).")
        log.info("Downstream env: ntt.__file__ = %s", ntt_file)
        log.info(
            "Downstream env vs its lock (locked -> installed): %s",
            ", ".join(f"{n} {a} -> {b}" for n, (a, b) in changes.items()) or "none",
        )
        core = sorted(_CORE_PACKAGES & changes.keys())
        if core:
            log.warning(
                "The host ntt overlay moved core packages off the downstream lock: %s", core
            )
        return {
            "ntt_file": ntt_file,
            "host_root": str(host_root),
            "submodule": str(submodule),
            "downstream_env": self.downstream_env,
            "built": built,
            "changes_vs_lock": changes,
        }

    def _output_dir(self, host_root: Path, submodule: Path) -> Path:
        """Where this run's outputs land (also where ``downstream_env.json`` goes)."""
        if self.task == "encoding":
            default = host_root / "results" / "downstream" / "encoding"
        elif self.mode == "single-run":
            default = host_root / "results" / "downstream"
        elif self.mode == "calibrate":
            default = host_root / "results" / "downstream" / "calibration"
        else:  # benchmark: the engine owns the location; job.out is ignored
            return submodule / "experiments_benchmark" / self._benchmark_tag()
        return Path(self._abspath(self.out) or default)

    # ── lifecycle ────────────────────────────────────────────────────────
    def run(self) -> Locals:
        host_root = self._host_root()
        submodule = (host_root / self.submodule_dir).resolve()

        self._preflight(submodule, host_root)
        # Build the command first: its config validation (selections, recipe
        # paths) must fail before a possibly multi-minute env build.
        if self.task == "encoding":
            cmd, env, out_desc = self._build_encoding(host_root, submodule)
        else:
            cmd, env, out_desc = self._build_decoding(host_root, submodule)
        env_record = self._ensure_env(submodule, host_root)

        printable = shlex.join(cmd)
        if self.dry_run:
            # The launcher sets exactly one env var on the child; everything
            # else is passed as a CLI flag. Showing only it avoids echoing the
            # inherited env / .env secrets (MLFLOW_TRACKING_*) merged in below.
            shown = {"NTT_SPLITS_DIR": env["NTT_SPLITS_DIR"]}
            log.info("[dry-run] cwd=%s", submodule)
            log.info("[dry-run] launcher-set env: %s", shown)
            log.info("[dry-run] outputs -> %s", out_desc)
            log.info("[dry-run] command:\n%s", printable)
            return locals()

        record_path = self._output_dir(host_root, submodule) / _ENV_RECORD
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(env_record, indent=2) + "\n")
        log.info("Downstream env record -> %s", record_path)
        log.info("Launching downstream eval (cwd=%s):\n%s", submodule, printable)
        subprocess.run(cmd, cwd=submodule, env=env, check=True)  # noqa: S603
        log.info("Downstream eval finished. Outputs -> %s", out_desc)
        return locals()

    # ── preflight ────────────────────────────────────────────────────────
    def _preflight(self, submodule: Path, host_root: Path) -> None:
        problems: list[str] = []
        if not (submodule / "pyproject.toml").is_file():
            problems.append(
                f"submodule not initialised at {submodule} — run:\n"
                f"    git submodule update --init {self.submodule_dir}"
            )
        elif self.downstream_env == "verify" and not (submodule / ".venv").is_dir():
            # (auto builds a missing env in _ensure_env.)
            problems.append(
                "submodule env not built and job.downstream_env=verify — build it once "
                "with job.downstream_env=auto, or run:\n" + self._build_hint(submodule, host_root)
            )
        if problems:
            msg = "\n".join(problems)
            if self.dry_run:
                log.warning("[dry-run] preflight warnings:\n%s", msg)
            else:
                raise RuntimeError(msg)

        if self.task == "decoding":
            if self.mode != "benchmark" and self.ckpt is None:
                raise ValueError("decoding requires job.ckpt (a frozen tokenizer .ckpt).")
            if self.mode == "single-run" and not self.session:
                raise ValueError("decoding/single-run requires job.session.")
            if self.mode == "benchmark":
                if self.ckpt is None and self.ckpt_root is None:
                    raise ValueError(
                        "decoding/benchmark requires job.ckpt or job.ckpt_root (a directory "
                        "of <name>/checkpoints/model.ckpt)."
                    )
                if (
                    self.ckpt is not None
                    and _is_uri(self.ckpt)
                    and not self.ckpt.startswith("hf://")
                ):
                    raise ValueError(
                        f"decoding/benchmark accepts job.ckpt only as an hf:// URI or a local "
                        f".ckpt path (got {self.ckpt!r}); download other URIs into a "
                        "job.ckpt_root layout."
                    )
        else:  # encoding
            if self.ckpt is None:
                raise ValueError("encoding requires job.ckpt (a frozen tokenizer .ckpt).")

    # ── decoding ─────────────────────────────────────────────────────────
    def _build_decoding(
        self, host_root: Path, submodule: Path
    ) -> tuple[list[str], dict[str, str], str]:
        data_root = self._abspath(self.data_root) or str(submodule / "data" / "processed")
        ckpt = self._ckpt_path() if self.mode != "benchmark" or self.ckpt else None

        if self.mode == "single-run":
            if self.mlflow is not None:
                raise ValueError(
                    "job.mlflow applies to mode=calibrate and mode=benchmark; single-run logs "
                    "to MLflow only with job.extra='[\"mlflow.enabled=true\"]'"
                )
            out = str(self._output_dir(host_root, submodule))
            env = self._clean_env(host_root, source_dotenv=submodule / ".env")
            # Promoted flags first, then raw Hydra passthrough (the launcher appends
            # positionals last; Hydra last-wins, so job.extra can override anything).
            # Arch defaults (model / unit_embedding_mode / sentinels) are owned by the
            # contract config (configs/cli/decode.yaml) — only explicit user overrides
            # are forwarded.
            cmd = [
                *self._cli(submodule, "decode"),
                "--tokenizer-kind",
                "frozen",
                "--tokenizer-ckpt",
                str(ckpt),
                "--data-root",
                data_root,
                "--output-root",
                out,
            ]
            if self.seed is not None:
                cmd += ["--seed", str(self.seed)]
            if self.dry_run:
                cmd.append("--dry-run")
            cmd.append(f"session={self.session}")
            for key, val in (
                ("epochs", self.epochs),
                ("batch_size", self.batch_size),
                ("num_workers", self.num_workers),
            ):
                if val is not None:
                    cmd.append(f"{key}={val}")
            if self.model_cfg is not None:
                cmd.append(f"model={self.model_cfg}")
            if self.unit_embedding_mode is not None:
                cmd.append(f"tokenizer.unit_embedding_mode={self.unit_embedding_mode}")
            if self.sentinels is not None:
                cmd.append(f"tokenizer.sentinels={'true' if self.sentinels else 'false'}")
            cmd.append("wandb.enabled=false")
            cmd += self.extra
            return cmd, env, out

        # calibrate / benchmark run in the submodule's frozen venv (no bash; no
        # env-var contract on a mutable script). The submodule .env is sourced so
        # MLflow creds reach the child. Bucket B cohorts are flattened from this
        # repo's selections so they win over the submodule's defaults. Both use the
        # versioned ``ntt-downstream-{calibrate,benchmark}`` console scripts: ntt
        # writes those cohorts (plus non-promoted engine knobs) into a sweep-config
        # YAML and points the CLI at it via ``--config-dir``.
        env = self._clean_env(host_root, source_dotenv=submodule / ".env")

        if self.mode == "calibrate":
            out = str(self._output_dir(host_root, submodule))
            sweep: dict[str, object] = {
                # Bucket B: this repo's selections win over the submodule's defaults.
                "train_sessions": self._selection_sessions(
                    host_root, self.train_selection or self._DEFAULT_SELECTIONS["train"]
                ),
                "other_sessions": self._selection_sessions(
                    host_root, self.eval_selection or self._DEFAULT_SELECTIONS["calib_eval"]
                ),
            }
            for key, val in (
                ("gpus", self.gpus),
                ("batch_size", self.batch_size),
                ("num_workers", self.num_workers),
                ("model_cfg", self.model_cfg),
                ("unit_embedding_mode", self.unit_embedding_mode),
                ("sentinels", self.sentinels),
                ("mlflow", self.mlflow),
            ):
                if val is not None:
                    sweep[key] = val
            cfg_dir = Path(out)
            # Keyed by the ``out`` dir — parallel calibrate sweeps need distinct
            # ``out`` dirs or they clobber each other's sweep YAML.
            self._write_sweep_config(cfg_dir / "ntt_calibrate.yaml", sweep)
            cmd = [
                *self._cli(submodule, "calibrate"),
                "--tokenizer-kind",
                "frozen",
                "--tokenizer-ckpt",
                str(ckpt),
                "--data-root",
                data_root,
                "--output-root",
                out,
                "--config-dir",
                str(cfg_dir),
                "--config-name",
                "ntt_calibrate",
            ]
            if self.seed is not None:
                cmd += ["--seed", str(self.seed)]
            if self.dry_run:
                cmd.append("--dry-run")
            cmd += self.extra  # positional passthrough -> engine per-cell extra
            if self.epochs is not None:
                log.warning(
                    "mode=calibrate is step-driven; job.epochs is ignored — use "
                    "job.extra (e.g. ft_max_steps=…) for per-cell overrides."
                )
            out_desc = f"{out} (fine-tune/calibration CSVs); base-train under {submodule}/outputs/"
            return cmd, env, out_desc

        # benchmark
        if self.out:
            log.warning(
                "job.out is ignored for mode=benchmark — the engine writes under "
                "the submodule's experiments_benchmark/<tag>/."
            )
        ckpt_root = self._benchmark_ckpt_root(host_root, ckpt)
        comparison_tag = self._benchmark_tag()
        sweep: dict[str, object] = {
            "ckpt_root": ckpt_root,
            "comparison_tag": comparison_tag,
            "sessions": self._selection_sessions(
                host_root, self.eval_selection or self._DEFAULT_SELECTIONS["benchmark_eval"]
            ),
        }
        for key, val in (
            ("gpus", self.gpus),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("num_workers", self.num_workers),
            ("model_cfg", self.model_cfg),
            ("unit_embedding_mode", self.unit_embedding_mode),
            ("sentinels", self.sentinels),
            ("mlflow", self.mlflow),
        ):
            if val is not None:
                sweep[key] = val
        cfg_dir = host_root / "results" / "downstream" / "sweep_configs"
        self._write_sweep_config(cfg_dir / f"{comparison_tag}.yaml", sweep)
        cmd = [
            *self._cli(submodule, "benchmark"),
            "--data-root",
            data_root,
            "--config-dir",
            str(cfg_dir),
            "--config-name",
            comparison_tag,
        ]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        if self.dry_run:
            cmd.append("--dry-run")
        cmd += self.extra
        out_desc = f"{submodule}/experiments_benchmark/{comparison_tag}/"
        return cmd, env, out_desc

    def _single_ckpt_name(self) -> tuple[str, str]:
        """``(name, identity)`` of ``job.ckpt``: the benchmark row name and a unique key.

        ``name`` is the Hub model name (or the local file's stem); ``identity`` is
        ``repo/name@revision`` for ``hf://``, else the resolved local path.
        """
        assert self.ckpt is not None
        if self.ckpt.startswith("hf://"):
            return _hub_ckpt_name(self.ckpt)
        local = Path(self._abspath(self.ckpt) or "")
        return local.stem, str(local)

    def _single_ckpt_root(self, host_root: Path) -> Path:
        """The ``ckpt_root`` ntt builds for a single ``job.ckpt`` in benchmark mode.

        ``results/downstream/benchmark_roots/<name>-<hash>``, ``<hash>`` being 8 hex of
        sha1 over the identity, so different repos/revisions never share a root (hence
        a tag). A pure function of the job config, so it is also valid under dry_run.
        """
        root_name = _ckpt_root_name(*self._single_ckpt_name())
        return host_root / "results" / "downstream" / "benchmark_roots" / root_name

    def _benchmark_ckpt_root(self, host_root: Path, ckpt: str | None) -> str:
        """``job.ckpt_root``, or the single-tokenizer root built from ``job.ckpt``.

        ``ckpt`` is the local checkpoint (materialized unless dry_run, when it is still
        the ``hf://`` URI and no layout is created).
        """
        if self.ckpt is None:
            return str(self._abspath(self.ckpt_root))
        root = self._single_ckpt_root(host_root)
        if not self.dry_run:
            link = root / self._single_ckpt_name()[0] / "checkpoints" / "model.ckpt"
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(Path(str(ckpt)).resolve())
        return str(root)

    def _benchmark_tag(self) -> str:
        """Deterministic COMPARISON_TAG so the printed output path is exact.

        Derived from the ``ckpt_root`` BASENAME, so two roots that share a
        basename collide on *both* the sweep-config path and the engine's
        ``experiments_benchmark/<tag>/`` output dir — give such roots distinct
        directory names. With ``job.ckpt`` the basename is ``<name>-<hash>`` (see
        ``_single_ckpt_root``), unique per tokenizer and revision.
        """
        if self.ckpt is not None:
            return f"ntt_downstream_{self._single_ckpt_root(self._host_root()).name}"
        return f"ntt_downstream_{Path(str(self.ckpt_root)).name}"

    # ── encoding ─────────────────────────────────────────────────────────
    def _resolve_recipe_path(self, host_root: Path) -> Path:
        """Resolve a path-form ``recipe`` to an absolute, existing config file.

        Absolute paths are used as-is; RELATIVE paths resolve against
        ``host_root`` (the ntt repo root, NOT cwd — ntt's Hydra run dir makes
        cwd unreliable, and the flagship use-case is configs shipped in ntt's
        ``config/`` folder). This is ntt-side validation of an ntt-owned path
        (not a re-mirror of submodule internals), so it runs under dry_run too.
        """
        p = Path(self.recipe).expanduser()
        resolved = (p if p.is_absolute() else host_root / p).resolve()
        if not resolved.is_file():
            raise ValueError(
                f"path-form job.recipe does not exist: {resolved}. Relative recipe "
                f"paths resolve against the ntt repo root ({host_root}), not the cwd "
                "— pass an absolute path or one relative to the repo root."
            )
        return resolved

    @staticmethod
    def _recipe_is_path_form(recipe: str) -> bool:
        """The path-vs-bare-name rule for ``recipe`` (documented in the field comment)."""
        return "/" in recipe or recipe.endswith((".yaml", ".yml"))

    def _build_encoding(
        self, host_root: Path, submodule: Path
    ) -> tuple[list[str], dict[str, str], str]:
        ckpt = self._ckpt_path()
        out = str(self._output_dir(host_root, submodule))
        loss_mode = self.loss_mode or "mse"
        env = self._clean_env(host_root, source_dotenv=submodule / ".env")
        # The encoding script needs a tracking URI only if it actually logs to
        # MLflow — a decision its own logger setup owns, and one reachable via
        # job.extra (trainer.mlflow_tracking_uri / trainer.logging=false). Rather
        # than mirror that branch logic here (it lived in another, separately
        # versioned repo and rotted on every submodule bump), warn and let the
        # child be the authority: it either runs fine or fails with its own
        # message. The warning surfaces the likely cause before the heavy boot.
        if not self.dry_run and not env.get("MLFLOW_TRACKING_URI"):
            log.warning(
                "MLFLOW_TRACKING_URI is not set (env or %s/.env). The encoding run will "
                "fail if it logs to MLflow without one — unless you pass "
                "trainer.mlflow_tracking_uri=... or trainer.logging=false via job.extra. "
                "Proceeding; the run will surface the authoritative error if it needs it.",
                submodule,
            )
        cmd = [
            *self._cli(submodule, "encode"),
            "--tokenizer-ckpt",
            str(ckpt),  # launcher maps → checkpoint_path + enabled=true
            "--output-root",
            out,  # launcher maps → hydra.run.dir
        ]
        if self.data_root is not None:
            cmd += ["--data-root", str(self._abspath(self.data_root))]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        # A recipe that looks like a PATH (contains "/" or a .yaml/.yml suffix)
        # points at an external Hydra config OUTSIDE the submodule: emit both
        # --config-dir <abs parent> and --config-name <stem>. Hydra ADDS that
        # dir to the search path (the submodule's configs/encoding/ stays on it),
        # so the external file can be a thin overlay composing the internal tree.
        # Passing --config-dir drops the curated configs/cli/ off the search
        # path, so --config-name MUST also be passed (the launcher default name
        # 'encode' would not resolve) — hence both flags, always.
        if self._recipe_is_path_form(self.recipe):
            recipe_path = self._resolve_recipe_path(host_root)
            cmd += [
                "--config-dir",
                str(recipe_path.parent),
                "--config-name",
                recipe_path.stem,
            ]
        elif self.recipe != "cowley_v4_base":
            # Non-default bare recipes resolve from the app's own configs/encoding/
            # search path; the default goes through the curated configs/cli/encode.yaml.
            cmd += ["--config-name", self.recipe]
        if self.dry_run:
            cmd.append("--dry-run")
        cmd += [
            # Not owned by the launcher's promoted mapping (verified against
            # cli/overrides.py @ 6da1028): force checkpoint-first load order and the
            # frozen trainer mode, so the frozen tokenizer is actually consumed.
            f"model.loss_mode={loss_mode}",
            "++tokenizer.checkpoint_first=true",
            "trainer.tokenizer_mode=frozen",
            *self.extra,
        ]
        return cmd, env, out

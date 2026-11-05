"""Base class for NTT jobs that work with tokenizers and data."""

import typing as T
from pathlib import Path

import lightning as L
import pydantic as pdt

from ntt.contrib import TokenizerSpecKind
from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig
from ntt.jobs.base import BaseJob
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NTTBaseJob(
    BaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Base class for NTT jobs that use a tokenizer and datamodule.

    Provides shared config (tokenizer, data, seed, checkpoint path)
    and handles seeding + datamodule creation in ``__enter__``.
    """

    seed: int | None = None
    ckpt_path: str | Path | None = None
    weights_only: bool = True
    tokenizer_config: TokenizerSpecKind
    data_config: NTTDataModuleConfig

    # Eval jobs score at ``eval_config.eval_window_seconds``, not at the spec's window.
    applies_eval_window: T.ClassVar[bool] = False

    _datamodule: NTTDataModule | None = pdt.PrivateAttr(default=None)
    # The window the evaluation datamodule actually loads (see ``_check_scored_window``).
    _scored_window_seconds: float | None = pdt.PrivateAttr(default=None)

    def __enter__(self) -> T.Self:
        """Enter job context — seeds RNG and creates datamodule."""
        if self.seed is not None:
            log.info(f"Setting random seed ({self.seed})")
            L.seed_everything(self.seed, workers=True)

        log.info("Instantiating datamodule...")
        self._datamodule = NTTDataModule(self.data_config, self.tokenizer_config)
        if self.applies_eval_window:
            from ntt.evaluation.utils.window_contract import warn_if_off_native_window

            # NTTDataModule(config, spec) overwrote the window with the spec's: put the eval
            # window back, as the all-neuron path does.
            window = self._resolve_eval_window_seconds()
            self.data_config.window_length_seconds = window
            self._check_scored_window(self._datamodule, type(self).__name__)
            warn_if_off_native_window(self.tokenizer_config, window)

        return super().__enter__()

    def _resolve_eval_window_seconds(self) -> float:
        """Authoritative eval window (seconds). Eval jobs carry an eval_config;
        non-eval jobs fall back to the 1 s default. ``"native"`` resolves to the tokenizer
        spec's own window."""
        from ntt.evaluation.utils.window_contract import resolve_eval_window_seconds

        return resolve_eval_window_seconds(
            self._requested_eval_window(), getattr(self, "tokenizer_config", None)
        )

    def _requested_eval_window(self) -> float | str:
        cfg = getattr(self, "_eval_config", None)
        if cfg is not None:
            return getattr(cfg, "eval_window_seconds", 1.0)
        raw = getattr(self, "eval_config", None) or {}
        return raw.get("eval_window_seconds", 1.0) if isinstance(raw, dict) else 1.0

    def _check_scored_window(self, datamodule: NTTDataModule, what: str) -> None:
        """Fail loud if ``datamodule`` loads another window than the eval window, and keep
        the window it loads as the one recorded."""
        from ntt.evaluation.utils.window_contract import check_datamodule_window

        self._scored_window_seconds = check_datamodule_window(
            datamodule, self._resolve_eval_window_seconds(), what=what
        )

    def _eval_window_record(self) -> dict[str, T.Any]:
        """The window scored (as loaded by the datamodule), as requested, and the spec's own
        (for the saved run metadata)."""
        from ntt.evaluation.utils.window_contract import eval_window_record

        scored = self._scored_window_seconds
        return eval_window_record(
            self.tokenizer_config,
            self._requested_eval_window(),
            self._resolve_eval_window_seconds() if scored is None else scored,
        )

    @property
    def datamodule(self) -> NTTDataModule:
        """Get the datamodule instance."""
        if self._datamodule is None:
            raise RuntimeError(
                "DataModule not instantiated. Use the job within a context manager."
            )
        return self._datamodule

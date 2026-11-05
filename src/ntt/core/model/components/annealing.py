"""General parameter-annealing schedule and Lightning callback.

Anneals a scalar attribute on a tokenizer submodule over training following
a configurable schedule. :class:`AnnealingCallback` is a base class: it carries
the schedule math but is target-agnostic. Concrete subclasses — defined next to
the module they drive (e.g. ``VAEBetaAnnealingCallback`` beside ``VAEBottleneck``,
``BSQZetaAnnealingCallback`` beside ``BSQBottleneck``) — set the ``target_class``
and ``target_param`` class attributes. The callback then locates every submodule
of ``pl_module.tokenizer`` that is an instance of ``target_class`` and anneals its
``target_param`` attribute (or a single named submodule via ``target_name``).
Configs target the subclasses, which sidesteps having to name an attribute that
may be ambiguous across submodules.

Schedule math follows Fu et al. (2019), "Cyclical Annealing Schedule: A Simple
Approach to Mitigating KL Vanishing" (NAACL 2019), supporting linear, cosine,
and sigmoid schedules with cyclical or monotonic annealing.
"""

from __future__ import annotations

import abc
import math
from typing import Any, ClassVar, Literal, cast

import pytorch_lightning as pl
from torch import nn


def compute_annealed_value(
    global_step: int,
    max_steps: int,
    schedule: str,
    value_min: float,
    value_max: float,
    n_cycles: int,
    ratio: float,
    delay_steps: int,
) -> float:
    """Compute the annealed value for a given training step.

    ``value_min`` is the start value and ``value_max`` the end value; the ramp
    runs from start to end, so ``value_min > value_max`` ramps DOWN (the math
    handles either direction).

    Args:
        global_step: Current training step.
        max_steps: Total number of training steps.
        schedule: One of "linear", "cosine", "sigmoid".
        value_min: Start value (held during the delay and at cycle start).
        value_max: End value (held once the annealing phase completes).
        n_cycles: Number of annealing cycles (M).
        ratio: Proportion of each cycle spent annealing (R).
        delay_steps: Steps before annealing begins.

    Returns:
        The annealed value for this step.
    """
    if global_step < delay_steps:
        return value_min

    t_eff = global_step - delay_steps
    effective_steps = max_steps - delay_steps

    if effective_steps <= 0 or ratio <= 0:
        return value_max

    period = effective_steps / n_cycles
    tau = (t_eff % period) / period

    if tau >= ratio:
        return value_max

    p = tau / ratio  # progress within annealing phase, in [0, 1]

    if schedule == "linear":
        frac = p
    elif schedule == "cosine":
        frac = 0.5 - 0.5 * math.cos(math.pi * p)
    elif schedule == "sigmoid":
        frac = 1.0 / (1.0 + math.exp(-(12.0 * p - 6.0)))
    else:
        msg = f"Unknown schedule '{schedule}'. Must be 'linear', 'cosine', or 'sigmoid'."
        raise ValueError(msg)

    return value_min + (value_max - value_min) * frac


def find_target_submodules(
    root: nn.Module,
    target_class: type[nn.Module],
    target_name: str | list[str] | None = None,
) -> list[tuple[str, nn.Module]]:
    """Locate ``(name, module)`` submodules of ``root`` matching ``target_class``.

    Unwraps a ``torch.compile`` wrapper on ``root`` if present. With
    ``target_name`` None, returns every matching submodule in ``named_modules``
    order. Otherwise ``target_name`` is a single name or a list of names and the
    matching submodules are returned in the requested order.

    Raises:
        RuntimeError: if no submodule matches, or if any requested name is absent
            (or names a submodule that is not a ``target_class`` instance).
    """
    inner = getattr(root, "_orig_mod", root)  # unwrap torch.compile
    assert isinstance(inner, nn.Module)

    matches = [
        (name, module)
        for name, module in inner.named_modules()
        if isinstance(module, target_class)
    ]

    if target_name is not None:
        wanted = [target_name] if isinstance(target_name, str) else list(target_name)
        by_name = dict(matches)
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            msg = f"Found no {target_class.__name__} submodule(s) named {unknown}."
            raise RuntimeError(msg)
        # Preserve the caller's requested order.
        return [(n, by_name[n]) for n in wanted]

    if not matches:
        msg = f"Found no {target_class.__name__} submodule in the module tree."
        raise RuntimeError(msg)
    return matches


class AnnealingCallback(pl.Callback, abc.ABC):
    """Base callback that anneals a scalar attribute on tokenizer submodule(s).

    This class is abstract with respect to *what* it drives. Concrete subclasses
    set two class attributes:

    - ``target_class``: the ``nn.Module`` subclass to locate within the tokenizer.
    - ``target_param``: the name of the scalar attribute to anneal on it.

    ``__init_subclass__`` enforces both at import time, so a subclass that omits
    either fails to even define. On train start the callback locates the matching
    submodule(s) of ``pl_module.tokenizer`` and updates each one's ``target_param``
    attribute every batch according to the schedule. See
    :func:`compute_annealed_value` for the math.

    By default every submodule that is an instance of ``target_class`` is annealed
    in lockstep. Pass ``target_name`` to restrict to the single submodule with that
    exact qualified name (as reported by ``nn.Module.named_modules``).

    Subclasses live next to the module they target (e.g. ``VAEBetaAnnealingCallback``
    beside ``VAEBottleneck``) and configs reference the subclass, so the schedule
    knobs below are the only per-experiment configuration.

    ``value_min`` is the start value and ``value_max`` the end value, so
    ``value_min > value_max`` ramps DOWN (e.g. the BSQ-zeta case: push codebook
    spread early, refine reconstruction late).

    Class attributes (set by subclasses):
        target_class: ``nn.Module`` subclass to locate within the tokenizer.
        target_param: Name of the scalar attribute to anneal.
        raw_attr: Optional attribute on the located submodule holding a raw
            diagnostic value (e.g. ``"_last_kl"``) to log each batch end. No
            raw logging if None.
        raw_log_key: Metric key for ``raw_attr``. Defaults to
            ``f"train/{raw_attr}"`` when ``raw_attr`` is set and this is None.
            When more than one submodule matches, each module's qualified name is
            appended (``f"{raw_log_key}/{name}"``) to keep the keys distinct.

    Args:
        schedule: "linear", "cosine", or "sigmoid".
        value_max: End value of the ramp.
        value_min: Start value of the ramp.
        n_cycles: Number of annealing cycles (1 = monotonic).
        ratio: Proportion of each cycle spent annealing.
        delay_steps: Steps before annealing begins.
        log_key: Metric key to log the annealed value under. Defaults to
            ``f"train/{target_param}"``.
        target_name: Optional exact qualified name(s) of the submodule(s) to
            target — a single name or a list of names (as reported by
            ``nn.Module.named_modules``). If None, every submodule matching
            ``target_class`` is annealed. Naming a module that does not exist (or
            is not a ``target_class`` instance) raises at train start.
    """

    # Set by subclasses to target a specific bottleneck and parameter.
    target_class: ClassVar[type[nn.Module] | None] = None
    target_param: ClassVar[str | None] = None
    # Optional raw diagnostic attribute on the located module (e.g. "_last_kl").
    raw_attr: ClassVar[str | None] = None
    # Not a ClassVar: subclasses may set it, but __init__ resolves a default and
    # stores the effective value on the instance.
    raw_log_key: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce the (target_class, target_param) contract at import time."""
        super().__init_subclass__(**kwargs)
        if cls.target_class is None or cls.target_param is None:
            msg = (
                f"{cls.__name__} must set the 'target_class' and 'target_param' class "
                "attributes (e.g. `target_class = VAEBottleneck; target_param = 'beta'`)."
            )
            raise TypeError(msg)

    def __init__(
        self,
        schedule: Literal["linear", "cosine", "sigmoid"] = "linear",
        value_max: float = 1.0,
        value_min: float = 0.0,
        n_cycles: int = 1,
        ratio: float = 0.5,
        delay_steps: int = 0,
        log_key: str | None = None,
        target_name: str | list[str] | None = None,
    ):
        super().__init__()
        # Subclasses are guaranteed valid by __init_subclass__; this guards the
        # base class being instantiated directly (e.g. a misconfigured _target_).
        if self.target_class is None or self.target_param is None:
            msg = (
                "AnnealingCallback is abstract; instantiate a concrete subclass "
                "(e.g. VAEBetaAnnealingCallback) that sets target_class/target_param."
            )
            raise TypeError(msg)

        self.schedule = schedule
        self.value_max = value_max
        self.value_min = value_min
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.delay_steps = delay_steps
        self.target_name = target_name
        self.log_key = log_key if log_key is not None else f"train/{self.target_param}"
        if self.raw_log_key is None and self.raw_attr is not None:
            self.raw_log_key = f"train/{self.raw_attr}"

        self._targets: list[tuple[str, nn.Module]] = []
        self._max_steps: int = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.max_steps is not None and trainer.max_steps > 0:
            self._max_steps = trainer.max_steps
        else:
            # float('inf') when the trainer has no step limit; kept as-is
            self._max_steps = cast(int, trainer.estimated_stepping_batches)

        assert self.target_class is not None  # guaranteed by __init__
        assert self.target_param is not None
        matches = find_target_submodules(
            cast(nn.Module, pl_module.tokenizer), self.target_class, self.target_name
        )

        # The target params (e.g. `beta`) are instance attributes set in the
        # module's __init__, so this is the earliest point we can verify they
        # exist — do it once here rather than on every batch.
        missing = [name for name, m in matches if not hasattr(m, self.target_param)]
        if missing:
            msg = (
                f"{self.target_class.__name__} submodule(s) {missing} have no "
                f"'{self.target_param}' attribute to anneal."
            )
            raise RuntimeError(msg)

        self._targets = matches

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert self._targets, "on_train_start must run before on_train_batch_start"
        value = compute_annealed_value(
            global_step=trainer.global_step,
            max_steps=self._max_steps,
            schedule=self.schedule,
            value_min=self.value_min,
            value_max=self.value_max,
            n_cycles=self.n_cycles,
            ratio=self.ratio,
            delay_steps=self.delay_steps,
        )
        assert self.target_param is not None  # guaranteed by __init__
        for _, module in self._targets:
            setattr(module, self.target_param, value)
        pl_module.log(self.log_key, value)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.raw_attr is None:
            return
        assert self.raw_log_key is not None  # set in __init__ when raw_attr is set
        multiple = len(self._targets) > 1
        for name, module in self._targets:
            raw = getattr(module, self.raw_attr, None)
            if raw is None:
                continue
            key = f"{self.raw_log_key}/{name}" if multiple else self.raw_log_key
            pl_module.log(key, raw)

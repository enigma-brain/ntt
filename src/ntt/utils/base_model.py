from typing import Any

from pydantic import BaseModel, Field

from ntt.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def loggable_flat_dump(model: BaseModel, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Dump a spec/config to nested plain containers for experiment loggers.

    Lightning's ``MLFlowLogger.log_hyperparams`` truncates every param value to
    250 characters, and the ``_flatten_dict`` it calls first does not descend
    into Pydantic models. Handing a spec straight to ``save_hyperparameters``
    therefore logs one clipped ``str(spec)`` instead of one param per knob.
    Returning nested dicts lets ``_flatten_dict`` expand every field into its
    own ``spec/field`` param, each well under the limit.

    ``serialize_as_any=True`` is required: without it Pydantic serialises
    polymorphic fields (e.g. ``int | Dist``) through the *declared* base schema
    and silently drops subclass fields such as ``UniformDist.low`` / ``.high``
    — the same pitfall documented on :class:`NTTBaseModel`.
    """
    try:
        return model.model_dump(
            mode="json",
            exclude=exclude,
            serialize_as_any=True,
            warnings=False,
        )
    except Exception as exc:  # pragma: no cover - never fail a run over logging
        log.warning("Could not dump %s for hyperparameter logging: %s", type(model).__name__, exc)
        return {"repr": str(model)[:250]}


# TODO: Figure out how to make pylance recognize that subclasses will also be frozen/unfrozen
class NTTBaseModel(BaseModel, strict=False, frozen=True, extra="forbid"):  # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
    """Base Pydantic model with Hydra _target_ compatibility.

    Allows YAML configs to work with both Hydra instantiation and Pydantic validation.
    Subclasses can still configure model options (strict, extra, etc.) as normal,
    but frozen cannot be overridden (subclasses remain frozen).

    Provides automatic checkpoint migration: when a checkpoint is
    unpickled and lacks fields the current schema defines, ``__setstate__``
    detects the gap, re-validates through ``model_validate`` (which runs all
    ``@model_validator`` hooks to compute derived defaults), and restores a
    fully up-to-date instance.

    Pickling uses pydantic's default ``__newobj__`` + ``__setstate__`` path
    (i.e. no custom ``__reduce__``). That path preserves concrete subclass
    identity for nested polymorphic fields — ``model_dump``-based reducers
    collapse a ``float | Dist`` field annotated by its base class down to
    the base schema, silently losing subclass-specific data.
    """

    hydra_target: str | None = Field(
        default=None,
        alias="_target_",
        exclude=True,
        description="Hydra instantiation target (ignored). "
        "Allows YAML configs to work with both Hydra and Pydantic.",
    )

    def __setstate__(self, state: dict) -> None:
        """Migrate pickled state from checkpoints that lack current fields on unpickle.

        If any field defined in the current schema is missing from the
        pickled ``__dict__``, re-validate through ``model_validate`` so
        that Pydantic validators (including ``@model_validator``) can
        compute derived defaults. No-op when all fields are present.
        """
        stored = state.get("__dict__", {})
        cls = type(self)
        current_fields = cls.model_fields
        missing = [f for f in current_fields if f not in stored]

        if not missing:
            super().__setstate__(state)
            return

        for field_name in missing:
            log.warning(
                f"Checkpoint spec {cls.__name__} is missing field "
                f"'{field_name}' — will re-validate to fill defaults",
            )

        # Keep only fields the current schema knows about (drops removed
        # fields that would trip extra="forbid"), then re-validate so
        # model_validator hooks can compute derived values.
        data = {k: v for k, v in stored.items() if k in current_fields}
        migrated = cls.model_validate(data, by_name=True)
        state["__dict__"] = migrated.__dict__
        state["__pydantic_fields_set__"] = migrated.__pydantic_fields_set__
        super().__setstate__(state)


class NTTMutableBaseModel(BaseModel, strict=False, frozen=False, extra="forbid"):  # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
    """Mutable variant of NTTBaseModel for classes that need to be modified after creation.

    Use this for classes like jobs that may need mutable state.
    """

    hydra_target: str | None = Field(
        default=None,
        alias="_target_",
        exclude=True,
        description="Hydra instantiation target (ignored). "
        "Allows YAML configs to work with both Hydra and Pydantic.",
    )

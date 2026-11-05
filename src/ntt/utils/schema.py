from typing import Any, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pydantic import BaseModel
from pydantic.fields import FieldInfo

T = TypeVar("T")


def fields_by_json_schema_extra(model: BaseModel, extra_flag: str) -> dict[str, Any]:
    """Utility to get Pydantic model fields filtered by a specific flag key in their
      `json_schema_extra`, where the flag value is `True`.

    :param model: The Pydantic BaseModel to inspect.
    :param extra_flag: The flag key in `json_schema_extra` to filter fields by.
    :return: A dictionary of attribute values from fields that contain the specified extra flag.
    """
    field_infos: dict[str, FieldInfo] = model.__class__.model_fields

    def _field_filter(fi: FieldInfo) -> bool:
        jse: dict | None = getattr(fi, "json_schema_extra", None)
        if jse is None:
            return False
        flag = jse.get(extra_flag, False)
        return flag is True

    return {fn: getattr(model, fn) for fn, fi in field_infos.items() if _field_filter(fi)}


def dict_config_to_list(**cfg: DictConfig) -> list[Any]:
    """
    Utility to convert a DictConfig to a list of it's values,
    Inteded to support nested instantiation of a `Trainer` object,
    which takes lists of callbacks, loggers, etc.

    TODO: Is this too hacky?
    """
    return [v for v in cfg.values() if v is not None]


def register_config(group: str | None = None, name: str | None = None):
    """Decorator that registers a dataclass or Pydantic model to Hydra's ConfigStore.

    The ConfigStore is a singleton, so all registrations are available globally
    when Hydra's @hydra.main decorator runs.

    :param group: Config group (e.g., "job", "model", "reader"). Maps to config subdirectory.
    :param name: Config name within the group. Defaults to the class name if not provided.

    Example:
        @register_config(group="job", name="training")
        class TrainingJob(BaseJob):
            ...

        # Or with automatic name:
        @register_config(group="model")
        class BaselineModel:  # Registered as "BaselineModel"
            ...

    NOTE: See `issue #1982 <https://github.com/facebookresearch/hydra/issues/1982>`_
          for related discussion.

    TODO: Pydantic BaseModel is not currently supported; use pydantic.dataclasses.dataclass
          instead or wait for OmegaConf support. See `issue #2667
          <https://github.com/facebookresearch/hydra/issues/2667>`_ for details.
    """

    def decorator(cls: type[T]) -> type[T]:
        cs = ConfigStore.instance()
        config_name = name if name is not None else cls.__name__
        cs.store(group=group, name=config_name, node=cls)
        return cls

    return decorator

"""Base for high-level project jobs."""

import abc
import types as TS
import typing as T

from omegaconf import DictConfig

from ntt.utils import NTTMutableBaseModel, RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# Local job variables
Locals = dict[str, T.Any]


class BaseJob(abc.ABC, NTTMutableBaseModel, strict=True):
    """Base class for a job.

    use a job to execute runs in  context.
    e.g., to define common services like logger

    """

    KIND: str

    def process_config(self, cfg: DictConfig | None = None) -> None:
        """Process the full configuration.

        This method is called before entering the job context and can be
        overridden by subclasses to handle config-related tasks like
        uploading to experiment tracking services, etc.

        Note: This method should not modify the job (which is frozen),
        only observe and process the config.

        Args:
            cfg (DictConfig | None): Optional full configuration object.
        """
        pass

    def __enter__(self) -> T.Self:
        """Enter the job context.

        Returns:
            T.Self: return the current object.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        """Exit the job context.

        Args:
            exc_type (T.Type[BaseException] | None): ignored.
            exc_value (BaseException | None): ignored.
            exc_traceback (TS.TracebackType | None): ignored.

        Returns:
            T.Literal[False]: always propagate exceptions.
        """
        return False  # re-raise

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """

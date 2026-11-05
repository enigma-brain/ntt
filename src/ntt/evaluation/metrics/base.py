"""Base class and registry for metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ..core.types import (
    DataType,
    EvalData,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)

if TYPE_CHECKING:
    from ..core.intermediates import IntermediatesCache


@dataclass(frozen=True)
class IntermediateRequest:
    """A parameterized request for an intermediate computation.

    Two requests with the same ``key`` and ``params`` share a single
    cache entry.  Different ``params`` produce separate computations.
    """

    key: str
    params: frozenset[tuple[str, Any]] = field(default_factory=frozenset)

    @staticmethod
    def from_dict(key: str, params: dict[str, Any]) -> "IntermediateRequest":
        """Create from a dict, converting non-hashable values to tuples."""
        hashable_items: list[tuple[str, Any]] = []
        for k, v in sorted(params.items()):
            if isinstance(v, list):
                v = tuple(v)
            hashable_items.append((k, v))
        return IntermediateRequest(key, frozenset(hashable_items))

    @property
    def params_dict(self) -> dict[str, Any]:
        """Return params as a plain dict (lists restored from tuples)."""
        return dict(self.params)


# Global registry of metrics
METRIC_REGISTRY: dict[str, type["BaseMetric"]] = {}


def register_metric(cls: type["BaseMetric"]) -> type["BaseMetric"]:
    """Decorator to register a metric class."""
    METRIC_REGISTRY[cls.name] = cls
    return cls


def get_all_metrics() -> list["BaseMetric"]:
    """Instantiate all registered metrics."""
    return [cls() for cls in METRIC_REGISTRY.values()]


def get_metrics_by_category(category: MetricCategory) -> list["BaseMetric"]:
    """Instantiate all registered metrics in a given category."""
    return [cls() for cls in METRIC_REGISTRY.values() if cls.category == category]


def get_metric_names() -> list[str]:
    """Return all registered metric names, sorted for stable display."""
    return sorted(METRIC_REGISTRY.keys())


def get_metrics_by_names(names: list[str]) -> list["BaseMetric"]:
    """
    Instantiate metrics by name.

    Raises:
        ValueError: if any requested metric names are not registered. The
        error message includes the unknown names and the list of available
        metric names for easier debugging.
    """
    metrics: list[BaseMetric] = []
    missing: list[str] = []

    for name in names:
        cls = METRIC_REGISTRY.get(name)
        if cls is None:
            missing.append(name)
        else:
            metrics.append(cls())

    if missing:
        available = ", ".join(get_metric_names())
        unknown = ", ".join(sorted(missing))
        raise ValueError(
            f"Unknown metric name(s) requested: {unknown}. Available metrics are: {available}"
        )

    return metrics


def get_metrics_for_data_type(data_type: DataType) -> list["BaseMetric"]:
    """Get metrics compatible with a given data type."""
    return [cls() for cls in METRIC_REGISTRY.values() if data_type in cls.supported_data_types]


class BaseMetric(ABC):
    """
    Base class for evaluation metrics.

    Subclasses must define class attributes:
        - name: unique identifier for the metric
        - category: which evaluation category this belongs to
        - direction: whether higher or lower is better
        - level: granularity (per_unit, per_pair, population)
        - _required_intermediates: list of intermediate keys needed from cache
            it can override get_required_intermediates(TYPE) if the intermediates
            depend on the data type (SPIKE_TIMES, SPIKES, RATES)
        - supported_data_types: set of DataTypes this metric can work with
        - preferred_data_type: the ideal DataType for this metric
        - _default_params: dict of parameter names to default values.
            Subclasses declare their configurable parameters here.
            Instances receive overrides via __init__(**params).

    Subclasses can define:
        - min_window_size_seconds: minimum window size required to compute the metric
        - max_bin_size_seconds: maximum bin size supported to compute the metric
                                (not checked for spike time data)

    Subclasses must implement:
        - compute(): compute the metric from intermediates
    """

    name: str
    category: MetricCategory
    direction: MetricDirection
    level: MetricLevel
    _required_intermediates: list[str]
    _default_params: ClassVar[dict[str, Any]] = {}
    value_range: tuple[float | None, float | None] = (None, None)
    min_window_length_seconds: float | None = None
    max_bin_size_seconds: float | None = None

    # Data type compatibility
    supported_data_types: set[DataType] = {
        DataType.SPIKE_TIMES,
        DataType.SPIKES,
        DataType.RATES,
    }
    preferred_data_type: DataType = DataType.SPIKE_TIMES

    aggregation_fallback_policy: Literal["session_aggregated", "per_window", "contiguous_only"] = (
        "contiguous_only"
    )

    # Optional dataset filter — set via MetricInstanceConfig, not the
    # metric constructor.  When non-None, the metric is only computed for
    # sessions whose session_id contains at least one of the strings
    # (substring match).  Set by instantiate_metrics_from_config().
    limit_to_datasets: list[str] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        allowed_policies = {"session_aggregated", "per_window", "contiguous_only"}
        has_accumulate = any(
            "_accumulate_window" in base.__dict__
            for base in cls.__mro__
            if base not in {BaseMetric, object}
        )
        has_finalize = any(
            "_finalize_session_state" in base.__dict__
            for base in cls.__mro__
            if base not in {BaseMetric, object}
        )
        policy = cls.__dict__.get("aggregation_fallback_policy", cls.aggregation_fallback_policy)

        if policy not in allowed_policies:
            raise TypeError(
                f"{cls.__name__} sets aggregation_fallback_policy={policy!r}, "
                f"but allowed values are {sorted(allowed_policies)}."
            )

        if policy == "session_aggregated":
            if not has_accumulate:
                raise TypeError(
                    f"{cls.__name__} sets aggregation_fallback_policy='session_aggregated' "
                    "but does not override _accumulate_window()."
                )
            if not has_finalize:
                raise TypeError(
                    f"{cls.__name__} sets aggregation_fallback_policy='session_aggregated' "
                    "but does not override _finalize_session_state()."
                )
            return

        if has_accumulate or has_finalize:
            raise TypeError(
                f"{cls.__name__} overrides _accumulate_window or _finalize_session_state "
                f"but aggregation_fallback_policy is {policy!r}, not 'session_aggregated'."
            )

    def __init__(self, **params: Any):
        unknown = set(params) - set(self._default_params)
        if unknown:
            raise TypeError(
                f"Unknown parameter(s) for {type(self).__name__}: {unknown}. "
                f"Valid parameters: {set(self._default_params) or 'none'}"
            )
        self.params: dict[str, Any] = {**self._default_params, **params}

    def should_run_for_session(self, session_id: str) -> bool:
        """Check whether this metric should run for the given session.


        Returns True if limit_to_datasets is None (no filter) or if session_id
        contains at least one of the limit_to_datasets strings (substring match).
        """
        if self.limit_to_datasets is None:
            return True
        return any(ds in session_id for ds in self.limit_to_datasets)

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return a copy of this metric's default parameters."""
        return dict(cls._default_params)

    @abstractmethod
    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        """
        Compute the metric from pre-computed intermediates.

        Args:
            intermediates: dict mapping intermediate names to their values

        Returns:
            MetricResult containing raw metric value and metadata
        """
        pass

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates: "IntermediatesCache",
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} sets aggregation_fallback_policy='session_aggregated' "
            "but does not override _accumulate_window()."
        )

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        raise NotImplementedError(
            f"{type(self).__name__} sets aggregation_fallback_policy='session_aggregated' "
            "but does not override _finalize_session_state()."
        )

    def _check_bin_and_window_sizes(
        self, bin_size_seconds: float | None, window_length_seconds: float | None
    ) -> tuple[bool, str]:
        """Checks if window and bin sizes are compatible with the metric."""
        if self.min_window_length_seconds is not None:
            if window_length_seconds is None:
                return False, (
                    "Data window length not specified, "
                    "cannot check minimum window length requirement"
                )
            else:
                if window_length_seconds < self.min_window_length_seconds:
                    return False, (
                        f"Data window length {window_length_seconds} is less than "
                        f"minimum required {self.min_window_length_seconds}"
                    )

        # Conversion to smaller bin sizes is not possible, so if
        # the current bin size is too large, it is not compatible.
        if self.max_bin_size_seconds is not None:
            if bin_size_seconds is None:
                return False, (
                    "Data bin size not specified, cannot check maximum bin size requirement"
                )
            if bin_size_seconds > self.max_bin_size_seconds:
                return False, (
                    f"Data bin size {bin_size_seconds} is greater than "
                    f"maximum required {self.max_bin_size_seconds}"
                )

        return True, ""

    def is_compatible(
        self,
        data_type: DataType,
        bin_size_seconds: float | None = None,
        window_length_seconds: float | None = None,
        allow_conversion: bool = True,
        data: EvalData | None = None,
    ) -> tuple[bool, str]:
        """
        Check if metric can be computed with the given data type, and bin and window size.

        Args:
            data_type: The data type available
            bin_size_seconds: Bin size of the data (None for spike times)
            window_length_seconds: Total window length of the data
            allow_conversion: Whether to consider conversions
            data: The data itself, if available; enables data-dependent conversion
                rules (``DataConverter.can_convert_data``)

        Returns:
            Tuple of (is_compatible, reason_string)
        """

        sizes_match, reason = self._check_bin_and_window_sizes(
            bin_size_seconds, window_length_seconds
        )
        if not sizes_match:
            return False, reason

        if data_type in self.supported_data_types:
            return True, ""

        if not allow_conversion:
            return (
                False,
                f"Data type {data_type.value} not in supported types {self.supported_data_types}",
            )

        from ..core.converters import DataConverter

        converter = DataConverter()
        can_convert = (
            converter.can_convert_data(data, self.preferred_data_type)
            if data is not None
            else converter.can_convert(data_type, self.preferred_data_type, bin_size_seconds)
        )

        if can_convert:
            return True, ""
        else:
            reason = converter.get_conversion_error_message(
                data_type, self.preferred_data_type, bin_size_seconds
            )
            return False, reason

    def _make_result(
        self,
        raw_value: float | Any,
        metadata: dict[str, Any] | None = None,
        source_data_type: DataType | None = None,
    ) -> MetricResult:
        """Helper to construct a MetricResult with this metric's attributes."""
        return MetricResult(
            name=self.name,
            category=self.category,
            direction=self.direction,
            level=self.level,
            raw_value=raw_value,
            metadata=metadata or {},
            value_range=self.value_range,
            source_data_type=source_data_type,
        )

    def expand_results(self, result: MetricResult, display_name: str) -> dict[str, MetricResult]:
        """Optionally expand a single result into multiple report entries.

        Multi-scale metrics (e.g. spike count correlation at several bin
        sizes) override this to promote per-scale values from metadata
        into first-class report entries.  The pipeline calls this after
        ``compute()`` and merges the returned dict into the report.

        Args:
            result: The primary MetricResult from ``compute()``.
            display_name: The display name used for the primary result.

        Returns:
            Dict mapping ``display_name_suffix`` → ``MetricResult`` for
            each additional entry.  Default: empty dict (no expansion).
        """
        return {}

    def get_required_intermediates(self, data_type: DataType) -> list[str]:
        """Return the intermediates required for a given target data type."""
        return list(self._required_intermediates)

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        """Return parameterized intermediate requests for a given data type.

        The default implementation wraps ``get_required_intermediates()``
        in parameter-free requests.  Subclasses should override to attach
        their own ``self.params`` to the requests so that different metric
        instances can share or separate cache entries as appropriate.
        """
        return [IntermediateRequest(key) for key in self.get_required_intermediates(data_type)]

"""Base classes for distributions."""

import abc
import importlib
import logging
from typing import Any

import numpy as np
import torch.distributions as torch_dist
from pydantic import PrivateAttr, model_serializer, model_validator

from ntt.utils import NTTMutableBaseModel


class Dist(abc.ABC, NTTMutableBaseModel):
    """Base class for a distribution.

    Allows tokenizer specs to specify distributional parameters
    (e.g., sample window_length from a range during training).

    Subclasses should implement sample() method.

    Attributes:
        return_int: If True, sample() returns integers. If False, returns floats.

    TODO: Integrate :py:class:`torch.distributions.distribution.Distribution`
    """

    return_int: bool = False

    @model_serializer(mode="plain")
    def _serialize_as_concrete_type(self) -> dict[str, Any]:
        """Dump the concrete subclass's fields, tagged with its Hydra ``_target_``.

        Fields are annotated with the base type (``float | Dist``) and pydantic
        serializes through the declared schema, so without this every Dist dumped as
        ``{"return_int": ...}`` and lost its type and parameters. The ``_target_``
        tag lets ``TokenizerSpec`` re-instantiate the dump.
        """
        cls = type(self)
        fields = {name: getattr(self, name) for name in cls.model_fields if name != "hydra_target"}
        return {"_target_": f"{cls.__module__}.{cls.__qualname__}", **fields}

    @property
    def return_type(self) -> type:
        """The return type of sample() method."""
        return int if self.return_int else float

    @abc.abstractmethod
    def sample(self) -> float | int:
        """Sample a value from this distribution.

        Returns:
            Sampled value (type depends on distribution).
        """
        raise NotImplementedError("sample() must be implemented by Dist subclasses")

    @property
    @abc.abstractmethod
    def min(self) -> float | int:
        """Minimum possible value from this distribution."""
        raise NotImplementedError("min must be implemented by Dist subclasses")

    @property
    @abc.abstractmethod
    def max(self) -> float | int:
        """Maximum possible value from this distribution."""
        raise NotImplementedError("max must be implemented by Dist subclasses")


class TriangleDist(Dist):
    r"""Triangular distribution with a peak at mode_units, going from min_units to
    max_units.

    The unnormalized density function is defined as:

    .. math::
        f(x) =
        \\begin{cases}
        0 & \\text{if } x < \\text{min_units} \\\\
        1 + (\\text{peak} - 1) \\cdot \\frac{x - \\text{min_units}}
        {\\text{mode_units} - \\text{min_units}}
        & \\text{if } \\text{min_units} \\leq x \\leq \\text{mode_units} \\\\
        \\text{peak} - (\\text{peak} - 1) \\cdot \\frac{x - \\text{mode_units}}
        {\\text{tail_right} - \\text{mode_units}}
        & \\text{if } \\text{mode_units} \\leq x \\leq \\text{tail_right} \\\\
        1 & \\text{if } \\text{tail_right} \\leq  x \\leq \\text{max_units}\\\\
        0 & \\text{otherwise}
        \\end{cases}

    To sample from the distribution, we use rejection sampling. We sample from a uniform
    distribution between `min_units` and `max_units` and accept the sample with
    probability :math:`\frac{f(x)}{M \cdot q(x)}`, where :math:`q(x)` is the proposal
    distribution.

    Attributes:
        min_units: Minimum number of units to sample.
        mode_units: Mode of the distribution.
        max_units: Maximum number of units to sample.
        tail_right: Right tail of the distribution. If None, it is set to `max_units`.
        peak: Height of the peak of the distribution.
        M: Normalization constant for the proposal distribution.
        max_attempts: Maximum number of attempts to sample from the distribution.
        seed: Seed for the random number generator. If None, the generator will be
            unseeded and non-deterministic.
        return_int: If True, return integer samples. If False, return float samples.
    """

    min_units: int = 20
    mode_units: int = 100
    max_units: int = 300
    tail_right: int | None = None
    peak: float = 4.0
    M: int = 10
    max_attempts: int = 100
    seed: int | None = None
    return_int: bool = True

    _rng: np.random.Generator = PrivateAttr()

    @model_validator(mode="after")
    def _set_defaults_and_init_rng(self) -> "TriangleDist":
        if self.tail_right is None:
            self.tail_right = self.max_units
        self._rng = np.random.default_rng(seed=self.seed)
        return self

    def unnormalized_density_function(self, x: float) -> float:
        """Compute the unnormalized density at point x."""
        if x < self.min_units:
            return 0.0
        if x <= self.mode_units:
            return 1 + (self.peak - 1) * (x - self.min_units) / (self.mode_units - self.min_units)
        if x <= self.tail_right:  # type: ignore
            return self.peak - (self.peak - 1) * (x - self.mode_units) / (
                self.tail_right - self.mode_units  # type: ignore
            )
        if x <= self.max_units:
            return 1.0
        return 0.0

    def sample(self) -> float | int:
        """Sample a value from the triangular distribution using rejection sampling.

        Returns:
            Sampled value as float or int depending on return_int.
        """
        num_attempts = 0
        while True:
            # Sample from the proposal distribution (uniform)
            x = self.min_units + self._rng.uniform() * (self.max_units - self.min_units)
            u = self._rng.uniform()

            # Accept with probability f(x) / (M * q(x))
            # q(x) is uniform, so it's constant and cancels out
            if u <= self.unnormalized_density_function(x) / self.M:
                if self.return_int:
                    return int(round(x))
                else:
                    return x

            num_attempts += 1
            if num_attempts > self.max_attempts:
                logging.warning(
                    f"Could not sample from TriangleDist after {num_attempts} attempts, "
                    f"returning mode_units={self.mode_units}."
                )
                if self.return_int:
                    return self.mode_units
                else:
                    return float(self.mode_units)

    @property
    def min(self) -> float | int:
        return self.min_units if self.return_int else float(self.min_units)

    @property
    def max(self) -> float | int:
        return self.max_units if self.return_int else float(self.max_units)


class UniformDist(Dist):
    """Uniform distribution between low and high.

    Attributes:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive for float, inclusive for int).
        seed: Seed for the random number generator. If None, the generator will be
            unseeded and non-deterministic.
        return_int: If True, return integer samples. If False, return float samples.
    """

    low: float = 0.0
    high: float = 1.0
    seed: int | None = None
    return_int: bool = False

    _rng: np.random.Generator = PrivateAttr()

    @model_validator(mode="after")
    def _init_rng(self) -> "UniformDist":
        self._rng = np.random.default_rng(seed=self.seed)
        return self

    def sample(self) -> float | int:
        """Sample a value from the uniform distribution.

        Returns:
            Sampled value as float or int depending on return_int.
        """
        if self.return_int:
            # For integers, high is inclusive
            return self._rng.integers(low=int(self.low), high=int(self.high) + 1).item()
        else:
            return self._rng.uniform(low=self.low, high=self.high)

    @property
    def min(self) -> float | int:
        return int(self.low) if self.return_int else self.low

    @property
    def max(self) -> float | int:
        return int(self.high) if self.return_int else self.high


class GaussianDist(Dist):
    """Gaussian (normal) distribution with mean and standard deviation.

    Attributes:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
        seed: Seed for the random number generator. If None, the generator will be
            unseeded and non-deterministic.
        clip_low: Optional lower bound to clip samples.
        clip_high: Optional upper bound to clip samples.
        return_int: If True, return integer samples. If False, return float samples.
    """

    mean: float = 0.0
    std: float = 1.0
    seed: int | None = None
    clip_low: float | None = None
    clip_high: float | None = None
    return_int: bool = False

    _rng: np.random.Generator = PrivateAttr()

    @model_validator(mode="after")
    def _init_rng(self) -> "GaussianDist":
        self._rng = np.random.default_rng(seed=self.seed)
        return self

    def sample(self) -> float | int:
        """Sample a value from the Gaussian distribution.

        Returns:
            Sampled value as float or int depending on return_int.
        """
        value = self._rng.normal(loc=self.mean, scale=self.std)

        # Apply clipping if bounds are specified
        if self.clip_low is not None:
            value = max(value, self.clip_low)
        if self.clip_high is not None:
            value = min(value, self.clip_high)

        if self.return_int:
            return int(round(value))
        else:
            return value

    @property
    def min(self) -> float | int:
        if self.clip_low is None:
            raise ValueError(
                "GaussianDist.min requires clip_low to be set "
                "(unbounded Gaussian has no finite minimum)"
            )
        return int(round(self.clip_low)) if self.return_int else self.clip_low

    @property
    def max(self) -> float | int:
        if self.clip_high is None:
            raise ValueError(
                "GaussianDist.max requires clip_high to be set "
                "(unbounded Gaussian has no finite maximum)"
            )
        return int(round(self.clip_high)) if self.return_int else self.clip_high


class PytorchDist(Dist):
    """Wrapper around PyTorch distributions.

    Allows using any PyTorch distribution class by specifying its name and parameters.
    The distribution is instantiated from the specified module and class, then
    sample() returns a single scalar value.

    Attributes:
        dist_class: Name of the distribution class (e.g., "Normal", "Exponential").
        dist_module: Module path containing the distribution class.
            Defaults to "torch.distributions".
        dist_params: Dictionary of parameters to pass to the distribution constructor.
        return_int: If True, return integer samples. If False, return float samples.

    Example:
        >>> dist = PytorchDist(
        ...     dist_class="Normal",
        ...     dist_params={"loc": 0.0, "scale": 1.0},
        ... )
        >>> dist.sample()
        0.1234  # random float from standard normal

        >>> dist = PytorchDist(
        ...     dist_class="Exponential",
        ...     dist_params={"rate": 0.5},
        ... )
        >>> dist.sample()
        1.567  # random float from exponential distribution

        >>> # Using a distribution from a different module
        >>> dist = PytorchDist(
        ...     dist_class="ExponentialFamily",
        ...     dist_module="torch.distributions.exp_family",
        ...     dist_params={...},
        ... )
    """

    dist_class: str
    dist_module: str = "torch.distributions"
    dist_params: dict[str, Any] = {}
    return_int: bool = False
    bound_low: float | None = None
    bound_high: float | None = None

    _dist: torch_dist.Distribution = PrivateAttr()

    @model_validator(mode="after")
    def _init_distribution(self) -> "PytorchDist":
        try:
            module = importlib.import_module(self.dist_module)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Module '{self.dist_module}' not found. "
                f"Ensure the module path is correct (e.g., 'torch.distributions')."
            ) from e

        try:
            dist_cls = getattr(module, self.dist_class)
        except AttributeError as e:
            raise ValueError(
                f"Distribution class '{self.dist_class}' not found in module "
                f"'{self.dist_module}'. Check the class name or see "
                f"https://pytorch.org/docs/stable/distributions.html for available distributions."
            ) from e

        self._dist = dist_cls(**self.dist_params)
        return self

    def sample(self) -> float | int:
        """Sample a single value from the PyTorch distribution.

        Returns:
            Sampled value as float or int depending on return_int.
        """
        value = self._dist.sample().item()
        if self.return_int:
            return int(round(value))
        return value

    @property
    def min(self) -> float | int:
        if self.bound_low is None:
            raise ValueError("PytorchDist.min requires bound_low to be set explicitly")
        return int(round(self.bound_low)) if self.return_int else self.bound_low

    @property
    def max(self) -> float | int:
        if self.bound_high is None:
            raise ValueError("PytorchDist.max requires bound_high to be set explicitly")
        return int(round(self.bound_high)) if self.return_int else self.bound_high

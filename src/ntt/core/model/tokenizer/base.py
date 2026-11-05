"""Base classes and registry for tokenizers."""

from __future__ import annotations

import abc
from typing import ClassVar, Literal

import pydantic as pdt
import torch
from torch import nn

from ntt.core.components.dist import Dist
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    RatesInput,
    ReconstructionOutput,
    SamplingStrategy,
    ShufflingStrategy,
    SpikeTimesInput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.validation import make_dummy_input
from ntt.utils import NTTBaseModel, RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _omegaconf_to_plain(value: object) -> object:
    """Replace OmegaConf containers in ``value`` by plain ones, also inside plain containers."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[arg-type]
    if type(value) is dict:
        return {key: _omegaconf_to_plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_omegaconf_to_plain(item) for item in value]
    if type(value) is tuple:
        return tuple(_omegaconf_to_plain(item) for item in value)
    return value


class CapabilityValidationError(ValueError):
    """Exception raised when tokenizer capability validation fails.

    Collects all validation errors and presents them together.

    Args:
        errors: List of validation error messages
        tokenizer_name: Name of the tokenizer class that failed validation
    """

    def __init__(self, errors: list[str], tokenizer_name: str):
        self.errors = errors
        self.tokenizer_name = tokenizer_name
        error_count = len(errors)
        error_list = "\n  - ".join(errors)
        message = (
            f"Capability validation failed for {tokenizer_name} "
            f"({error_count} error{'s' if error_count != 1 else ''}):\n  - {error_list}"
        )
        super().__init__(message)


class TokenizerSpec(
    # populate_by_name lets specs validate from dumps keyed by field name
    # (model_dump() without by_alias emits `hydra_target`, not its `_target_` alias).
    abc.ABC,
    NTTBaseModel,
    strict=False,
    frozen=True,
    extra="forbid",
    populate_by_name=True,
):
    """Base class for a tokenizer configuration.

    This spec contains both:
    1. Shared attributes needed by datamodule (n_neurons, window_length, etc.)
    2. Algorithm-specific attributes defined in subclasses
    3. Capability metadata describing what the tokenizer can do

    Parameters:
        KIND (str): Discriminator field identifying the tokenizer type. Subclasses must
            override with a Literal type (e.g., KIND: Literal["vqvae"] = "vqvae")
        n_neurons (int | Dist | Literal[False]): Number of neurons to use. False -> full population
        window_length_seconds (float | Dist): Duration of input window in seconds
        bin_size_ms (float | Literal[False]): Temporal bin size. False -> spike times (unbinned)
        shuffling_strategy (str): How to shuffle data during training (``ShufflingStrategy``)
        sampling_strategy (str): How to sample windows from data (``SamplingStrategy``)

    Capability Metadata (must be set by subclasses):
        latent_type: Type of latent representation (continuous or discrete)
        compress_n: Whether the tokenizer compresses along the neuron dimension
        compress_t: Whether the tokenizer compresses along the time dimension
        variable_n: Whether the tokenizer supports variable number of neurons
        variable_t: Whether the tokenizer supports variable time length
        variable_dt: Whether the tokenizer supports variable time step (bin size)
        input_type: Type of input data the tokenizer expects
        output_type: Type of data the tokenizer reconstructs
    """

    # Discriminator field - subclasses must override with a Literal type
    KIND: str

    hydra_target: str | None = pdt.Field(
        default=None,
        alias="_target_",
        description="Hydra instantiation target (ignored). "
        "Allows YAML configs to work with both Hydra and Pydantic.",
    )

    # Data loading parameters
    n_neurons: int | Dist | Literal[False] = False
    window_length_seconds: float | Dist
    bin_size_ms: float | Literal[False] = False

    @pdt.model_validator(mode="before")
    @classmethod
    def _instantiate_hydra_dicts(cls, data: dict) -> dict:
        """Instantiate nested Hydra _target_ dicts into proper objects.

        When Hydra resolves YAML configs, nested objects with _target_ arrive
        as raw dicts (DictConfig) rather than instantiated Python objects.
        This validator detects and instantiates them before Pydantic validation.
        """
        if not isinstance(data, dict):
            # OmegaConf DictConfig is dict-like but not dict
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(data):
                    data = OmegaConf.to_container(data, resolve=True)  # type: ignore[assignment]
                else:
                    return data
            except ImportError:
                return data

        # Hydra passes nested config values as DictConfig / ListConfig (``_convert_="none"``).
        # Loosely typed fields (``dict``, ``Any``) kept them as-is, so they were pickled into
        # checkpoints and broke the spec's JSON dump. Store plain containers instead, also
        # when the config sits inside a plain dict/list (e.g. ``vqpt_kwargs["levels"]``).
        data = {key: _omegaconf_to_plain(val) for key, val in data.items()}

        for field in ("window_length_seconds", "n_neurons"):
            value = data.get(field)
            if isinstance(value, dict) and "_target_" in value:
                import hydra

                data[field] = hydra.utils.instantiate(value)
        return data

    shuffling_strategy: ShufflingStrategy = ShufflingStrategy.NONE
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM

    # Whether to validate reconstruct_from_input() with dummy data during tokenizer creation
    validate_from_data: bool = False

    # Capability metadata - must be set by subclasses
    latent_type: LatentType
    compress_n: bool
    compress_t: bool
    variable_n: bool
    variable_t: bool
    variable_dt: bool
    input_type: DataType
    output_type: DataType
    # Only consulted when input_type == DataType.SPIKES; controls whether
    # the data pipeline binarizes spike counts in DataToModelInput.
    # Validator below rejects an explicit False on non-SPIKES specs to
    # avoid the silent-ignore footgun.
    binarize_spikes: bool = True

    # Temporal resolution of the tokenizer's latent tokens.
    # Must be set explicitly by tokenizers that compress time (compress_t=True)
    # and produce uniform-duration tokens.
    # Defaults to bin_size_ms when compress_t=False (no temporal compression).
    #
    # Dynamic tokenizers with variable token durations should leave this as
    # None and instead return token timestamps directly from encode().
    # See encode() docstring for the return format.
    token_bin_size_ms: float | None = None

    # Latent dimension semantics - describes which dimensions the tokenizer
    # conceptually produces (True even if singleton in edge cases).
    # All tokenizers output [B, T, U, D] with singleton dims where needed.
    has_time_dim: bool = False  # Tokenizer produces multiple tokens over time
    has_unit_dim: bool = False  # Tokenizer produces per-unit representations
    has_latent_dim: bool = False  # Tokenizer produces multi-dim latent per token

    @pdt.model_validator(mode="after")
    def _validate_binarize_spikes_coherence(self) -> TokenizerSpec:
        if not self.binarize_spikes and self.input_type is not DataType.SPIKES:
            raise ValueError(
                f"{type(self).__name__}: binarize_spikes=False has no effect "
                f"unless input_type=DataType.SPIKES (got input_type={self.input_type}). "
                f"Either set input_type=DataType.SPIKES or leave binarize_spikes=True."
            )
        return self

    @pdt.model_validator(mode="after")
    def _set_default_token_bin_size(self) -> TokenizerSpec:
        """Infer token_bin_size_ms when possible, error when ambiguous.

        - compress_t=False: token_bin_size_ms = bin_size_ms (no compression)
        - compress_t=True, has_time_dim=False: one token per window,
          so token_bin_size_ms = window_length_seconds * 1000
        - compress_t=True, has_time_dim=True: ambiguous, must be set explicitly
        """
        if self.token_bin_size_ms is not None:
            return self
        if not self.compress_t:
            if self.bin_size_ms is not False:
                object.__setattr__(self, "token_bin_size_ms", float(self.bin_size_ms))
            # else: spike-times input, no bin size — leave None
        elif not self.has_time_dim:
            # Whole window → single token
            if isinstance(self.window_length_seconds, Dist):
                raise ValueError(
                    "Cannot derive token_bin_size_ms from a variable window length (Dist)"
                )
            object.__setattr__(self, "token_bin_size_ms", self.window_length_seconds * 1000.0)
        else:
            raise ValueError(
                f"{type(self).__name__}: compress_t=True and has_time_dim=True "
                f"but token_bin_size_ms is not set. Tokenizers that compress "
                f"time into multiple tokens must explicitly declare their "
                f"output token temporal resolution."
            )
        return self

    @property
    def bin_size_seconds(self) -> float | Literal[False]:
        """Get bin size in seconds (or False for unbinned)."""
        if isinstance(self.bin_size_ms, float):
            return self.bin_size_ms / 1000.0
        return self.bin_size_ms

    @property
    def window_length_bins(self) -> int:
        """Get window length in number of bins."""
        if self.bin_size_seconds is False:
            raise ValueError(
                "Cannot compute window_length_bins for unbinned data (bin_size_ms=False)"
            )
        if isinstance(self.window_length_seconds, Dist):
            raise ValueError(
                "Cannot compute window_length_bins for variable window lengths (Dist)"
            )
        return round(self.window_length_seconds / self.bin_size_seconds)

    @property
    def effective_window_length_seconds(self) -> float:
        """Resolve window_length_seconds to a concrete float.

        Returns Dist.max for distributions, the value itself for floats.
        Use this whenever a single deterministic window length is needed
        (e.g., evaluation, inference, duration metadata).
        """
        if isinstance(self.window_length_seconds, Dist):
            return float(self.window_length_seconds.max)
        return self.window_length_seconds

    def get_vocab_size(self) -> int:
        """Return the vocabulary size (number of discrete tokens).

        Returns:
            Size of the discrete token vocabulary.
        """
        raise NotImplementedError(
            "get_vocab_size must be implemented by subclasses with a discrete latent type"
        )

    def get_codes_per_token(self) -> int:
        """Return how many codes from ``get_vocab_size()`` one latent position holds.

        A latent position is one ``[t, u]`` entry of ``encode()``'s ``[B, T, U, D]``
        output. It carries ``get_codes_per_token() * log2(get_vocab_size())`` bits,
        whatever ``D`` is. ``D`` says how the position is stored, not how much it
        holds. An FSQ/BSQ/LFQ quantizer returns D floats that together form ONE code,
        so the answer is 1. A residual quantizer stores one code per stage, so the
        answer is ``num_quantizers``.

        ``compute_compression`` reads this to count output bits. Discrete specs must
        override it. The base raises rather than guess, so a spec that does not
        override it gets a NaN compression rate and a warning, not a wrong number.

        Returns:
            Number of vocabulary-sized codes per latent position.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not declare how many codes it emits per latent "
            "position; discrete specs must implement get_codes_per_token()"
        )


class BaseTokenizer(abc.ABC, nn.Module):
    """Base class for all tokenizer implementations.

    Tokenizers convert neural spike data to/from discrete or continuous tokens.

    Subclasses must define a `spec_class` class attribute that points to their
    corresponding TokenizerSpec subclass. This is used by the registry to link
    spec types to tokenizer implementations.

    Example:
        class MyTokenizer(BaseTokenizer):
            spec_class: ClassVar[type[MyTokenizerSpec]] = MyTokenizerSpec

            def __init__(self, spec: MyTokenizerSpec):
                super().__init__(spec)
                # Initialize your architecture here

    Args:
        spec: TokenizerSpec instance containing configuration
    """

    # Required class attribute: subclasses must set this to their spec class
    spec_class: ClassVar[type[TokenizerSpec]]

    def __init__(self, spec: TokenizerSpec):
        super().__init__()
        # Validate that spec is of the correct type
        if not isinstance(spec, self.spec_class):
            raise TypeError(
                f"spec must be an instance of {self.spec_class.__name__}, "
                f"got {type(spec).__name__}"
            )
        self._spec = spec

        # Automatically validate capabilities on initialization
        self._validate_capabilities()

    @property
    def spec(self) -> TokenizerSpec:
        """Get the tokenizer specification."""
        return self._spec

    def get_vocab_size(self) -> int:
        """Return the vocabulary size (number of discrete tokens).

        Delegates to the spec's get_vocab_size() method for convenience.
        This allows accessing vocab_size directly on the tokenizer instance.
        Can be left not implemented for continuous tokenizers.

        Returns:
            Size of the discrete token vocabulary.

        Raises:
            NotImplementedError: If spec's get_vocab_size() is not implemented
        """
        return self.spec.get_vocab_size()

    @abc.abstractmethod
    def encode(self, x: SpikeTimesInput | BinnedSpikesInput | RatesInput) -> torch.Tensor | dict:
        """Encode neural data to latent tokens.

        Args:
            x: Input neural data

        Returns:
            Either a plain tensor or a dict. The plain tensor form is for
            uniform-bin tokenizers where token_bin_size_ms is set in the spec.

            **Dynamic tokenizers** (variable token durations, UNTESTED) must
            return a dict with the following keys::

                {
                    "latents":          torch.Tensor,  # [B, T', U, D] encoded tokens
                    "token_timestamps": torch.Tensor,  # [B, T'] center time (seconds, absolute)
                    "token_start_time": torch.Tensor,  # [B, T'] start time (seconds, absolute)
                    "token_end_time":   torch.Tensor,  # [B, T'] end time (seconds, absolute)
                }

            ``"token_timestamps"`` (center times) is **required** — the
            validation pipeline uses it for behavioral data alignment,
            and as fallback for padding stripping and overlap trimming.
            ``"token_start_time"`` and ``"token_end_time"`` are recommended
            — the validation pipeline uses them for padding removal and
            overlap trimming. When absent, center times are used as fallback
            (UNTESTED).

            When these keys are present, ``encode_with_timestamps()`` passes
            them through directly instead of computing uniform timestamps
            from ``spec.token_bin_size_ms``.
        """
        raise NotImplementedError("encode() must be implemented by BaseTokenizer subclasses")

    @abc.abstractmethod
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to neural data.

        Args:
            tokens: Token indices (discrete) or latent embeddings (continuous)

        Returns:
            Reconstructed neural data
        """
        raise NotImplementedError("decode() must be implemented by BaseTokenizer subclasses")

    @abc.abstractmethod
    def forward(
        self, x: SpikeTimesInput | BinnedSpikesInput
    ) -> TokenizerOutput | tuple[SpikeTimesInput | BinnedSpikesInput, TokenizerOutput]:
        """Forward pass through tokenizer.

        Subclasses must implement this method to:
        1. Encode input to tokens/latents
        2. Decode back to reconstruction
        3. Compute any auxiliary loss (KL, commitment, etc.) if applicable

        Note: Reconstruction loss is computed externally by the litmodule using
        a configurable LossSpec. The tokenizer only computes auxiliary losses
        that are intrinsic to its architecture (e.g., KL divergence for VAE).

        Returning a `(batch, output)` tuple lets the tokenizer replace the
        litmodule's local batch reference with a transformed copy (e.g.
        WanNeuralTokenizer's tile-batched curvature path tiles 1s inputs into
        100ms windows; the modified batch carries the per-tile spike_matrix
        and pad_and_gap_mask so the standard loss / Pearson / raster path
        sees shape-matched targets). Subclasses that don't need the override should
        return only `TokenizerOutput`. IMPORTANT: the tuple form MUST NOT be returned
        during eval (``self.training == False``) — the litmodule's
        ``validation_step`` handles the tuple path but rebinding the batch
        there would replace validation targets with tiled inputs, corrupting
        metrics. Only return the tuple during training.

        Args:
            x: Input neural data

        Returns:
            Either a `TokenizerOutput` (default) or a tuple
            `(modified_batch, TokenizerOutput)` where `modified_batch`
            replaces the caller's batch downstream.
        """
        raise NotImplementedError("forward() must be implemented by BaseTokenizer subclasses")

    def get_effective_pad_mask(self) -> torch.Tensor | None:
        """Return a modified pad mask reflecting training-time input corruptions.

        Tokenizers that drop or zero out input positions during training
        (e.g. neuron dropout) should override this to return a mask that
        excludes those positions, so the reconstruction loss does not
        penalize the model for failing to reconstruct unseen data.

        Called by the litmodule after ``forward()`` to obtain the mask
        passed to the loss function.  Returns ``None`` by default, which
        tells the litmodule to use the batch's own pad mask unchanged.
        """
        return None

    def reconstruct(self, latents: torch.Tensor) -> ReconstructionOutput:
        """Decode continuous latents into a reconstruction comparable with the input.

        This is the tokenizer's **inference function**.  The default
        implementation calls ``decode(latents)`` and wraps the result with
        ``data_type`` and ``bin_size_seconds`` from the spec.

        **Subclasses SHOULD override this** whenever the raw ``decode()``
        output is not directly comparable with the original input.  For
        example, a Poisson-loss tokenizer's ``decode()`` returns rate
        (lambda) estimates; its ``reconstruct()`` should apply Poisson
        sampling so the output looks like spike counts.

        Override contract:
            - Must return a ``ReconstructionOutput``.
            - ``data_type`` must match ``self.spec.output_type``.
            - ``bin_size_seconds`` must be consistent with the spec.

        Args:
            latents: Continuous latent embeddings returned by ``encode()`` or
                converted from discrete token IDs by ``tokens_to_latents()``.

        Returns:
            ReconstructionOutput with reconstruction tensor and type metadata.
        """
        reconstruction = self.decode(latents)
        bin_size_seconds = (
            self.spec.bin_size_ms / 1000.0 if self.spec.bin_size_ms is not False else None
        )
        return ReconstructionOutput(
            reconstruction=reconstruction,
            data_type=self.spec.output_type,
            bin_size_seconds=bin_size_seconds,
        )

    def tokens_to_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens from ``encode()`` into continuous latent vectors.

        The default implementation returns *tokens* unchanged, which is
        correct for tokenizers whose ``encode()`` already produces
        continuous representations (e.g. VAE, WANNeural FSQ/BSQ).

        **Subclasses SHOULD override this** when ``encode()`` returns
        discrete indices rather than continuous embeddings — for example,
        a VQ tokenizer should look up the codebook vectors here.  The
        eval pipeline calls this before feeding latents to downstream
        metrics (behavioural regression, structure preservation, etc.).

        Args:
            tokens: Output of ``encode()`` — shape varies by tokenizer.

        Returns:
            Continuous latent tensor suitable for distance / regression
            computations.  Shape may differ from *tokens* (e.g. the
            last dimension grows from 1 to ``embed_dim`` after a
            codebook lookup).
        """
        return tokens

    def _unpack_encode_result(
        self,
        result: torch.Tensor | dict,
    ) -> tuple[torch.Tensor, dict | None]:
        """Split an ``encode()`` return into raw tokens + optional metadata.

        ``encode()`` may return either a plain tensor (uniform-bin tokenizers)
        or a dict with ``"latents"`` plus ``"token_timestamps"`` (and
        optionally ``"token_start_time"`` / ``"token_end_time"``) for dynamic
        tokenizers.  The dict's ``"latents"`` slot holds the raw tokens —
        for VQ-style tokenizers these are discrete indices, NOT codebook
        vectors.  This helper returns the raw tokens plus a dict of the
        remaining keys (or ``None`` when there are none) so callers can
        thread token-level timestamp metadata through without re-encoding.
        """
        if isinstance(result, dict):
            tokens = result["latents"]
            provided = {k: v for k, v in result.items() if k != "latents"}
            return tokens, (provided if provided else None)
        if isinstance(result, torch.Tensor):
            return result, None
        raise TypeError(
            f"encode() returned unexpected type {type(result).__name__}. "
            f"Expected torch.Tensor or dict with 'latents' key."
        )

    def _compute_token_timestamps(
        self,
        x: SpikeTimesInput | BinnedSpikesInput | RatesInput,
        n_token_bins: int,
        provided: dict | None = None,
    ) -> dict:
        """Return the token timestamp metadata dict for ``x``.

        If ``provided`` carries ``"token_timestamps"`` (dynamic-tokenizer
        path, UNTESTED), passes those through together with any
        ``token_start_time`` / ``token_end_time`` keys it carries.
        Otherwise derives uniform per-token timestamps from
        ``x.absolute_start`` and ``self.spec.token_bin_size_ms``::

            token_timestamps[i] = absolute_start + (i + 0.5) * token_bin_size_sec
            token_start_time[i] = absolute_start + i * token_bin_size_sec
            token_end_time[i]   = absolute_start + (i + 1) * token_bin_size_sec
        """
        if provided is not None and "token_timestamps" in provided:
            metadata = {"token_timestamps": provided["token_timestamps"]}
            # Start/end times are optional; the validation pipeline falls
            # back to center times for stripping/overlap if these are absent.
            if "token_start_time" in provided:
                metadata["token_start_time"] = provided["token_start_time"]
            if "token_end_time" in provided:
                metadata["token_end_time"] = provided["token_end_time"]
            return metadata

        assert x.absolute_start is not None, (
            "Token timestamps require absolute_start to be set on the batch "
            "when the tokenizer does not provide its own."
        )

        token_bin_size_ms = self.spec.token_bin_size_ms
        assert token_bin_size_ms is not None, (
            f"{type(self).__name__}: spec.token_bin_size_ms is None and encode() "
            f"did not return token timestamps. Either set token_bin_size_ms in "
            f"the tokenizer spec, or have encode() return a dict with "
            f"'token_timestamps'."
        )

        token_bin_size_sec = token_bin_size_ms / 1000.0
        abs_start = x.absolute_start.to(dtype=torch.float64)  # [B]
        indices = torch.arange(n_token_bins, dtype=torch.float64, device=abs_start.device)  # [T']

        # Per-sample timestamps: [B, T']
        return {
            "token_timestamps": abs_start[:, None] + (indices[None, :] + 0.5) * token_bin_size_sec,
            "token_start_time": abs_start[:, None] + indices[None, :] * token_bin_size_sec,
            "token_end_time": abs_start[:, None] + (indices[None, :] + 1) * token_bin_size_sec,
        }

    def encode_with_timestamps(
        self,
        x: BinnedSpikesInput,
    ) -> tuple[torch.Tensor, dict]:
        """Encode input and return continuous latents + per-token timestamps.

        Calls ``encode(x)`` to obtain raw tokens, applies
        ``self.tokens_to_latents`` so the returned ``latents`` are always
        the continuous representation (codebook vectors for VQ-style
        tokenizers, identity passthrough for VAE/FSQ/BSQ), and derives
        per-token absolute timestamps from ``x.absolute_start`` and
        ``self.spec.token_bin_size_ms``.  If the encoder already returns
        timestamps (dict result with a ``"token_timestamps"`` key), those
        are passed through unchanged.

        Use ``self.encode(x)`` directly when you need the *raw* tokens
        (e.g. to feed into ``self.reconstruct``); ``encode_with_timestamps``
        is the consistent latents+timestamps entry point for downstream
        metrics that operate on continuous latents.

        Token timestamps give the absolute CENTER time of each token's window::

            token_timestamps[i] = absolute_start + (i + 0.5) * token_bin_size_sec
            token_start_time[i] = absolute_start + i * token_bin_size_sec
            token_end_time[i]   = absolute_start + (i + 1) * token_bin_size_sec

        Args:
            x: Batched input data.  Must have ``absolute_start`` set.

        Returns:
            ``(latents, metadata)`` where *latents* is the continuous
            latent tensor (typically ``[B, T', U, D]``) and *metadata* is
            a dict with keys ``"token_timestamps"``, ``"token_start_time"``,
            ``"token_end_time"``, each of shape ``[B, T']``.
        """
        if not isinstance(x, (BinnedSpikesInput, RatesInput)):
            raise TypeError(
                f"encode_with_timestamps requires BinnedSpikesInput or RatesInput, "
                f"got {type(x).__name__}"
            )
        # TODO: dynamic tokenizers may not need absolute_start (they provide their own timestamps)
        assert x.absolute_start is not None, (
            "encode_with_timestamps requires absolute_start to be set on the input"
        )

        tokens, provided = self._unpack_encode_result(self.encode(x))
        latents = self.tokens_to_latents(tokens)
        metadata = self._compute_token_timestamps(x, latents.shape[1], provided)
        return latents, metadata

    def reconstruct_from_input(
        self,
        x: SpikeTimesInput | BinnedSpikesInput | RatesInput,
        return_latents: bool = False,
        return_latent_timestamps: bool = True,
    ) -> ReconstructionOutput | dict:
        """Encode input and reconstruct — the default eval pipeline entry point.

        Calls ``encode(x)`` to obtain raw tokens (which may be discrete
        indices for VQ-style tokenizers), converts them with
        ``tokens_to_latents()``, then reconstructs from the continuous
        representation. When ``return_latents=True`` the returned dict's
        ``"latents"`` field contains that same representation — codebook
        vectors for VQ, identity for VAE/FSQ/BSQ.

        Args:
            x: Input neural data (BinnedSpikesInput, SpikeTimesInput, or RatesInput).
            return_latents: If True, return a dict with the reconstruction
                (: Tensor), continuous latents (: Tensor), and (optionally)
                token timestamps.  The dict format is compatible with
                ``iter_collated_validation_segments``.
            return_latent_timestamps: If True, and if ``return_latents`` is
                True, include token timestamps in the returned dict.

        Returns:
            ReconstructionOutput when ``return_latents=False`` (default), or
            a dict with keys ``"reconstruction": Tensor``,
            ``"reconstruction_data_type": DataType`` (the type ``reconstruct()``
            declared), ``"latents": Tensor`` (continuous), and token timestamp
            entries when True.
        """
        tokens, provided = self._unpack_encode_result(self.encode(x))
        latents = self.tokens_to_latents(tokens)
        result = self.reconstruct(latents)

        if not return_latents:
            return result

        out: dict = {
            "reconstruction": result.reconstruction,
            "reconstruction_data_type": result.data_type,
            "latents": latents,
        }
        if return_latent_timestamps:
            out.update(self._compute_token_timestamps(x, latents.shape[1], provided))
        return out

    def _validate_capabilities(self) -> None:
        """Validate that the implementation matches declared capabilities.

        This is called automatically during __init__ and raises CapabilityValidationError
        if any validation errors are found.

        Raises:
            CapabilityValidationError: If validation fails with one or more errors
        """
        if errors := self._collect_validation_errors():
            raise CapabilityValidationError(errors, self.__class__.__name__)
        if self.spec.validate_from_data:
            self.validate_with_dummy_data()

    def _collect_validation_errors(self) -> list[str]:
        """Collect validation errors without raising exceptions.

        This method can be overridden by subclasses to add implementation-specific
        validation. The base implementation checks basic consistency.

        Returns:
            List of validation error messages (empty if all checks pass).
        """
        errors: list[str] = []

        # Check that latent_type matches implementation
        # If latent_type is DISCRETE, vocab_size should be > 0 and get_vocab_size
        #  must be implemented
        if self.spec.latent_type == LatentType.DISCRETE:
            # Check if get_vocab_size is actually implemented (doesn't raise NotImplementedError)
            try:
                vocab_size = self.spec.get_vocab_size()
            except NotImplementedError:
                errors.append(
                    "latent_type is DISCRETE but get_vocab_size() is not implemented "
                    "(must implement get_vocab_size() for discrete tokenizers)"
                )
            else:
                if vocab_size <= 0:
                    errors.append(
                        f"latent_type is DISCRETE but vocab_size is {vocab_size}"
                        " (must be > 0 for discrete tokenizers)"
                    )

        # If variable_n is False, n_neurons should be fixed (not False or Dist)
        if not self.spec.variable_n and self.spec.n_neurons is False:
            errors.append(
                "variable_n is False but n_neurons is False (must specify fixed number of neurons)"
            )

        # If variable_t is False, window_length_seconds should be fixed (not Dist)
        if not self.spec.variable_t and isinstance(self.spec.window_length_seconds, Dist):
            errors.append(
                "variable_t is False but window_length_seconds is a Dist (must specify fixed"
                " window length)"
            )

        # If variable_dt is False, bin_size_ms should be fixed (not False or Dist)
        if not self.spec.variable_dt:
            if self.spec.bin_size_ms is False:
                errors.append(
                    "variable_dt is False but bin_size_ms is False (must specify fixed bin size)"
                )
            elif isinstance(self.spec.bin_size_ms, Dist):
                errors.append(
                    "variable_dt is False but bin_size_ms is a Dist (must specify fixed bin size)"
                )

        return errors

    def _validate_reconstruct_with_dummy_data(self) -> list[str]:
        """Run reconstruct_from_input() with synthetic data and verify output.

        Called by validate_with_dummy_data() after full init (not during
        __init__, since subclass architecture may not be built yet).

        Returns:
            List of validation error messages (empty if all checks pass).
        """
        errors: list[str] = []
        try:
            dummy_input = make_dummy_input(self.spec)
            with torch.no_grad():
                result = self.reconstruct_from_input(dummy_input)
        except Exception as e:
            errors.append(
                f"reconstruct_from_input() failed on dummy input: {type(e).__name__}: {e}"
            )
            return errors

        if not isinstance(result, ReconstructionOutput):
            errors.append(
                f"reconstruct_from_input() returned {type(result).__name__}, "
                f"expected ReconstructionOutput"
            )
            return errors

        if result.data_type != self.spec.output_type:
            errors.append(
                f"reconstruct_from_input() output data_type is {result.data_type}, "
                f"expected {self.spec.output_type} (from spec.output_type)"
            )

        expected_bin_size = (
            self.spec.bin_size_ms / 1000.0 if self.spec.bin_size_ms is not False else None
        )
        if result.bin_size_seconds != expected_bin_size:
            errors.append(
                f"reconstruct_from_input() output bin_size_seconds is "
                f"{result.bin_size_seconds}, expected {expected_bin_size}"
            )

        if result.reconstruction.ndim < 2:
            errors.append(
                f"reconstruct_from_input() output tensor has "
                f"{result.reconstruction.ndim} dimensions, "
                f"expected at least 2 (batch, time, ...)"
            )

        return errors

    def validate_with_dummy_data(self) -> None:
        """Validate that reconstruct_from_input() works on synthetic data.

        Called automatically by create_tokenizer() after full initialization.
        Can also be called manually after constructing a tokenizer directly.

        Raises:
            CapabilityValidationError: If validation fails
        """
        if errors := self._validate_reconstruct_with_dummy_data():
            raise CapabilityValidationError(errors, self.__class__.__name__)


# Global registry mapping tokenizer names to (SpecClass, TokenizerClass)
_TOKENIZER_REGISTRY: dict[str, tuple[type[TokenizerSpec], type[BaseTokenizer]]] = {}


def register_tokenizer(tokenizer_class: type[BaseTokenizer]) -> type[BaseTokenizer]:
    """Decorator to register a tokenizer implementation.

    Usage:
        @register_tokenizer
        class VQVAETokenizer(BaseTokenizer):
            spec_class = VQVAETokenizerSpec  # must have KIND: Literal["vqvae"] = "vqvae"

    The decorator expects:
    1. A spec_class class attribute pointing to a TokenizerSpec subclass
    2. The spec_class must have a `KIND` field with a Literal type default value

    The registration name is automatically inferred from the spec_class's KIND field.

    Returns:
        The decorated tokenizer class
    """
    # Get the spec class from the tokenizer class
    if not hasattr(tokenizer_class, "spec_class"):
        raise ValueError(
            f"Tokenizer class {tokenizer_class.__name__} must define 'spec_class' class attribute"
        )

    spec_class = tokenizer_class.spec_class
    if not issubclass(spec_class, TokenizerSpec):
        raise TypeError(f"spec_class must be a subclass of TokenizerSpec, got {spec_class}")

    # Get the KIND from the spec's model fields
    kind_field = spec_class.model_fields.get("KIND")
    if kind_field is None:
        raise ValueError(f"Spec class {spec_class.__name__} must define a 'KIND' field")

    name = kind_field.default
    if name is None or not isinstance(name, str):
        raise ValueError(
            f"Spec class {spec_class.__name__} must have a string default for KIND field"
        )

    if name in _TOKENIZER_REGISTRY:
        log.warning(f"Tokenizer '{name}' is already registered!")

    _TOKENIZER_REGISTRY[name] = (spec_class, tokenizer_class)
    return tokenizer_class


def get_tokenizer_class(name: str) -> tuple[type[TokenizerSpec], type[BaseTokenizer]]:
    """Get tokenizer spec and implementation classes by name.

    Args:
        name: Registered tokenizer name

    Returns:
        Tuple of (spec_class, tokenizer_class)

    Raises:
        KeyError: If tokenizer name not found in registry
    """
    if name not in _TOKENIZER_REGISTRY:
        available = ", ".join(_TOKENIZER_REGISTRY.keys())
        raise KeyError(f"Tokenizer '{name}' not found. Available tokenizers: {available}")
    return _TOKENIZER_REGISTRY[name]


def create_tokenizer(spec: TokenizerSpec) -> BaseTokenizer:
    """Factory function to create a tokenizer from a spec.

    The spec must be registered via @register_tokenizer decorator.
    Uses the spec's `KIND` field to look up the tokenizer class.

    After construction, runs dummy-data validation to verify that
    reconstruct() works correctly and returns properly typed output.

    Args:
        spec: TokenizerSpec instance (must be a registered type)

    Returns:
        Instantiated and validated tokenizer

    Raises:
        KeyError: If tokenizer KIND not found in registry
        CapabilityValidationError: If dummy-data validation fails
    """
    KIND = spec.KIND

    if KIND not in _TOKENIZER_REGISTRY:
        available = ", ".join(_TOKENIZER_REGISTRY.keys())
        raise KeyError(
            f"Tokenizer KIND '{KIND}' not found in registry. "
            f"Available tokenizers: {available}. "
            f"Did you forget to import the tokenizer implementation?"
        )

    _spec_class, tokenizer_class = _TOKENIZER_REGISTRY[KIND]
    return tokenizer_class(spec)


def list_tokenizers() -> list[str]:
    """List all registered tokenizer names.

    Returns:
        List of registered tokenizer names
    """
    return list(_TOKENIZER_REGISTRY.keys())

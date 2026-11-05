"""Compressed sensing tokenizer."""

import math
from typing import Any, Literal

import pydantic as pdt
import torch
from einops import rearrange

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    ReconstructionOutput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer import register_tokenizer
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec


class CompressedSensingTokenizerSpec(TokenizerSpec, frozen=True):
    """Specification for the compressed sensing tokenizer.

    Parameters:
        compress_mode: Controls which dimensions the sensing matrix acts on.

            - ``"per_neuron"`` (default): Sensing matrix ``[m, T]`` applied
              independently to each neuron's time series. Latent shape:
              ``[B, U, m]``. Compresses time only (``compress_t=True``).

            - ``"per_time_bin"``: Sensing matrix ``[m, U]`` applied independently
              at each time bin across all neurons. Latent shape: ``[B, T, m]``.
              Compresses neurons only (``compress_n=True``). Requires ``n_neurons``
              to be a fixed int.

            - ``"full_window"``: Sensing matrix ``[m, T*U]`` applied to the
              flattened spatiotemporal window. Latent shape: ``[B, m]``.
              Compresses both dimensions (``compress_n=True, compress_t=True``).
              Requires ``n_neurons`` to be a fixed int.

        latent_dim: Number of compressed measurements.
        trainable: If True, the sensing matrix is learnable.
        random_seed: Optional seed for deterministic matrix initialization.
        solver: Reconstruction solver. ``"fista"`` (default) uses a pure-PyTorch
            GPU-native FISTA algorithm (no external dependencies). ``"cvxpy"``
            uses cvxpylayers on CPU (requires cvxpy and cvxpylayers).
        fista_iterations: Number of FISTA iterations for reconstruction.
        fista_lambda: L1 regularization strength as a fraction of lambda_max
            (= max |A^T y| per problem). Smaller = sparser but needs more
            iterations. 0.01 is a good default for spike data.
        nonneg: If True, enforce non-negativity in reconstruction
            (natural for spike counts).
    """

    KIND: Literal["compressed_sensing"] = "compressed_sensing"

    # Data parameters
    n_neurons: int = 1
    window_length_seconds: float = 1.0
    bin_size_ms: float = 1.0

    # Architecture parameters
    compress_mode: Literal["per_neuron", "per_time_bin", "full_window"] = "per_neuron"
    latent_dim: int = 50
    trainable: bool = False
    random_seed: int | None = None

    # Solver parameters
    solver: Literal["fista", "cvxpy"] = "fista"
    fista_iterations: int = 500
    fista_lambda: float = 0.01
    nonneg: bool = True

    # Training loss weights (only used when trainable=True)
    lambda_coherence: float = 0.1
    lambda_structure: float = 0.05
    lambda_smoothness: float = 0.0
    lambda_temporal: float = 0.0
    max_pairs: int = 512
    temporal_max_pairs: int = 1024

    # Matrix initialization
    matrix_type: Literal["gaussian", "toeplitz", "sinusoidal"] = "gaussian"
    sinusoidal_base: float = 10000.0
    sinusoidal_norm: bool = True

    # Capability metadata (auto-derived from compress_mode via validator)
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = False
    compress_t: bool = True
    variable_n: bool = False
    variable_t: bool = False
    variable_dt: bool = False
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.SPIKES

    # Latent dimension semantics (auto-derived from compress_mode via validator)
    # All modes produce latent_dim > 1, so has_latent_dim is always True.
    has_time_dim: bool = False  # per_time_bin has time dim
    has_unit_dim: bool = True  # per_neuron has unit dim
    has_latent_dim: bool = True  # Always True for CS

    # Debug info (how smooth/nonsmooth is the reconstruction?)

    @pdt.model_validator(mode="before")
    @classmethod
    def _set_capabilities_from_mode(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mode = data.get("compress_mode", "per_neuron")
            if mode == "per_neuron":
                data.setdefault("compress_n", False)
                data.setdefault("compress_t", True)
                data.setdefault("has_time_dim", False)
                data.setdefault("has_unit_dim", True)
                data.setdefault("has_latent_dim", True)
            elif mode == "per_time_bin":
                data.setdefault("compress_n", True)
                data.setdefault("compress_t", False)
                data.setdefault("has_time_dim", True)
                data.setdefault("has_unit_dim", False)
                data.setdefault("has_latent_dim", True)
                # per_time_bin tokenizes each time bin independently, so it can
                # consume variable-length windows -> opt out of fixed-input chunking.
                data.setdefault("variable_t", True)
                n = data.get("n_neurons", 1)
                if not isinstance(n, int) or n < 1:
                    raise ValueError(
                        f"compress_mode='per_time_bin' requires n_neurons to be a "
                        f"positive int, got {n!r}"
                    )
            elif mode == "full_window":
                data.setdefault("compress_n", True)
                data.setdefault("compress_t", True)
                data.setdefault("has_time_dim", False)
                data.setdefault("has_unit_dim", False)
                data.setdefault("has_latent_dim", True)
                n = data.get("n_neurons", 1)
                if not isinstance(n, int) or n < 1:
                    raise ValueError(
                        f"compress_mode='full_window' requires n_neurons to be a "
                        f"positive int, got {n!r}"
                    )
        return data

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Continuous latent -> no discrete vocabulary")


@register_tokenizer
class CompressedSensingTokenizer(BaseTokenizer):
    """Compressed sensing tokenizer.

    Compresses neural spike data via random linear measurements and
    reconstructs binned spikes through L1-minimization (basis pursuit).

    Uses FISTA (Fast Iterative Shrinkage-Thresholding) by default — a
    pure-PyTorch solver that runs on whatever device the model is on.

    Three compression modes are supported, controlled by
    ``spec.compress_mode``:

    **per_neuron** (default)
        Sensing matrix shape ``[m, T]``, applied independently to each
        neuron's time series.

        - Input: ``[B, T, U]`` -> Latent: ``[B, U, m]``
        - ``compress_t=True, compress_n=False``

    **per_time_bin**
        Sensing matrix shape ``[m, U]``, applied independently at each
        time bin across all neurons.

        - Input: ``[B, T, U]`` -> Latent: ``[B, T, m]``
        - ``compress_t=False, compress_n=True``
        - Requires ``n_neurons`` to be a fixed int.

    **full_window**
        Sensing matrix shape ``[m, T*U]``, applied to the flattened
        spatiotemporal window.

        - Input: ``[B, T, U]`` -> Latent: ``[B, m]``
        - ``compress_t=True, compress_n=True``
        - Requires ``n_neurons`` to be a fixed int.
    """

    spec_class = CompressedSensingTokenizerSpec
    spec: CompressedSensingTokenizerSpec

    def __init__(self, spec: CompressedSensingTokenizerSpec):
        super().__init__(spec)

        total_time_bins = math.ceil(
            self.spec.window_length_seconds * 1000.0 / self.spec.bin_size_ms
        )
        self.total_time_bins = total_time_bins
        self.latent_dim = self.spec.latent_dim
        self.compress_mode = self.spec.compress_mode

        # Compute signal dimension based on compression mode.
        if self.compress_mode == "per_neuron":
            signal_dim = total_time_bins
        elif self.compress_mode == "per_time_bin":
            signal_dim = self.spec.n_neurons
        else:  # full_window
            signal_dim = total_time_bins * self.spec.n_neurons
        self.signal_dim = signal_dim

        # Initialize sensing matrix.
        # Non-Toeplitz: stored as self.random_matrix (Parameter or buffer).
        # Toeplitz+trainable: stored as self._toeplitz_vector (Parameter);
        #   self.random_matrix is intercepted by __getattr__ to build the
        #   full matrix on the fly.
        # Use _get_sensing_matrix() internally for all paths.
        self._use_toeplitz = self.spec.matrix_type == "toeplitz" and self.spec.trainable
        if self._use_toeplitz:
            gen_vector = self._init_toeplitz_vector(self.latent_dim, signal_dim)
            self._toeplitz_vector = torch.nn.Parameter(gen_vector, requires_grad=True)
        else:
            if self.spec.matrix_type == "sinusoidal":
                matrix = self._init_sinusoidal_matrix(self.latent_dim, signal_dim)
            else:
                matrix = self._init_random_matrix(self.latent_dim, signal_dim)
            if self.spec.trainable:
                self.random_matrix = torch.nn.Parameter(matrix, requires_grad=True)
            else:
                self.register_buffer("random_matrix", matrix)

        if self.spec.trainable:
            from ntt.contrib.loss.compressed_sensing import CSTokenizerLoss

            self._cs_loss = CSTokenizerLoss(
                lambda_coherence=self.spec.lambda_coherence,
                lambda_structure=self.spec.lambda_structure,
                lambda_smoothness=self.spec.lambda_smoothness,
                lambda_temporal=self.spec.lambda_temporal,
                max_pairs=self.spec.max_pairs,
                temporal_max_pairs=self.spec.temporal_max_pairs,
            )

        if self.spec.solver == "cvxpy":
            self._setup_cvxpy_solver()

        self._lipschitz_bound = None

    def _setup_cvxpy_solver(self) -> None:
        """Set up the cvxpy/cvxpylayers solver (CPU-only)."""
        import cvxpy as cp
        from cvxpylayers.torch.cvxpylayer import CvxpyLayer

        x_hat = cp.Variable(self.signal_dim)
        y_param = cp.Parameter(self.latent_dim)
        A_param = cp.Parameter((self.latent_dim, self.signal_dim))
        problem = cp.Problem(
            cp.Minimize(cp.norm(x_hat, 1)),
            [A_param @ x_hat == y_param],
        )
        self._cvxpy_layer = CvxpyLayer(
            problem,
            parameters=[y_param, A_param],
            variables=[x_hat],
        )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, x: BinnedSpikesInput) -> torch.Tensor:
        """Encode binned spike counts to compressed measurements.

        Returns
        -------
        torch.Tensor
            Unified shape ``[B, T, U, D]`` (batch, time, units, latent_dim).
            Singleton dimensions are inserted where the mode does not have
            that dimension:

            - per_neuron:   ``[B, 1, U, m]`` (time=1, has units and latent)
            - per_time_bin: ``[B, T, 1, m]`` (units=1, has time and latent)
            - full_window:  ``[B, 1, 1, m]`` (time=1, units=1, has latent)
        """
        x_spike_matrix = x.spike_matrix
        if x_spike_matrix.ndim != 3:
            raise ValueError(
                "Expected spike_matrix with shape [batch, time, units], "
                f"got shape {tuple(x_spike_matrix.shape)}"
            )

        x_float = x_spike_matrix.float()
        A_T = self._get_sensing_matrix().T  # [signal_dim, m]

        if self.compress_mode == "per_neuron":
            # [b,t,u] x [t,m] -> [b,u,m] -> [b,1,u,m]
            z = torch.einsum("btu,tl->bul", x_float, A_T)
            z = z.unsqueeze(1)  # Insert time dimension
        elif self.compress_mode == "per_time_bin":
            # [b,t,u] x [u,m] -> [b,t,m] -> [b,t,1,m]
            z = torch.einsum("btu,ul->btl", x_float, A_T)
            z = z.unsqueeze(2)  # Insert unit dimension
        else:  # full_window
            # [b, t*u] x [t*u, m] -> [b, m] -> [b,1,1,m]
            x_flat = rearrange(x_float, "b t u -> b (t u)")
            z = x_flat @ A_T
            z = z.unsqueeze(1).unsqueeze(2)  # Insert time and unit dimensions

        return z

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode compressed measurements back to binned spike counts.

        Parameters
        ----------
        z : torch.Tensor
            Compressed measurements in unified shape ``[B, T, U, D]``:

            - per_neuron:   ``[B, 1, U, m]`` (time dim is singleton)
            - per_time_bin: ``[B, T, 1, m]`` (unit dim is singleton)
            - full_window:  ``[B, 1, 1, m]`` (both are singletons)

        Returns
        -------
        torch.Tensor
            Reconstructed spike matrix ``[B, T, U]``.
        """
        context = torch.no_grad() if not self.spec.trainable else torch.enable_grad()
        with context:
            if self.compress_mode == "full_window":
                return self._decode_full_window(z)
            elif self.compress_mode == "per_time_bin":
                return self._decode_per_time_bin(z)
            else:
                return self._decode_per_neuron(z)

    def _decode_per_neuron(self, z: torch.Tensor) -> torch.Tensor:
        """Decode per-neuron latent [B, 1, U, m] -> [B, T, U]."""
        z = z.squeeze(1)  # [B, 1, U, m] -> [B, U, m]
        batch_size, num_units, latent_dim = z.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        z_flat = z.reshape(batch_size * num_units, latent_dim)
        x_hat_flat = self._solve_l1(z_flat)
        x_hat = rearrange(x_hat_flat, "(b u) t -> b t u", b=batch_size, u=num_units)
        return x_hat

    def _decode_per_time_bin(self, z: torch.Tensor) -> torch.Tensor:
        """Decode per-time-bin latent [B, T, 1, m] -> [B, T, U]."""
        z = z.squeeze(2)  # [B, T, 1, m] -> [B, T, m]
        batch_size, num_time_bins, latent_dim = z.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        z_flat = z.reshape(batch_size * num_time_bins, latent_dim)
        x_hat_flat = self._solve_l1(z_flat)
        x_hat = rearrange(x_hat_flat, "(b t) u -> b t u", b=batch_size, t=num_time_bins)
        return x_hat

    def _decode_full_window(self, z: torch.Tensor) -> torch.Tensor:
        """Decode full-window latent [B, 1, 1, m] -> [B, T, U]."""
        z = z.squeeze(1).squeeze(1)  # [B, 1, 1, m] -> [B, m]
        batch_size, latent_dim = z.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        x_hat_flat = self._solve_l1(z)
        x_hat = rearrange(
            x_hat_flat,
            "b (t u) -> b t u",
            t=self.total_time_bins,
            u=self.spec.n_neurons,
        )
        return x_hat

    # ------------------------------------------------------------------
    # L1 solvers
    # ------------------------------------------------------------------

    def _solve_l1(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Solve L1 minimization for a batch of measurements.

        Parameters
        ----------
        z_flat : torch.Tensor
            Shape ``[N, latent_dim]`` where N is the flattened batch.

        Returns
        -------
        torch.Tensor
            Shape ``[N, signal_dim]``, the reconstructed signals.
        """
        if self.spec.solver == "fista":
            return self._solve_l1_fista(z_flat)
        return self._solve_l1_cvxpy(z_flat)

    def _solve_l1_fista(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Reconstruct signals via FISTA (Fast Iterative Shrinkage-Thresholding).

        Solves LASSO: min 0.5 ||Ax - y||^2 + lam * ||x||_1
        which approximates basis pursuit (min ||x||_1 s.t. Ax = y) for
        small lam.
        """
        A = self._get_sensing_matrix()  # [m, n]
        y = z_flat  # [N, m]

        # Step size = 1/L where L = largest eigenvalue of A^T A
        # = largest eigenvalue of AA^T (since m < n typically).
        eigvalsh_result = torch.linalg.eigvalsh(A @ A.T)

        L = eigvalsh_result[-1]
        step = 1.0 / L

        # Per-sample lambda: fista_lambda * max|A^T y_i| for each problem.
        # lambda_max is the smallest lambda making x=0 optimal.
        Aty = y @ A  # [N, n]
        lam_max = Aty.abs().amax(dim=1, keepdim=True)  # [N, 1]
        threshold = self.spec.fista_lambda * lam_max * step  # [N, 1]

        # FISTA iterations
        x_prev = torch.zeros_like(Aty)
        x = x_prev
        z_iter = x_prev.clone()
        t = 1.0
        nonneg = self.spec.nonneg

        for _ in range(self.spec.fista_iterations):
            # Gradient of 0.5 ||Az - y||^2 w.r.t. z: A^T(Az - y)
            grad = (z_iter @ A.T - y) @ A  # [N, n]
            v = z_iter - step * grad

            # Proximal step: soft-thresholding (+ non-negativity if enabled)
            if nonneg:
                x = torch.clamp(v - threshold, min=0)
            else:
                x = torch.sign(v) * torch.clamp(v.abs() - threshold, min=0)

            # Nesterov momentum
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            z_iter = x + ((t - 1.0) / t_new) * (x - x_prev)

            x_prev = x
            t = t_new

        return x

    def _solve_l1_cvxpy(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Solve via cvxpylayers (CPU). Results moved back to input device."""
        z_cpu = z_flat.cpu()
        A_cpu = self._get_sensing_matrix().cpu()
        (x_hat,) = self._cvxpy_layer(z_cpu, A_cpu)
        return x_hat.to(z_flat.device)

    # ------------------------------------------------------------------
    # Forward / Reconstruct
    # ------------------------------------------------------------------

    def forward(self, x: BinnedSpikesInput) -> TokenizerOutput:
        z = self.encode(x)

        if self.spec.trainable:
            # Compute auxiliary CS losses (coherence + structure).
            cs_losses = self._cs_loss(
                sensing_matrix=self._get_sensing_matrix(),
                x_original=x.spike_matrix.float(),
                z_compressed=z,
                compress_mode=self.compress_mode,
            )
            # Return "total" as the grad-carrying loss under a clean name.
            # The individual detached components (coherence, structure, etc.)
            # are not included since the litmodule sums all dict values.
            aux_loss = {"cs": cs_losses["total"]}

            if self.training:
                # Training: skip expensive FISTA decode.
                reconstruction = torch.zeros_like(x.spike_matrix, dtype=torch.float32)
            else:
                # Eval: full decode for reconstruction quality monitoring.
                reconstruction = self.decode(z)

            return TokenizerOutput(tokens=z, reconstruction=reconstruction, aux_loss=aux_loss)

        # Non-trainable: standard encode + decode, no aux_loss.
        x_hat = self.decode(z)
        return TokenizerOutput(tokens=z, reconstruction=x_hat)

    def reconstruct(self, tokens: torch.Tensor) -> ReconstructionOutput:
        # Simply round the solver's output to the nearest integer.
        return ReconstructionOutput(
            reconstruction=torch.round(self.decode(tokens)),
            data_type=self.spec.output_type,
            bin_size_seconds=self.spec.bin_size_ms / 1000.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Make self.random_matrix work in Toeplitz mode too."""
        if name == "random_matrix" and getattr(self, "_use_toeplitz", False):
            return self._build_toeplitz_matrix()
        return super().__getattr__(name)

    def _get_sensing_matrix(self) -> torch.Tensor:
        """Return the [m, signal_dim] sensing matrix."""
        if self._use_toeplitz:
            return self._build_toeplitz_matrix()
        return self.random_matrix

    def _build_toeplitz_matrix(self) -> torch.Tensor:
        """Build [m, n] Toeplitz matrix from the generating vector."""
        m, n = self.latent_dim, self.signal_dim
        rows = torch.arange(m, device=self._toeplitz_vector.device).unsqueeze(1)
        cols = torch.arange(n, device=self._toeplitz_vector.device).unsqueeze(0)
        indices = n - 1 + rows - cols  # [m, n]
        return self._toeplitz_vector[indices]

    def _init_toeplitz_vector(self, latent_dim: int, signal_dim: int) -> torch.Tensor:
        """Initialize the generating vector for a Toeplitz sensing matrix."""
        generator = None
        if self.spec.random_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.spec.random_seed)
        vec_len = latent_dim + signal_dim - 1
        # Explicit device="cpu" to avoid Lightning's __torch_function__
        # redirect to CUDA; the tensor is moved to the right device later.
        vector = torch.randn(vec_len, generator=generator, device="cpu")
        return vector / math.sqrt(latent_dim)

    def _init_sinusoidal_matrix(self, latent_dim: int, signal_dim: int) -> torch.Tensor:
        """Initialize a sensing matrix with sinusoidal basis functions.

        Rows come in sin/cos pairs at geometrically spaced frequencies,
        analogous to transformer positional encodings. Each pair of rows
        captures a different frequency component of the signal.

        If latent_dim is odd, the last row is an extra sin at the highest
        frequency index.
        """
        base = self.spec.sinusoidal_base
        positions = torch.arange(signal_dim, dtype=torch.float32, device="cpu")
        n_sin = (latent_dim + 1) // 2  # ceil(latent_dim / 2)
        n_cos = latent_dim // 2
        n_freqs = max(n_sin, n_cos)
        freq_indices = torch.arange(n_freqs, dtype=torch.float32, device="cpu")
        div_term = torch.exp(freq_indices * (-math.log(base) / latent_dim * 2.0))
        # [n_freqs, signal_dim]: each row is positions scaled by a frequency
        angles = positions.unsqueeze(0) * div_term.unsqueeze(1)

        matrix = torch.zeros(latent_dim, signal_dim, device="cpu")
        matrix[0::2, :] = torch.sin(angles[:n_sin])
        matrix[1::2, :] = torch.cos(angles[:n_cos])
        if self.spec.sinusoidal_norm:
            matrix = matrix / math.sqrt(latent_dim)
        return matrix

    def _init_random_matrix(self, latent_dim: int, signal_dim: int) -> torch.Tensor:
        generator = None
        if self.spec.random_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.spec.random_seed)
        # Explicit device="cpu" to avoid Lightning's __torch_function__
        # redirect to CUDA; the tensor is moved to the right device later.
        if self.spec.matrix_type == "toeplitz":
            vec_len = latent_dim + signal_dim - 1
            vector = torch.randn(vec_len, generator=generator, device="cpu")
            rows = torch.arange(latent_dim).unsqueeze(1)
            cols = torch.arange(signal_dim).unsqueeze(0)
            random_matrix = vector[signal_dim - 1 + rows - cols]
        else:
            random_matrix = torch.randn(latent_dim, signal_dim, generator=generator, device="cpu")
        return random_matrix / math.sqrt(latent_dim)

    @property
    def reconstruction_lipschitz_bound(self) -> float:
        """
        Computes the stability bound (Lipschitz constant) of the
        reconstruction operator based on the current sensing matrix.

        A value near 1.0 is ideal. Higher values indicate
        potential instability to noise.
        """
        with torch.no_grad():
            A = self._get_sensing_matrix()
            # Compute eigenvalues of the gram matrix AA^T
            eigenvalues = torch.linalg.eigvalsh(A @ A.T)
            # eigenvalues[0] is the smallest (most critical for stability)
            smallest_eigval = eigenvalues[0]

            # Using a very small epsilon to allow tracking extremely
            # large constants without overflow.
            return (1.0 / torch.sqrt(smallest_eigval + 1e-14)).item()

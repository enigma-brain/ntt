"""Regularization losses for compressed sensing tokenizers.

Four losses that can be combined for training:

1. **Coherence loss** (RIP proxy): Encourages column orthogonality via the
   Gram matrix of the column-normalized sensing matrix.

2. **Structure loss** (JL-like): Preserves pairwise batch distances between
   original and compressed spaces.

3. **Column smoothness loss**: Penalizes differences between consecutive
   columns of the sensing matrix — a Laplacian regularizer that encourages
   temporally nearby spike positions to have similar latent representations.
   Most meaningful in ``per_neuron`` mode where column index = time bin.

4. **Temporal correlation loss**: Directly encourages the Pearson correlation
   between column-pair latent distance and temporal distance to be positive.
   Samples random column pairs; stochastic but targets the desired property.

Note: coherence loss (column orthogonality) and smoothness/temporal losses
(column similarity for neighbors) pull in opposite directions.  Reduce
``lambda_coherence`` when enabling temporal losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ======================================================================
# Individual loss functions (stateless, pure-functional)
# ======================================================================


def coherence_loss(sensing_matrix: torch.Tensor) -> torch.Tensor:
    """Mutual-coherence penalty on the sensing matrix.

    Computes the Gram matrix of the *column-normalized* sensing matrix
    and penalizes the maximum absolute off-diagonal entry (mutual
    coherence μ).  Minimizing μ is a tractable proxy for encouraging RIP:
    if μ is small, RIP holds for sufficiently sparse signals.

    We return both the max coherence and a smooth differentiable surrogate
    (squared Frobenius norm of off-diagonal entries) that is easier to
    optimize with SGD.

    Parameters
    ----------
    sensing_matrix : torch.Tensor
        Shape ``[m, n]`` — the measurement / sensing matrix.

    Returns
    -------
    torch.Tensor
        Scalar loss: mean squared off-diagonal Gram entry (smooth,
        differentiable, easy to optimize).  Drives all column pairs
        toward orthogonality.
    """
    A = sensing_matrix  # [m, n]
    # Column-normalize: each column has unit ℓ2 norm.
    A_norm = A / (A.norm(dim=0, keepdim=True) + 1e-8)  # [m, n]
    # Gram matrix of normalized columns.
    G = A_norm.T @ A_norm  # [n, n]
    # Zero out diagonal — we only care about cross-column coherence.
    n = G.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=G.device)
    off_diag = G[mask]
    # Smooth loss: mean squared off-diagonal entry.
    return (off_diag**2).mean()


def structure_loss(
    x_original: torch.Tensor,
    z_compressed: torch.Tensor,
    max_pairs: int | None = 512,
    eps: float = 1e-8,
) -> torch.Tensor:
    """JL-like pairwise distance preservation loss.

    For items in the batch, measures how well pairwise L2 distances are
    preserved between the original and compressed spaces.  The loss
    penalizes the *relative* distortion: for each pair (i, j),

        distortion_{ij} = |d_z(i,j) / d_x(i,j) - c|

    where c is the median ratio (a learned-free normalization that
    accounts for the global scale difference between spaces).  This is
    scale-invariant and focuses purely on whether the *ordering and
    ratios* of distances are preserved.

    Parameters
    ----------
    x_original : torch.Tensor
        Original signals flattened to ``[B, D_x]``.
    z_compressed : torch.Tensor
        Compressed representations flattened to ``[B, D_z]``.
    max_pairs : int | None
        If not None, randomly subsample this many pairs to keep cost
        bounded for large batches.  512 pairs is usually plenty.
    eps : float
        Small constant to avoid division by zero for identical inputs.

    Returns
    -------
    torch.Tensor
        Scalar loss — mean squared relative distortion.
    """
    B = x_original.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=x_original.device)

    # Flatten to 2-D if needed.
    x_flat = x_original.reshape(B, -1)
    z_flat = z_compressed.reshape(B, -1)

    # All-pairs indices (upper triangle).
    idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=x_flat.device)
    n_pairs = idx_i.shape[0]

    # Subsample pairs if needed.
    if max_pairs is not None and n_pairs > max_pairs:
        perm = torch.randperm(n_pairs, device=x_flat.device)[:max_pairs]
        idx_i = idx_i[perm]
        idx_j = idx_j[perm]

    # Pairwise ℓ2 distances.
    d_x = (x_flat[idx_i] - x_flat[idx_j]).norm(dim=1)  # [P]
    d_z = (z_flat[idx_i] - z_flat[idx_j]).norm(dim=1)  # [P]

    # Filter out near-zero original distances (identical inputs).
    valid = d_x > eps
    if valid.sum() < 2:
        return torch.tensor(0.0, device=x_original.device)
    d_x = d_x[valid]
    d_z = d_z[valid]

    # Ratio of compressed to original distances.
    ratios = d_z / d_x  # [P']

    # Normalize by median ratio so we're scale-invariant.
    # (detach median so it doesn't create weird gradients)
    median_ratio = ratios.median().detach()
    normalized_ratios = ratios / (median_ratio + eps)

    # Loss: variance of normalized ratios around 1.
    # Perfect JL embedding -> all ratios equal -> zero loss.
    return ((normalized_ratios - 1.0) ** 2).mean()


def column_smoothness_loss(sensing_matrix: torch.Tensor) -> torch.Tensor:
    """Laplacian smoothness penalty on consecutive columns of the sensing matrix.

    Penalizes ``||A[:, i+1] - A[:, i]||²`` across all consecutive column
    pairs.  In ``per_neuron`` mode, columns correspond to time bins, so
    this encourages neighboring spike positions to produce similar latent
    measurements.

    Parameters
    ----------
    sensing_matrix : torch.Tensor
        Shape ``[m, n]``.

    Returns
    -------
    torch.Tensor
        Scalar loss: mean squared consecutive-column difference.
    """
    diffs = sensing_matrix[:, 1:] - sensing_matrix[:, :-1]  # [m, n-1]
    return (diffs**2).mean()


def temporal_correlation_loss(
    sensing_matrix: torch.Tensor,
    max_pairs: int = 1024,
) -> torch.Tensor:
    """Encourage latent distance to correlate with temporal (column-index) distance.

    Samples random column pairs, computes the L2 distance between their
    measurement vectors and the absolute index distance, then minimizes the
    negative Pearson correlation.  A perfect result (correlation = +1) gives
    loss = −1.

    Parameters
    ----------
    sensing_matrix : torch.Tensor
        Shape ``[m, n]``.
    max_pairs : int
        Number of column pairs to sample.  1024 is usually plenty.

    Returns
    -------
    torch.Tensor
        Scalar loss in ``[-1, 1]``.  Lower is better.
    """
    n = sensing_matrix.shape[1]
    if n < 2:
        return torch.tensor(0.0, device=sensing_matrix.device)

    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=sensing_matrix.device)
    total_pairs = idx_i.shape[0]

    if max_pairs < total_pairs:
        perm = torch.randperm(total_pairs, device=sensing_matrix.device)[:max_pairs]
        idx_i = idx_i[perm]
        idx_j = idx_j[perm]

    # Latent distance (L2 between column vectors)
    latent_dist = (sensing_matrix[:, idx_i] - sensing_matrix[:, idx_j]).norm(dim=0)

    # Temporal distance (absolute index difference, no grad)
    temporal_dist = (idx_i - idx_j).abs().float()

    # Pearson correlation → minimize -r to maximize positive correlation
    ld = latent_dist - latent_dist.mean()
    td = temporal_dist - temporal_dist.mean()
    return -(ld * td).sum() / (ld.norm() * td.norm() + 1e-8)


# ======================================================================
# Combined loss module
# ======================================================================


class CSTokenizerLoss(nn.Module):
    """Combined regularization losses for a trainable CS tokenizer.

    Parameters
    ----------
    lambda_coherence : float
        Weight for the coherence (RIP-proxy) loss.
    lambda_structure : float
        Weight for the structure-preservation (JL-like) loss.
    lambda_smoothness : float
        Weight for column smoothness (Laplacian) loss.  0 = disabled.
    lambda_temporal : float
        Weight for temporal correlation loss.  0 = disabled.
    max_pairs : int | None
        Max random pairs for structure loss.
    temporal_max_pairs : int
        Max random column pairs for temporal correlation loss.
    """

    def __init__(
        self,
        lambda_coherence: float = 0.1,
        lambda_structure: float = 0.05,
        lambda_smoothness: float = 0.0,
        lambda_temporal: float = 0.0,
        max_pairs: int | None = 512,
        temporal_max_pairs: int = 1024,
    ):
        super().__init__()
        self.lambda_coherence = lambda_coherence
        self.lambda_structure = lambda_structure
        self.lambda_smoothness = lambda_smoothness
        self.lambda_temporal = lambda_temporal
        self.max_pairs = max_pairs
        self.temporal_max_pairs = temporal_max_pairs

    def forward(
        self,
        sensing_matrix: torch.Tensor,
        x_original: torch.Tensor,
        z_compressed: torch.Tensor,
        compress_mode: str = "per_neuron",
    ) -> dict[str, torch.Tensor]:
        """Compute all regularization losses.

        Parameters
        ----------
        sensing_matrix : torch.Tensor
            The ``[m, n]`` sensing matrix.
        x_original : torch.Tensor
            Original spike matrix, shape ``[B, T, U]``.
        z_compressed : torch.Tensor
            Compressed output from ``tokenizer.encode()``.
        compress_mode : str
            One of ``"per_neuron"``, ``"per_time_bin"``, ``"full_window"``.

        Returns
        -------
        dict with keys ``"coherence"``, ``"structure"``, ``"smoothness"``,
        ``"temporal"``, ``"total"``.  Individual losses are detached
        (for logging); only ``"total"`` carries gradients.
        """
        dev = sensing_matrix.device

        # --- Coherence loss (on the sensing matrix itself) ---
        L_coh = coherence_loss(sensing_matrix)

        # --- Structure loss (on batch pairwise distances) ---
        B = x_original.shape[0]
        x_flat = x_original.reshape(B, -1)
        z_flat = z_compressed.reshape(B, -1)
        L_str = structure_loss(x_flat, z_flat, max_pairs=self.max_pairs)

        total = self.lambda_coherence * L_coh + self.lambda_structure * L_str

        # --- Column smoothness loss ---
        L_smooth = torch.tensor(0.0, device=dev)
        if self.lambda_smoothness > 0:
            L_smooth = column_smoothness_loss(sensing_matrix)
            total = total + self.lambda_smoothness * L_smooth

        # --- Temporal correlation loss ---
        L_temp = torch.tensor(0.0, device=dev)
        if self.lambda_temporal > 0:
            L_temp = temporal_correlation_loss(sensing_matrix, max_pairs=self.temporal_max_pairs)
            total = total + self.lambda_temporal * L_temp

        return {
            "coherence": L_coh.detach(),
            "structure": L_str.detach(),
            "smoothness": L_smooth.detach(),
            "temporal": L_temp.detach(),
            "total": total,
        }

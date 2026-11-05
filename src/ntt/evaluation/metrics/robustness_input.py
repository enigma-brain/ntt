"""input_robustness (M1) metric.

Consumes the replica-fan-out passthroughs
(``replica_latents`` and ``replica_metadata``) to quantify how much
the tokenizer's latents move under input-level corruptions, normalized
by the tokenizer's natural latent scale σ (joint + per-dim).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.types import (
    DataType,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from .base import BaseMetric, register_metric
from .robustness_latent_stats import LatentStatsCache


def _to_ndarray(x: Any) -> np.ndarray:
    """Convert tensor/ndarray-like to a numpy array without copying when possible.

    Lists are NOT accepted here — variable-length per-window data is
    handled explicitly in :meth:`InputCorruptionRobustness.compute`
    rather than being silently stacked.
    """
    # Accept torch.Tensor without importing torch at module import time.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_window_list(x: Any) -> list[np.ndarray]:
    """Normalise replica_latents into a list of per-window ``[K+1, T_w, D]`` arrays.

    Accepts:
      * a stacked ``[N_windows, K+1, T_tok, D]`` torch tensor or ndarray
        (fixed-T case),
      * a Python list of per-window torch tensors / ndarrays of shape
        ``[K+1, T_w, D]`` where ``T_w`` may vary across windows.

    Returns the list-of-arrays form so downstream code can iterate
    windows without assuming uniform ``T_tok``.
    """
    if isinstance(x, list):
        items = [_to_ndarray(item) for item in x]
        # Per-window items may have a per-unit leading-D split, e.g.
        # ``[K+1, T_w, U, F]`` from per-unit tokenizers. Flatten trailing
        # dims so each window is 3D ``[K+1, T_w, D]`` with
        # ``D = U * F``.
        out: list[np.ndarray] = []
        for it in items:
            if it.ndim == 3:
                out.append(it)
            elif it.ndim >= 4:
                out.append(it.reshape(it.shape[0], it.shape[1], -1))
            else:
                raise ValueError(
                    f"replica_latents per-window entry must be at least 3D "
                    f"[K+1, T_w, D]; got shape {it.shape}"
                )
        return out
    arr = _to_ndarray(x)
    # Some tokenizers produce per-unit latents with shape
    # ``[N_windows, K+1, T_tok, U, F]`` (e.g. Poisson baseline). The
    # input_robustness (M1) metric treats the "latent dim" as everything past the time
    # axis, so flatten any trailing axes into a single ``D``.
    if arr.ndim < 4:
        raise ValueError(
            f"replica_latents must be at least 4D [N_windows, K+1, T_tok, D]; "
            f"got ndarray with shape {arr.shape}"
        )
    if arr.ndim > 4:
        arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], -1)
    return [arr[i] for i in range(arr.shape[0])]


@register_metric
class InputCorruptionRobustness(BaseMetric):
    """input_robustness (M1): how much input corruptions move the tokenizer latents.

    ``compute()`` returns M1-a: mean/std latent-space distance between
    replica-0 (clean) and each corrupted replica k≥1, normalized by a
    cached latent σ (joint + per-dim).
    """

    name = "robustness_input"
    category = MetricCategory.LATENT_QUALITY
    direction = MetricDirection.LOWER_IS_BETTER  # distance — smaller is more robust
    level = MetricLevel.POPULATION
    _required_intermediates = [
        "replica_latents",
        "replica_metadata",
    ]

    supported_data_types = {DataType.SPIKES, DataType.SPIKE_TIMES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    _default_params: dict[str, Any] = {}

    def __init__(
        self,
        latent_stats_cache: LatentStatsCache | None = None,
        **params: Any,
    ):
        """Construct the metric.

        Args:
            latent_stats_cache: injectable σ cache. A fresh in-memory cache
                is created when ``None``.
        """
        super().__init__(**params)
        self._latent_stats_cache: LatentStatsCache = (
            latent_stats_cache if latent_stats_cache is not None else LatentStatsCache()
        )

    # ------------------------------------------------------------------ M1-a
    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        """Compute M1-a: normalized latent distances per corruption label.

        Steps:
          1. Extract ``replica_latents`` of shape ``[N_windows, K+1, T_tok, D]``
             (accepts tensor, ndarray, or a list of per-window entries).
          2. Flatten replica-0 across windows to ``(N_windows * T_tok, D)``
             and fetch σ via the ``LatentStatsCache``. The cache key is
             deliberately degenerate here (``id(self)`` + session info if
             available); the full D5 key composition is wired when input_robustness (M1) is
             integrated into the eval pipeline (Phase 4).
          3. Per window w and replica k ≥ 1, compute TWO normalised MSE
             metrics — both squared (no sqrt) so the ratio is a true
             normalised MSE with E = 2 under i.i.d. clean/corrupted:

             A. Joint NMSE (canonical M1-a value):
                ``joint_nmse_w_k = mean_t ||L_w_k[t,:] - L_w_0[t,:]||²
                                   / σ_token²``
                with ``σ_token² = mean_t ||L_t - μ||²`` where μ is the
                mean token vector over all clean tokens.

             B. Per-dim NMSE (averaged across dims):
                ``per_dim_nmse_w_k = mean_d (mean_t (L_w_k[t,d] -
                                     L_w_0[t,d])²) / σ_d²``
                Same E = 2 under i.i.d., but each dim contributes equally
                so off-manifold corruptions that disproportionately move
                low-σ dims show up as larger values than under metric A.
          4. Group by ``replica_metadata["corruption_labels"][k]``
             (falling back to ``"replica_{k}"`` when unavailable), then
             reduce to per-label dicts with keys ``mean_token_nmse``,
             ``std_token_nmse``, ``mean_per_dim_nmse``, ``std_per_dim_nmse``,
             ``per_dim_nmse_breakdown`` (a [D] vector for diagnostic
             plots).
        """
        replica_latents_raw = intermediates["replica_latents"]
        replica_metadata = intermediates["replica_metadata"]

        windows = _to_window_list(replica_latents_raw)  # list[K+1, T_w, D]
        if not windows:
            raise ValueError("replica_latents is empty (no windows to evaluate).")

        # Validate per-window shapes and find K_plus_1 + D.
        K_plus_1: int | None = None
        D: int | None = None
        T_tok_per_window: list[int] = []
        for i, w in enumerate(windows):
            if w.ndim != 3:
                raise ValueError(f"replica_latents[{i}] must be 3D [K+1, T_w, D]; got {w.shape}")
            if K_plus_1 is None:
                K_plus_1 = int(w.shape[0])
                D = int(w.shape[2])
            else:
                if int(w.shape[0]) != K_plus_1 or int(w.shape[2]) != D:
                    raise ValueError(
                        f"replica_latents[{i}] shape {w.shape} disagrees with "
                        f"the first window's K+1={K_plus_1} and D={D}. "
                        f"All windows in a session must share the K+1 (replica) "
                        f"and D (latent-dim) axes; only T_tok may vary."
                    )
            T_tok_per_window.append(int(w.shape[1]))
        assert K_plus_1 is not None and D is not None  # for type-checker
        if K_plus_1 < 2:
            raise ValueError(f"replica_latents must have K+1>=2 along axis 0; got {K_plus_1}")
        K = K_plus_1 - 1
        N_windows = len(windows)

        # σ_latent: concatenate replica-0 tokens across windows (variable T_w
        # tolerated by construction) and compute the per-element std. This
        # is identical to what ``clean.reshape(-1, D).std()`` produced in the
        # fixed-T_tok case; the list-of-windows form is just the variable-T
        # generalisation. See replan/discussion notes for the alternative we
        # rejected (drop short windows + truncate to T_min).
        clean_flat = np.concatenate(
            [w[0].reshape(-1, D) for w in windows], axis=0
        )  # [sum_w T_w, D]

        session_tag = None
        if isinstance(replica_metadata, dict):
            session_tag = replica_metadata.get("session_id")
        key_dict = {
            "metric_instance_id": id(self),
            "session_tag": session_tag,
            "n_windows": int(N_windows),
            "n_tokens_total": int(clean_flat.shape[0]),
            "D": int(D),
        }

        def _latents_flat() -> np.ndarray:
            return clean_flat

        stats = self._latent_stats_cache.get_or_compute(key_dict, _latents_flat)
        sigma_joint = stats.sigma_joint  # per-scalar std over (window, token, dim).
        sigma_per_dim = np.asarray(stats.sigma_per_dim, dtype=float)  # [D]

        # Normalisation — squared-error in numerator AND denominator so the
        # ratio is a normalised MSE (no sqrt-bias).
        #
        # Two metrics, both stored:
        #
        # A) Joint normalised MSE (population / "token-level" ruler):
        #
        #    σ_token² := mean_{w,t} ||L_{w,t,:} - μ||²
        #              = mean_{w,t} sum_d (L_{w,t,d} - μ_d)²
        #
        # where μ is the mean token vector over all clean tokens in the
        # session/segment. This is the per-token joint variance (squared
        # distance to the mean token), i.e. the scalar "token variance"
        # requested for input_robustness (M1).
        #
        #    joint_nmse_w_k = mean_t ||L_w_k[t,:] - L_w_0[t,:]||² / σ_token²
        #
        # B) Per-dim normalised MSE (per-axis ruler, then averaged):
        #
        #    per_dim_mse_w_k_d = mean_t (L_w_k[t,d] - L_w_0[t,d])²
        #    per_dim_nmse_w_k = mean_d per_dim_mse_w_k_d / σ_d²
        sigma_per_dim_sq = sigma_per_dim**2  # [D]
        # Canonical scalar: mean per-token squared L2 distance to the mean token.
        clean_mean_token = clean_flat.mean(axis=0, keepdims=True)  # [1, D]
        sigma_token_sq = float(np.mean(np.sum((clean_flat - clean_mean_token) ** 2, axis=1)))

        # Guard against divide-by-zero per-dim and per-scalar.
        safe_sigma_per_dim_sq = np.where(sigma_per_dim_sq > 0, sigma_per_dim_sq, 1.0)
        safe_sigma_token_sq = sigma_token_sq if sigma_token_sq > 0 else 1.0

        # Default label fallback.
        corruption_labels_raw: list[Any] = []
        if isinstance(replica_metadata, dict):
            corruption_labels_raw = list(replica_metadata.get("corruption_labels", []))

        def _label_for(k: int) -> str:
            # k is the 1-indexed corrupted-replica index (1..K)
            # corruption_labels, if present, may be length K or K+1.
            if corruption_labels_raw:
                if len(corruption_labels_raw) == K_plus_1:
                    return str(corruption_labels_raw[k])
                if len(corruption_labels_raw) == K:
                    return str(corruption_labels_raw[k - 1])
            return f"replica_{k}"

        # Accumulators keyed by (label, replica_id k).
        #   joint_mse_dists[label][k]   = list of per-window raw tokenwise
        #                                 squared-L2 scalars for metric A
        #   joint_nmse_dists[label][k]  = list of per-window scalars for metric A
        #   per_dim_nmse_dists[label][k] = list of per-window scalars for metric B
        #   per_dim_breakdown[label][k]  = list of per-window per-dim NMSE arrays
        #                                  (the per-dim breakdown that B is the
        #                                  mean of). Aggregated to a single [D]
        #                                  vector per label for diagnostics.
        # Per-window scalars are first averaged within a replica, then mean+std
        # are taken across replicas.
        joint_mse_dists: dict[str, dict[int, list[float]]] = {}
        joint_nmse_dists: dict[str, dict[int, list[float]]] = {}
        per_dim_nmse_dists: dict[str, dict[int, list[float]]] = {}
        per_dim_breakdown: dict[str, dict[int, list[np.ndarray]]] = {}

        for w_idx, w in enumerate(windows):
            T_w = T_tok_per_window[w_idx]
            if T_w == 0:
                continue  # zero-token window; skip silently (no contribution)
            clean_w = w[0]  # [T_w, D]
            for k in range(1, K_plus_1):
                diff = w[k] - clean_w  # [T_w, D]
                # Metric A — joint NMSE: mean over tokens of the squared L2
                # distance between corrupted and clean token vectors,
                # divided by the clean token variance above.
                joint_mse = float(np.mean(np.sum(diff**2, axis=1)))
                joint_nmse = joint_mse / safe_sigma_token_sq

                # Metric B — per-dim NMSE: per-dim MSE over t (mean over T_w),
                # divided by per-dim σ_d², then mean over dims.
                per_dim_mse = (diff**2).mean(axis=0)  # [D]
                per_dim_nmse_d = per_dim_mse / safe_sigma_per_dim_sq  # [D]
                per_dim_nmse = float(np.mean(per_dim_nmse_d))

                label = _label_for(k)
                joint_mse_dists.setdefault(label, {}).setdefault(k, []).append(joint_mse)
                joint_nmse_dists.setdefault(label, {}).setdefault(k, []).append(joint_nmse)
                per_dim_nmse_dists.setdefault(label, {}).setdefault(k, []).append(per_dim_nmse)
                per_dim_breakdown.setdefault(label, {}).setdefault(k, []).append(
                    per_dim_nmse_d[None, :]
                )  # [1, D]

        # Two-step reduction per label:
        #   1. per replica k: replica_mean = nanmean(per_window_scalars)
        #   2. across replicas: raw_mean = nanmean(replica_means),
        #      raw_std = nanstd(replica_means)
        # Both metric A (joint NMSE) and metric B (per-dim NMSE) follow the
        # same two-step reduction.
        def _reduce(by_replica: dict[int, list[float]]) -> tuple[float, float, int]:
            """Return (mean, std, n_replicas_used) for one label's by-replica dict."""
            replica_means: list[float] = []
            for _k, vals in by_replica.items():
                arr = np.asarray(vals, dtype=float)
                with np.errstate(invalid="ignore"):
                    replica_means.append(float(np.nanmean(arr)) if arr.size > 0 else float("nan"))
            r_arr = np.asarray(replica_means, dtype=float)
            with np.errstate(invalid="ignore"):
                m = float(np.nanmean(r_arr)) if r_arr.size > 0 else float("nan")
                s = float(np.nanstd(r_arr)) if r_arr.size > 0 else float("nan")
            return m, s, int(np.isfinite(r_arr).sum())

        raw_value: dict[str, dict[str, Any]] = {}
        per_label_std: dict[str, float] = {}
        per_label_mean_joint_mse: dict[str, float] = {}
        per_label_std_joint_mse: dict[str, float] = {}
        n_replicas_used: dict[str, int] = {}
        for label, by_replica_a in joint_nmse_dists.items():
            mean_joint_mse, std_joint_mse, _ = _reduce(joint_mse_dists.get(label, {}))
            mean_A, std_A, n_used = _reduce(by_replica_a)
            mean_B, std_B, _ = _reduce(per_dim_nmse_dists.get(label, {}))

            # Per-replica per-dim NMSE breakdown: average each replica's
            # per-window per-dim arrays, then average across replicas. Used
            # by diagnostic plots that show which dims absorb the corruption.
            replica_perdim: list[np.ndarray] = []
            for _k, lst in per_dim_breakdown.get(label, {}).items():
                pd_stack = np.concatenate(lst, axis=0)  # [n_w_k, D]
                replica_perdim.append(pd_stack.mean(axis=0))  # [D]
            per_dim_arr = (
                np.stack(replica_perdim, axis=0)
                if replica_perdim
                else np.zeros((0, sigma_per_dim.shape[0]), dtype=float)
            )
            per_dim_breakdown_mean = (
                per_dim_arr.mean(axis=0).tolist()
                if per_dim_arr.size > 0
                else [float("nan")] * int(sigma_per_dim.shape[0])
            )

            raw_value[label] = {
                # Raw numerator scale for the tokenwise joint metric.
                "mean_joint_mse": mean_joint_mse,
                "std_joint_mse": std_joint_mse,
                # Metric A — joint NMSE (canonical M1-a value).
                "mean_token_nmse": mean_A,
                "std_token_nmse": std_A,
                # Metric B — per-dim NMSE (averaged across dims).
                "mean_per_dim_nmse": mean_B,
                "std_per_dim_nmse": std_B,
                # Per-dim NMSE breakdown ([D] vector); plotting uses this
                # for the per-dim panel of the M1-a-per-dim figure.
                "per_dim_nmse_breakdown": per_dim_breakdown_mean,
            }
            per_label_std[label] = std_A
            per_label_mean_joint_mse[label] = mean_joint_mse
            per_label_std_joint_mse[label] = std_joint_mse
            n_replicas_used[label] = n_used

        # Diagnostic σ flavors (do NOT change the metric scale —
        # they're for users to inspect the latent variance structure):
        #
        # - ``sigma_within_window``: mean over windows of the per-window
        #   per-element std (taken across that window's T_w × D scalars).
        #   If σ_within ≈ σ_joint, latent variability is mostly within
        #   a single window. If σ_within < σ_joint, there is meaningful
        #   between-window variance that σ_joint absorbs but σ_within
        #   doesn't — informative when comparing to the input_robustness (M1) ratio.
        # - ``sigma_per_position``: a curve over token position. For
        #   every token index t, std over the (window × dim) scalars
        #   at that position. Useful for spotting positional structure
        #   in the latent variance (e.g. early vs late tokens). Stored
        #   for the typical (most common) T_tok across windows; windows
        #   shorter than that contribute up to their own length.
        within_stds: list[float] = []
        for w in windows:
            clean_w = w[0]  # [T_w, D]
            if clean_w.size > 0:
                within_stds.append(float(clean_w.std()))
        sigma_within_window = float(np.mean(within_stds)) if within_stds else 0.0

        # Per-token-position curve: for each t in 0..T_max-1, gather all
        # L[w, t, :] across windows that have token t, then std-flatten.
        T_max = max(T_tok_per_window) if T_tok_per_window else 0
        sigma_per_position: list[float] = []
        if T_max > 0:
            for t in range(T_max):
                vals = []
                for w_idx, w in enumerate(windows):
                    T_w = T_tok_per_window[w_idx]
                    if t < T_w:
                        vals.append(w[0, t, :].reshape(-1))
                if vals:
                    sigma_per_position.append(float(np.concatenate(vals).std()))
                else:
                    sigma_per_position.append(0.0)

        metadata = {
            # ``sigma_token_sq`` is the canonical denominator for metric A
            # (joint NMSE): mean per-token squared distance to the mean token.
            # ``sigma_per_dim`` (a [D] vector of per-dim std) is the
            # denominator for metric B (squared, per-dim).
            # ``sigma_joint`` is kept as a diagnostic for comparing
            # alternative scalarizations.
            "sigma_token_sq": float(sigma_token_sq),
            "sigma_joint": float(sigma_joint),
            "sigma_per_dim": sigma_per_dim.tolist(),
            "sigma_within_window": sigma_within_window,
            "sigma_per_position": sigma_per_position,
            "n_windows": int(N_windows),
            "n_replicas": int(K),
            # Per-label std is computed across replicas (each replica's
            # per-window mean is one sample). When all K replicas under
            # one corruption spec share the same label, this is the
            # spread across the K random samples — the wanted semantic.
            "per_label_std": per_label_std,
            "per_label_mean_joint_mse": per_label_mean_joint_mse,
            "per_label_std_joint_mse": per_label_std_joint_mse,
            "n_replicas_used": n_replicas_used,
        }

        return self._make_result(raw_value=raw_value, metadata=metadata)

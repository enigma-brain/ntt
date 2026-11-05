"""Windowed (session-aggregated) implementation for MutualInformation.

The non-windowed ``compute()`` and the MI numerical kernels (``_discrete_mi``,
``_discrete_mi_from_sparse_counts``, ``_entropy_bits``) stay in
:mod:`ntt.evaluation.metrics.information`; only the per-window accumulation /
finalization lives here. ``MutualInformation`` mixes this in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.types import MetricResult
from ._shared import MetricMixinBase, metric_requests_by_key


class MutualInformationWindowedMixin(MetricMixinBase):
    """Session-aggregated NMI: accumulate sparse contingency tables per window."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        requests = metric_requests_by_key(self)
        counts_orig = window_intermediates.get(requests["binned_counts_original"])
        counts_recon = window_intermediates.get(requests["binned_counts_reconstructed"])

        per_bin_state = session_state.setdefault("per_bin_state", {})
        for bin_size in sorted(counts_orig):
            if bin_size not in counts_recon:
                raise ValueError(
                    "MutualInformation._accumulate_window: reconstructed counts missing "
                    f"bin size {bin_size}ms."
                )
            orig_at_bin = np.asarray(counts_orig[bin_size], dtype=np.float64)
            recon_at_bin = np.asarray(counts_recon[bin_size], dtype=np.float64)
            if orig_at_bin.shape != recon_at_bin.shape:
                raise ValueError(
                    "MutualInformation._accumulate_window: original/reconstructed "
                    f"shape mismatch at {bin_size}ms: {orig_at_bin.shape} vs {recon_at_bin.shape}."
                )
            if orig_at_bin.ndim != 2:
                raise ValueError(
                    "MutualInformation._accumulate_window: expected 2-D binned counts "
                    f"at {bin_size}ms, got shape {orig_at_bin.shape}."
                )

            bin_state = per_bin_state.setdefault(
                bin_size,
                {
                    "joint_counts_per_unit": [{} for _ in range(orig_at_bin.shape[0])],
                    "orig_counts_per_unit": [{} for _ in range(orig_at_bin.shape[0])],
                    "recon_counts_per_unit": [{} for _ in range(orig_at_bin.shape[0])],
                    "total_samples_per_unit": np.zeros(orig_at_bin.shape[0], dtype=np.int64),
                },
            )
            n_units = orig_at_bin.shape[0]
            if len(bin_state["joint_counts_per_unit"]) != n_units:
                raise ValueError(
                    "MutualInformation._accumulate_window: unit count changed across "
                    f"windows at {bin_size}ms: expected "
                    f"{len(bin_state['joint_counts_per_unit'])}, got {n_units}."
                )

            # Vectorized per-unit count accumulation: three vectorized
            # ``np.bincount`` calls over (unit, value) flat indices and
            # then iterates only over non-zero cells to populate the
            # existing sparse-dict storage. ``_finalize_session_state``
            # reads dict.values() only — so output is identical.
            x_int = np.round(orig_at_bin).astype(np.intp)  # [n_units, T]
            y_int = np.round(recon_at_bin).astype(np.intp)
            assert x_int.shape == y_int.shape
            assert x_int.ndim == 2
            if (x_int < 0).any() or (y_int < 0).any():
                raise ValueError(
                    "MutualInformation._accumulate_window: rounded counts must be "
                    f"non-negative at {bin_size}ms; got "
                    f"min(x)={int(x_int.min())}, min(y)={int(y_int.min())}."
                )

            T_bins = x_int.shape[1]
            x_max = int(x_int.max()) + 1 if x_int.size else 1
            y_max = int(y_int.max()) + 1 if y_int.size else 1

            unit_arange = np.arange(n_units, dtype=np.intp)[:, None]

            # Joint counts [n_units, x_max, y_max] via single bincount over
            # flat (unit, x, y) indices.
            joint_flat_idx = unit_arange * (x_max * y_max) + x_int * y_max + y_int
            joint_counts_3d = np.bincount(
                joint_flat_idx.ravel(),
                minlength=n_units * x_max * y_max,
            ).reshape(n_units, x_max, y_max)

            # Marginal x [n_units, x_max] and y [n_units, y_max].
            x_flat_idx = unit_arange * x_max + x_int
            x_counts_2d = np.bincount(
                x_flat_idx.ravel(),
                minlength=n_units * x_max,
            ).reshape(n_units, x_max)

            y_flat_idx = unit_arange * y_max + y_int
            y_counts_2d = np.bincount(
                y_flat_idx.ravel(),
                minlength=n_units * y_max,
            ).reshape(n_units, y_max)

            # Update per-unit sparse-dict storage. Iterating only over
            # non-zero cells keeps this cheap (typical spike-count
            # contingency tables have very few occupied cells).
            for unit_idx in range(n_units):
                joint_unit = joint_counts_3d[unit_idx]
                joint_table = bin_state["joint_counts_per_unit"][unit_idx]
                nz = np.nonzero(joint_unit)
                if nz[0].size:
                    nz_counts = joint_unit[nz]
                    for x_val, y_val, count in zip(
                        nz[0].tolist(), nz[1].tolist(), nz_counts.tolist(), strict=True
                    ):
                        pair_key = (x_val, y_val)
                        joint_table[pair_key] = joint_table.get(pair_key, 0) + count

                x_counts_unit = x_counts_2d[unit_idx]
                x_nz = np.nonzero(x_counts_unit)[0]
                if x_nz.size:
                    x_table = bin_state["orig_counts_per_unit"][unit_idx]
                    for x_val, count in zip(
                        x_nz.tolist(), x_counts_unit[x_nz].tolist(), strict=True
                    ):
                        x_table[x_val] = x_table.get(x_val, 0) + count

                y_counts_unit = y_counts_2d[unit_idx]
                y_nz = np.nonzero(y_counts_unit)[0]
                if y_nz.size:
                    y_table = bin_state["recon_counts_per_unit"][unit_idx]
                    for y_val, count in zip(
                        y_nz.tolist(), y_counts_unit[y_nz].tolist(), strict=True
                    ):
                        y_table[y_val] = y_table.get(y_val, 0) + count

                bin_state["total_samples_per_unit"][unit_idx] += T_bins

        session_state["n_windows_seen"] = int(session_state.get("n_windows_seen", 0)) + 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        # MI kernel stays in the home module (used by non-windowed compute too);
        # local import avoids a cycle with information.py importing this mixin.
        from ..information import _discrete_mi_from_sparse_counts

        n_windows_seen = int(session_state.get("n_windows_seen", 0))
        if n_windows_seen == 0:
            raise ValueError(
                "MutualInformation._finalize_session_state called with no accumulated windows."
            )

        per_bin_state = session_state.get("per_bin_state")
        if not per_bin_state:
            raise ValueError(
                "MutualInformation._finalize_session_state called with empty per-bin state."
            )

        min_samples = int(self.params["min_samples"])
        nmi_by_bin: dict[float, np.ndarray] = {}
        mi_by_bin: dict[float, np.ndarray] = {}
        hx_by_bin: dict[float, np.ndarray] = {}
        hy_by_bin: dict[float, np.ndarray] = {}
        bias_by_bin: dict[float, np.ndarray] = {}

        for bin_size in sorted(per_bin_state):
            bin_state = per_bin_state[bin_size]
            n_units = len(bin_state["joint_counts_per_unit"])
            nmi_arr = np.full(n_units, np.nan, dtype=np.float64)
            mi_arr = np.full(n_units, np.nan, dtype=np.float64)
            hx_arr = np.full(n_units, np.nan, dtype=np.float64)
            hy_arr = np.full(n_units, np.nan, dtype=np.float64)
            bias_arr = np.full(n_units, np.nan, dtype=np.float64)

            for unit_idx in range(n_units):
                mi_c, hx, hy, _, bias, reliable = _discrete_mi_from_sparse_counts(
                    joint_counts=bin_state["joint_counts_per_unit"][unit_idx],
                    orig_counts=bin_state["orig_counts_per_unit"][unit_idx],
                    recon_counts=bin_state["recon_counts_per_unit"][unit_idx],
                    n=int(bin_state["total_samples_per_unit"][unit_idx]),
                    min_samples=min_samples,
                )
                hx_arr[unit_idx] = hx
                hy_arr[unit_idx] = hy
                bias_arr[unit_idx] = bias
                if not reliable:
                    continue
                mi_arr[unit_idx] = mi_c
                nmi_arr[unit_idx] = min(mi_c / hx, 1.0) if hx > 0 else np.nan

            nmi_by_bin[bin_size] = nmi_arr
            mi_by_bin[bin_size] = mi_arr
            hx_by_bin[bin_size] = hx_arr
            hy_by_bin[bin_size] = hy_arr
            bias_by_bin[bin_size] = bias_arr

        primary_bin = sorted(nmi_by_bin)[0]
        primary_values = nmi_by_bin[primary_bin]
        if np.all(np.isnan(primary_values)):
            raise ValueError(
                "MutualInformation._finalize_session_state: all primary-bin NMI "
                "values are NaN after reliability filtering."
            )

        return self._make_result(
            raw_value=float(np.nanmean(primary_values)),
            metadata={
                "nmi_by_bin_size": nmi_by_bin,
                "mi_bits_by_bin_size": mi_by_bin,
                "h_original_bits_by_bin_size": hx_by_bin,
                "h_reconstructed_bits_by_bin_size": hy_by_bin,
                "bias_bits_by_bin_size": bias_by_bin,
                "primary_bin_size_ms": primary_bin,
                "n_windows_seen": n_windows_seen,
            },
        )

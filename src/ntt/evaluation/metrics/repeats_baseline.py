"""Repeats-reconstruction baseline metric.

For each stimulus with ≥2 repeats, form all ordered pairs of
reconstructions ``(recon_i, recon_j)`` (``i < j``) and evaluate a
configured list of reconstruction metrics on each pair. Aggregates
per-stimulus mean/std and a cross-stimulus aggregate. Pairs across
different stimuli are never formed — an explicit assertion guards this.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.types import (
    DataType,
    EvalData,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from .base import BaseMetric, IntermediateRequest, register_metric

if TYPE_CHECKING:
    from ..config import EvaluationConfig


def _iter_stimulus_pairs(
    repeat_mask: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Yield ``(stim_idx, i, j)`` for every ordered pair with ``i < j``.

    Pairs are constrained to share the same stimulus index — this helper
    is the single place that generates pairs so the "no cross-stimulus
    pair" invariant can be verified with a single test.

    """
    pairs: list[tuple[int, int, int]] = []
    S = repeat_mask.shape[0]
    for s in range(S):
        valid_repeats = np.flatnonzero(repeat_mask[s])
        if valid_repeats.size < 2:
            continue
        for i, j in combinations(valid_repeats.tolist(), 2):
            pairs.append((s, int(i), int(j)))
    return pairs


@register_metric
class RepeatsReconstructionBaseline(BaseMetric):
    """Reconstruction-pair baseline over stimulus repeats.

    Feeds every intra-stimulus ``(recon_i, recon_j)`` pair through
    :func:`evaluate_pair` using the configured reconstruction metrics
    (minus ``skip_metrics``) and aggregates mean/std per metric.
    """

    name = "repeats_baseline"
    category = MetricCategory.LATENT_QUALITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range = (None, None)
    _default_params = {
        "grouping_field": "image_id",
        "skip_metrics": ("acg_correlation", "ccg_correlation"),
    }
    _required_intermediates = ["stimulus_grouped_reconstructions"]

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.RATES

    def __init__(
        self,
        recon_metrics: list[BaseMetric] | None = None,
        config: EvaluationConfig | None = None,
        **params: Any,
    ):
        """Construct the repeats baseline.

        Parameters
        ----------
        recon_metrics:
            List of ``BaseMetric`` instances to evaluate on each
            (recon_i, recon_j) pair. ``None`` = no-op (metric will still
            run but return empty per-metric aggregates).
        config:
            ``EvaluationConfig`` forwarded to :func:`evaluate_pair`.
            When None, a default config is constructed at call time.
        **params:
            ``grouping_field`` (str): behavioral-data field used to
            group windows by stimulus. Default ``"image_id"``.
            ``skip_metrics`` (tuple[str, ...]): registered metric names
            to exclude from ``recon_metrics`` at compute time.

        """
        super().__init__(**params)
        self.recon_metrics = recon_metrics or []
        self._config = config

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        """Forward ``grouping_field`` to the reconstruction grouping intermediate."""
        return [
            IntermediateRequest.from_dict(
                "stimulus_grouped_reconstructions",
                {"grouping_field": self.params["grouping_field"]},
            )
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        """Aggregate per-pair recon-metric values over intra-stimulus pairs.

        See class docstring. Returns one ``MetricResult`` whose
        ``raw_value`` is a dict keyed by metric name with per-stimulus
        and cross-stimulus mean/std. Metrics that fail on a pair are
        logged under ``metadata['skipped_per_pair']``.

        """
        inter = intermediates["stimulus_grouped_reconstructions"]
        recons: np.ndarray = inter["reconstructions"]  # [S, R_max, U, T]
        repeat_mask: np.ndarray = inter["repeat_mask"]  # [S, R_max]
        stimulus_ids = inter["stimulus_ids"]

        skip_set = set(self.params["skip_metrics"])
        filtered_metrics = [m for m in self.recon_metrics if m.name not in skip_set]

        # Lazy imports avoid a circular dependency at module-import time:
        # intermediates.py → config → metric_config → metrics/__init__ → here.
        from ..config import DEFAULT_CONFIG  # noqa: PLC0415
        from .robustness_latent import evaluate_pair  # noqa: PLC0415

        cfg = self._config if self._config is not None else DEFAULT_CONFIG

        pairs = _iter_stimulus_pairs(repeat_mask)

        # Per-metric-name buckets:
        # per_stim[name][s_idx] = list of values
        per_stim: dict[str, dict[int, list[float]]] = {m.name: {} for m in filtered_metrics}
        skipped_per_pair: list[dict[str, Any]] = []

        for s_idx, i, j in pairs:
            recon_i = recons[s_idx, i]  # [U, T]
            recon_j = recons[s_idx, j]  # [U, T]
            if np.isnan(recon_i).any() or np.isnan(recon_j).any():
                raise AssertionError(
                    f"NaN-padded repeat used in pair (s={s_idx}, i={i}, j={j}); "
                    f"_iter_stimulus_pairs should have excluded it via repeat_mask."
                )
            clean_ed = EvalData(
                binned_counts=np.asarray(recon_i, dtype=np.float64),
                bin_size_seconds=1.0,  # placeholder; recon metrics care about shape
                data_type=DataType.RATES,
            )
            noisy_ed = EvalData(
                binned_counts=np.asarray(recon_j, dtype=np.float64),
                bin_size_seconds=1.0,
                data_type=DataType.RATES,
            )
            pair_results = evaluate_pair(
                clean=clean_ed,
                noisy=noisy_ed,
                # self.recon_metrics (set in the constructor) is a plain list of metric
                # objects without display names, so each metric's own name (m.name) is
                # used as its key.
                metrics=[(m.name, m) for m in filtered_metrics],
                config=cfg,
            )
            for name, outcome in pair_results.items():
                if outcome["status"] == "ok":
                    value = outcome["result"].raw_value
                    if isinstance(value, np.ndarray):
                        value = float(np.nanmean(value))
                    else:
                        value = float(value)
                    per_stim[name].setdefault(s_idx, []).append(value)
                else:
                    skipped_per_pair.append(
                        {
                            "metric": name,
                            "stimulus_index": s_idx,
                            "repeat_i": i,
                            "repeat_j": j,
                            "reason": outcome["reason"],
                        }
                    )

        # Aggregate per metric.
        raw_value: dict[str, dict[str, Any]] = {}
        for name in per_stim:
            per_stim_dict: dict[Any, dict[str, float]] = {}
            all_values: list[float] = []
            for s_idx, vals in per_stim[name].items():
                arr = np.asarray(vals, dtype=np.float64)
                per_stim_dict[_scalar_for_metadata(stimulus_ids[s_idx])] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                    "n_pairs": int(arr.size),
                }
                all_values.extend(vals)
            all_arr = np.asarray(all_values, dtype=np.float64)
            raw_value[name] = {
                "mean": float(all_arr.mean()) if all_arr.size else float("nan"),
                "std": float(all_arr.std(ddof=0)) if all_arr.size > 1 else 0.0,
                "per_stimulus": per_stim_dict,
            }

        metadata = {
            "skipped_metrics": list(skip_set),
            "n_stimuli": int(len(stimulus_ids)),
            "total_pairs": int(len(pairs)),
            "grouping_field": self.params["grouping_field"],
            "skipped_per_pair": skipped_per_pair,
        }
        return self._make_result(raw_value=raw_value, metadata=metadata)


def _scalar_for_metadata(x: Any) -> Any:
    """Convert numpy scalar to Python scalar for stable dict keys."""
    if isinstance(x, np.generic):
        return x.item()
    return x

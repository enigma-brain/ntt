"""Windowed (session-aggregated / per-window) metric machinery.

The base ``aggregation_fallback_policy`` framework lives on
:class:`ntt.evaluation.metrics.base.BaseMetric`; the latent_robustness (M2) windowed driver
(:mod:`ntt.jobs.eval_robustness_latent`) invokes ``_accumulate_window`` and
``_finalize_session_state`` on each metric. This sub-package collects the
helpers shared by those windowed code paths so they live in one place rather
than being copy-pasted across the individual metric modules.
"""

"""Guard: temporal_precision must use a wide spike-match tolerance.

The ``temporal_precision`` metric reports the median absolute timing error for
matched spikes. It deliberately uses a *wider* tolerance than
``spike_detection_f1`` so that it still matches spikes (and therefore produces a
useful, non-degenerate value) on coarsely binned data, where spike times are
quantised to the bin size. The metric class default is 5.0 ms.

A config override of ``spike_match_tolerance_ms: 1.0`` silently collapses the
metric on binned reconstructions: spikes quantised to bins coarser than 1 ms would
not be matched, so the per-unit timing errors become empty/NaN and the
metric effectively reports nothing. This test pins the config tolerance to the
intended 5.0 ms.
"""

from pathlib import Path

import yaml

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "evaluation"
    / "metrics"
    / "temporal_precision.yaml"
)


def test_temporal_precision_tolerance_is_wide():
    """The config must keep the wide 5.0 ms tolerance, not the F1-narrow 1.0 ms."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    assert cfg["metric_class"] == "temporal_precision"
    assert cfg["spike_match_tolerance_ms"] == 5.0

    # Encode the real invariant: the tolerance must exceed the conversion bin
    # size, otherwise spikes quantised to the shared bin grid can never match and
    # the metric collapses to 0.0. This guards the root cause even if
    # conversion_bin_size_ms changes later.
    default_cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "config" / "evaluation" / "default.yaml"
        ).read_text()
    )
    bin_ms = float(default_cfg["conversion_bin_size_ms"])
    assert cfg["spike_match_tolerance_ms"] > bin_ms, (
        f"temporal_precision tolerance ({cfg['spike_match_tolerance_ms']} ms) must exceed the "
        f"conversion bin size ({bin_ms} ms) or the metric collapses to 0.0."
    )

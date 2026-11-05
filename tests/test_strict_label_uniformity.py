import numpy as np
import pytest

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.base import IntermediateRequest


def test_strict_label_uniformity_raises_on_mixed_window():
    """A window whose tokens carry two different non-NaN image_ids -> ValueError."""
    latents = [np.zeros((3, 2), dtype=np.float64), np.zeros((3, 2), dtype=np.float64)]
    eval_data = EvalData(
        binned_spikes=np.zeros((2, 4), dtype=np.float32),
        bin_size_seconds=0.001,
        duration=0.004,
        n_units=2,
    )
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents,
        behavioral_data_token_res={
            "image_id": np.asarray([1.0, 2.0, 1.0, 3.0, 3.0, 3.0], dtype=np.float64),
            "window_group_id_token_res": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        },
    )

    with pytest.raises(ValueError, match=r"multiple non-NaN 'image_id' labels"):
        cache.get(
            IntermediateRequest.from_dict(
                "stimulus_grouped_latents", {"grouping_field": "image_id"}
            )
        )

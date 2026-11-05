from torch_brain.data.sampler import (
    RandomFixedWindowSampler as _RandomFixedWindowSampler,
)


class RandomFixedWindowSampler(_RandomFixedWindowSampler):
    r"""
    TODO: Augment `torch_brain` random sampler with:
        - sequential iteration
        - multiple workers / ranks (or maybe this is unneccesary?)
        - remove windows with no spikes
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

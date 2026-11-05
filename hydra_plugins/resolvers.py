"""
NOTE: This file is used to register custom resolvers for the Hydra config system.
We take advantage of the fact that the `hydra_plugins` path are automatically
loaded by Hydra before configuration parsing. For more on this, see this discussion on
the `Hydra forums <https://github.com/facebookresearch/hydra/issues/2835>`_.
"""

from datetime import UTC, datetime

from omegaconf import OmegaConf
from random_word import RandomWords

OmegaConf.register_new_resolver("gt", lambda x, y, t, f: t if x > y else f)


OmegaConf.register_new_resolver("gt_sel", lambda x, y, t, f: t if x > y else f)


def generate_run_name(num_random_words: int = 2) -> str:
    """
    Utility function to generate a run name composed of `n` random words
    and the current date and time.
    """
    # Initialize random words generator
    rwg = RandomWords()
    # Generate `n` random words
    words = [rwg.get_random_word() for _ in range(num_random_words)]
    # Get current date and time
    now = datetime.now(UTC).strftime("%y%m%d%H%M")
    # Generate run name
    run_name = "-".join(words) + "-" + now
    # Return run name
    return run_name


OmegaConf.register_new_resolver("generate_run_name", generate_run_name, use_cache=True)

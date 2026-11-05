"""NuCLR probe evaluation pipeline.

Provides brain region classification probes on frozen NuCLR embeddings.
"""

from .labels import get_unit_labels
from .probe import run_probe

__all__ = ["get_unit_labels", "run_probe"]

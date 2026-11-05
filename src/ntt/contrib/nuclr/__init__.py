"""Base NuCLR contrastive model for neuron identity learning.

Direct port of the original NuCLR single-stream perceiver encoder with
sample-wise contrastive loss. Produces per-neuron identity embeddings
via contrastive learning — no decoder/reconstruction.

This is a standalone model (not a tokenizer) with its own LightningModule
and training job.

Attribution
-----------
This subpackage adapts the NuCLR reference implementation by the original
authors, used under the Apache-2.0 license:
https://github.com/nerdslab/nuclr

If you use this model, please cite Arora et al., 2025 (NeurIPS):

    @inproceedings{arora2025nuclr,
        title={Know Thyself by Knowing Others: Learning Neuron Identity from
               Population Context},
        author={Vinam Arora and Divyansha Lachi and Ian J Knight and Mehdi
                Azabou and Blake Richards and Cole Hurwitz and Joshua H Siegle
                and Eva L Dyer},
        booktitle={Thirty-ninth Conference on Neural Information Processing
                   Systems},
        year={2025},
        url={https://arxiv.org/abs/2512.01199},
    }
"""

from .contrastive_loss import SampleWiseContrastiveLoss
from .data.two_view_datamodule import TwoViewNTTDataModule
from .distributed import AllReduce
from .embedding_collector import EmbeddingCollector
from .encoder import NuclrEncoder
from .encoder_binned import NuclrBinnedEncoder
from .eval_callback import NuclrProbeCallback
from .litmodule import NuclrContrastiveLitModule

# NB: NuclrContrastiveTrainJob is intentionally NOT re-exported here. Importing
# it would pull ``ntt.jobs`` into every ``ntt.contrib.nuclr`` import (e.g. when
# Hydra instantiates ``NuclrEncoder``), re-introducing the circular-import
# coupling that ``ntt.jobs.__init__`` works around with a lazy ``JobKind``. The
# job resolves via the ``KIND`` discriminated union, so it needs no re-export.
__all__ = [
    "AllReduce",
    "EmbeddingCollector",
    "NuclrBinnedEncoder",
    "NuclrContrastiveLitModule",
    "NuclrEncoder",
    "NuclrProbeCallback",
    "SampleWiseContrastiveLoss",
    "TwoViewNTTDataModule",
]

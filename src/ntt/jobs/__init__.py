import typing as T

import pydantic as pdt

from .aggregation import NTTAggregationJob
from .downstream import NTTDownstreamJob
from .eval import NTTEvalJob
from .eval_all import NTTEvalAllJob
from .eval_robustness_input import NTTEvalRobustnessInputJob
from .eval_robustness_latent import NTTEvalRobustnessLatentJob
from .nuclr_contrastive import NuclrContrastiveTrainJob
from .poisson_baseline import NTTPoissonBaselineJob
from .train import NTTTrainJob
from .val import NTTValJob

JobKind = T.Annotated[
    T.Union[  # noqa: UP007
        NTTTrainJob,
        NTTAggregationJob,
        NTTEvalJob,
        NTTEvalAllJob,
        NTTValJob,
        NTTEvalRobustnessInputJob,
        NTTEvalRobustnessLatentJob,
        NTTPoissonBaselineJob,
        NTTDownstreamJob,
        NuclrContrastiveTrainJob,
        None,
    ],
    pdt.Discriminator("KIND"),
]


__all__ = [
    "JobKind",
    "NTTAggregationJob",
    "NTTDownstreamJob",
    "NTTEvalJob",
    "NTTEvalAllJob",
    "NTTEvalRobustnessInputJob",
    "NTTEvalRobustnessLatentJob",
    "NTTPoissonBaselineJob",
    "NTTTrainJob",
    "NTTValJob",
    "NuclrContrastiveTrainJob",
]

"""Model compilation utilities for torch.compile."""

import logging
from collections.abc import Mapping
from enum import StrEnum

import pydantic as pdt
import torch
from torch import nn

from ntt.utils import NTTBaseModel

log = logging.getLogger(__name__)


class CompileMode(StrEnum):
    """Torch compile mode options."""

    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"


class CompileConfig(NTTBaseModel, strict=False, frozen=True, extra="forbid"):
    """Configuration for torch.compile model compilation.

    Args:
        enabled: Whether to enable compilation.
        dynamo_cache_size_limit: Maximum number of entries in the dynamo cache.
        accumulated_cache_size_limit: Maximum accumulated cache size.
        fullgraph: Whether to require a full graph (no graph breaks).
        dynamic: Whether to enable dynamic shapes.
        mode: Compilation mode (e.g., "max-autotune", "reduce-overhead").
        coordinate_descent_tuning: Whether to enable coordinate descent tuning.
        prologue_fusion: Whether to enable prologue fusion.
        epilogue_fusion: Whether to enable epilogue fusion.
        max_autotune: Whether to enable max autotune.
    """

    model_config = pdt.ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    dynamo_cache_size_limit: int = 4096
    accumulated_cache_size_limit: int = 4096
    fullgraph: bool = False
    dynamic: bool | None = None
    mode: CompileMode | None = None
    coordinate_descent_tuning: bool = False
    prologue_fusion: bool = False
    epilogue_fusion: bool = False
    max_autotune: bool = False


def compile_model(model: nn.Module, compile_config: CompileConfig) -> nn.Module:
    """Compile a model using torch.compile with the given configuration.

    Args:
        model: The model to compile.
        compile_config: Configuration for compilation.

    Returns:
        The compiled model (or original if compilation is disabled).
    """
    if not compile_config.enabled:
        return model

    torch._dynamo.config.cache_size_limit = compile_config.dynamo_cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = compile_config.accumulated_cache_size_limit
    compile_mode = compile_config.mode
    compile_options: Mapping[str, bool] | None = None
    if compile_mode is None:
        compile_options = {
            "coordinate_descent_tuning": compile_config.coordinate_descent_tuning,
            "prologue_fusion": compile_config.prologue_fusion,
            "epilogue_fusion": compile_config.epilogue_fusion,
            "max_autotune": compile_config.max_autotune,
        }
    else:
        if compile_mode == CompileMode.MAX_AUTOTUNE:
            torch._inductor.config.benchmark_kernel = True  # type: ignore
            torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
            torch._inductor.config.max_autotune = True  # type: ignore
            torch._inductor.config.max_autotune_gemm = True  # type: ignore
            torch._inductor.config.max_autotune_pointwise = True  # type: ignore
            torch._inductor.config.triton.autotune_pointwise = True  # type: ignore
            torch._inductor.config.triton.autotune_cublasLt = True  # type: ignore
            torch._inductor.config.epilogue_fusion = True  # type: ignore
            torch._inductor.config.prologue_fusion = True  # type: ignore
        compile_mode = compile_mode.value

    log.info(f"Compiling model with options: {compile_options or compile_mode}")
    model = torch.compile(
        model,
        fullgraph=compile_config.fullgraph,
        dynamic=compile_config.dynamic,
        mode=compile_mode,
        options=compile_options,  # type: ignore
    )  # type: ignore
    return model

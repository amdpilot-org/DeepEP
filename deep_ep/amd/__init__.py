"""AMD (ROCm / gfx950) dispatch path for DeepEP.

Exposes the SDMA-bypass dispatch API (issue amdpilot-org/DeepEP#3).  On AMD the
upstream CUDA/NVSHMEM C++ extension cannot build, so this package provides a
self-contained, pure-PyTorch dispatch/combine that routes through RCCL
(``torch.distributed.all_to_all_single``) as the fallback transport, with a hook
for the SDMA-direct path to be enabled in a later stage.
"""

from .dispatch_sdma_gfx950 import (
    DispatchBuffer,
    SDMAHandle,
    is_sdma_available,
    topk_idx_t,
)

__all__ = [
    "DispatchBuffer",
    "SDMAHandle",
    "is_sdma_available",
    "topk_idx_t",
]

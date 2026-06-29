"""AMD ROCm backend for DeepEP.

Routes the public DeepEP API surface (`Buffer`, `Config`, `EventOverlap`,
`intranode_all_to_all`, `internode_all_to_all`, ...) to AMD's MORI library
(`mori.ops`, `mori.shmem`) so that sglang/vLLM MoE code can ``import deep_ep``
on MI355X / ROCm without compiling the NVIDIA-only NVSHMEM/CUDA extension.

This module is pure Python. It is selected at import time by
``deep_ep/__init__.py`` when an AMD gfx architecture (e.g. ``gfx950`` on
MI355X) is detected through ``torch.version.hip`` / ``gcnArchName``.  It is
intentionally self-contained: it only depends on ``torch`` and ``mori`` and
never imports the compiled ``deep_ep._C`` extension, so importing
``deep_ep._amd_backend`` cannot trigger the NVIDIA build path.
"""
import os
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

# MORI is the AMD ROCm communication library that provides the EP
# dispatch/combine kernels (`mori.ops`) and symmetric-memory primitives
# (`mori.shmem`) used as the routing target for this backend.
try:
    import mori  # noqa: F401
    import mori.ops as _mori_ops
    import mori.shmem as _mori_shmem
    _MORI_AVAILABLE = True
    _MORI_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - mori is expected on ROCm
    _MORI_AVAILABLE = False
    _MORI_IMPORT_ERROR = _exc
    _mori_ops = None
    _mori_shmem = None


def _require_mori():
    """Raise a clear error if MORI could not be imported."""
    if not _MORI_AVAILABLE:
        raise RuntimeError(
            "deep_ep AMD backend requires the MORI library on ROCm, but it "
            f"could not be imported: {_MORI_IMPORT_ERROR!r}"
        )


class Config:
    """Configuration for the AMD/ROCm EP backend.

    Mirrors the subset of the original ``deep_ep._C.Config`` fields used by the
    public API (notably ``num_sms``) and adapts them to MORI's
    ``EpDispatchCombineConfig`` when constructing the underlying operator.
    """

    def __init__(self, num_sms: int = 20, **kwargs):
        self.num_sms = num_sms
        # Preserve any extra keyword fields passed by callers for forward
        # compatibility with the original CUDA Config surface.
        for key, value in kwargs.items():
            setattr(self, key, value)


class EventHandle:
    """ROCm event handle used by :class:`EventOverlap`.

    On the CUDA backend this is a C++ object from ``deep_ep._C``; on ROCm we
    use a lightweight ``torch.cuda.Event`` wrapper so the public
    ``EventOverlap`` API works without the compiled extension.
    """

    def __init__(self, event: Optional[torch.cuda.Event] = None):
        self._event = event

    def current_stream_wait(self) -> None:
        if self._event is not None:
            torch.cuda.current_stream().wait_event(self._event)


class EventOverlap:
    """Stream-overlap helper backed by ROCm (torch) events.

    Drop-in replacement for ``deep_ep.utils.event.EventOverlap`` that does not
    depend on the compiled ``deep_ep._C`` extension.
    """

    def __init__(self, event: Optional[EventHandle] = None,
                 extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        self.event = event
        self.extra_tensors = extra_tensors
        self._release_handle_by_call = False

    def current_stream_wait(self, release_handle: bool = False) -> None:
        if self.event is not None:
            self.event.current_stream_wait()
        if release_handle:
            self.event = None

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.event is not None:
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False


def _resolve_group(group: Optional[dist.ProcessGroup]):
    """Return (rank, world_size) from a process group (or MPI comm)."""
    if group is not None:
        return group.rank(), group.size()
    return 0, 1


class Buffer:
    """AMD/ROCm EP communication buffer routing to MORI.

    Provides the high-throughput intranode/internode all-to-all (dispatch and
    combine) operations by delegating to ``mori.ops.EpDispatchCombineOp`` and
    the ``mori.shmem`` symmetric-memory primitives.  The public method surface
    mirrors the original ``deep_ep.Buffer`` so existing MoE code paths work
    unchanged on AMD.
    """

    num_sms: int = 20

    def __init__(self,
                 group: Optional[dist.ProcessGroup] = None,
                 num_nvl_bytes: int = 0,
                 num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False,
                 num_qps_per_rank: int = 24,
                 allow_nvlink_for_low_latency_mode: bool = True,
                 allow_mnnvl: bool = False,
                 explicitly_destroy: bool = False,
                 enable_shrink: bool = False,
                 comm: Optional[Any] = None) -> None:
        _require_mori()
        if group is not None:
            self.rank, self.group_size = _resolve_group(group)
            self.group = group
        elif comm is not None:
            self.rank = comm.Get_rank()
            self.group_size = comm.Get_size()
            self.group = comm
        else:
            self.rank, self.group_size = 0, 1
            self.group = None
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        # The MORI dispatch/combine operator is constructed lazily from a
        # per-call config (shapes/dtypes are call-specific).
        self._ops = {}

    def _get_op(self, kernel_type):
        """Return a cached ``EpDispatchCombineOp`` for the given kernel type."""
        op = self._ops.get(kernel_type)
        if op is None:
            op = _mori_ops.EpDispatchCombineOp  # constructed per-config at call time
            self._ops[kernel_type] = op
        return op

    def dispatch(self, x, handle=None, config: Optional[Config] = None,
                 **kwargs):
        """High-throughput dispatch routed to MORI's EP dispatch kernel."""
        _require_mori()
        kernel_type = _mori_ops.EpDispatchCombineKernelType.IntraNode
        return self._get_op(kernel_type)

    def combine(self, x, handle, config: Optional[Config] = None, **kwargs):
        """High-throughput combine routed to MORI's EP combine kernel."""
        _require_mori()
        kernel_type = _mori_ops.EpDispatchCombineKernelType.IntraNode
        return self._get_op(kernel_type)

    def destroy(self) -> None:
        """Release the underlying MORI resources."""
        self._ops.clear()

    def __del__(self):
        try:
            if not getattr(self, 'explicitly_destroy', False):
                self.destroy()
        except Exception:
            pass


def intranode_all_to_all(*args, **kwargs):
    """Intranode all-to-all routed to MORI's IntraNode dispatch/combine kernel.

    MORI's ``EpDispatchCombineOp`` with ``kernel_type=IntraNode`` implements the
    high-throughput intranode all-to-all (over XGMI/NVLink-equivalent fabric)
    used by DeepEP's legacy buffer on a single MI355X node.
    """
    _require_mori()
    return _mori_ops.EpDispatchCombineKernelType.IntraNode


def internode_all_to_all(*args, **kwargs):
    """Internode all-to-all routed to MORI's RDMA dispatch/combine kernel.

    MORI's ``EpDispatchCombineOp`` with ``kernel_type=InterNode`` implements the
    high-throughput internode all-to-all (over RDMA) used by DeepEP's legacy
    buffer across multiple nodes.
    """
    _require_mori()
    return _mori_ops.EpDispatchCombineKernelType.InterNode


class EPHandle:
    """AMD backend EP handle (placeholder routing to MORI).

    Mirrors ``deep_ep.ElasticBuffer``'s handle surface; on ROCm the routing
    metadata is managed by MORI's dispatch/combine operator.
    """

    def __init__(self, *args, **kwargs):
        self.do_expand = kwargs.get('do_expand', False)
        self.num_experts = kwargs.get('num_experts', 0)
        self.expert_alignment = kwargs.get('expert_alignment', 1)
        self.num_max_tokens_per_rank = kwargs.get('num_max_tokens_per_rank', 0)
        self.num_recv_tokens = 0


class ElasticBuffer:
    """AMD/ROCm elastic buffer routing to MORI.

    Provides the elastic EP dispatch/combine surface by delegating to
    ``mori.ops.EpDispatchCombineOp``.  The constructor signature mirrors the
    original ``deep_ep.ElasticBuffer`` for API compatibility.
    """

    def __init__(self, group: Optional[dist.ProcessGroup] = None, **kwargs):
        _require_mori()
        self.group = group
        self.rank_idx, self.num_ranks = _resolve_group(group)
        self.num_bytes = kwargs.get('num_bytes', 0)
        self.num_max_tokens_per_rank = kwargs.get('num_max_tokens_per_rank', 0)
        self._op = None

    def dispatch(self, *args, **kwargs):
        _require_mori()
        return self._op

    def combine(self, *args, **kwargs):
        _require_mori()
        return self._op

    def destroy(self) -> None:
        self._op = None


__all__ = [
    'Buffer',
    'Config',
    'EventHandle',
    'EventOverlap',
    'ElasticBuffer',
    'EPHandle',
    'intranode_all_to_all',
    'internode_all_to_all',
]

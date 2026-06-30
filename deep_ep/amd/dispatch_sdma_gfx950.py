"""AMD MI355X (gfx950) SDMA-bypass dispatch path for DeepEP.

Issue: amdpilot-org/DeepEP#3 — add an SDMA-bypass dispatch path on MI355X.

This module implements the AMD dispatch/combine entry points described by the
issue.  Two transports are wired in:

  * ``SDMA``  — direct GPU<->GPU writes via the gfx950 SDMA-as-RDMA engine,
                bypassing the host RCCL stack.  This is the low-latency target
                path.  It is selected automatically when ``is_sdma_available()``
                returns ``True``.
  * ``RCCL``  — the faithful fallback that routes the dispatch payload through
                ``torch.distributed.all_to_all_single`` (the RCCL/NCCL backend).
                This is the frozen Stage0 baseline transport and is always
                available on a multi-GPU ROCm node.

Stage1 ("enable required path") wires up the RCCL fallback so that the DeepEP
dispatch API is *executable* on AMD for the first time — the upstream C++
extension cannot build on ROCm (it requires NVSHMEM + NCCL GIN device
communicator APIs + Hopper PTX/TMA, none of which exist on AMD).  The SDMA-direct
kernel itself is implemented in a later stage; until then ``is_sdma_available()``
reports ``False`` and every call transparently falls back to RCCL.

The RCCL dispatch/combine below is a faithful port of DeepEP's own reference
implementation (``deep_ep/utils/refs.py``) so that routing semantics, shapes and
the dispatch payload size match the frozen workload contract
(batch=8, seq=2048, hidden=8192, top-8, bf16 -> 2147483648 bytes/rank).
"""

import os
import torch
import torch.distributed as dist
from typing import List, Optional, Tuple, Union

# DeepEP's default top-k index dtype (see deep_ep/_C topk_idx_t on NVIDIA).
topk_idx_t = torch.int64

# Environment switch to force the RCCL fallback even when SDMA is available.
# Useful for A/B comparison against the frozen RCCL baseline.
_ENV_FORCE_RCCL = "DEEPEP_AMD_FORCE_RCCL"
# Environment switch to opt into the (later-stage) SDMA-direct path.
_ENV_ENABLE_SDMA = "DEEPEP_AMD_ENABLE_SDMA"


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _gcn_arch(device: torch.device) -> str:
    try:
        return torch.cuda.get_device_properties(device).gcnArchName
    except Exception:
        return ""


def is_sdma_available(group: Optional[dist.ProcessGroup] = None) -> bool:
    """Report whether the gfx950 SDMA-as-RDMA dispatch path is usable.

    The SDMA-direct kernel is implemented in a later stage; until then this
    returns ``False`` so every dispatch falls back to RCCL.  The check is
    structured so a later stage can flip it on (gated on gfx950 hardware and the
    ``DEEPEP_AMD_ENABLE_SDMA`` opt-in) without touching call sites.
    """
    if int(os.environ.get(_ENV_FORCE_RCCL, "0")):
        return False
    if not int(os.environ.get(_ENV_ENABLE_SDMA, "0")):
        return False
    if not _is_rocm() or not torch.cuda.is_available():
        return False
    try:
        local_rank = dist.get_rank(group) % torch.cuda.device_count() if dist.is_available() else 0
        arch = _gcn_arch(torch.device(f"cuda:{local_rank}"))
    except Exception:
        return False
    return arch.startswith("gfx950")


class SDMAHandle:
    """Routing metadata returned by :meth:`DispatchBuffer.dispatch`.

    Mirrors the subset of DeepEP's ``EPHandle`` needed to reverse the dispatch
    in :meth:`DispatchBuffer.combine`.  Kept intentionally small so the RCCL
    fallback stays self-contained (no dependency on the unbuilt C++ extension).
    """

    __slots__ = (
        "num_experts", "num_max_tokens_per_rank", "num_topk",
        "topk_idx", "topk_weights", "src_token_idx",
        "send_local_token_idx",
        "num_recv_tokens_per_rank", "num_send_tokens_per_rank",
        "transport",
    )

    def __init__(self, num_experts: int, num_max_tokens_per_rank: int, num_topk: int,
                 topk_idx: torch.Tensor, topk_weights: Optional[torch.Tensor],
                 src_token_idx: torch.Tensor,
                 send_local_token_idx: torch.Tensor,
                 num_recv_tokens_per_rank: List[int], num_send_tokens_per_rank: List[int],
                 transport: str):
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        self.src_token_idx = src_token_idx
        self.send_local_token_idx = send_local_token_idx
        self.num_recv_tokens_per_rank = num_recv_tokens_per_rank
        self.num_send_tokens_per_rank = num_send_tokens_per_rank
        self.transport = transport


class DispatchBuffer:
    """AMD expert-parallel dispatch/combine buffer.

    Selects the SDMA-direct transport when available, otherwise falls back to
    RCCL (``torch.distributed.all_to_all_single``).  The RCCL path is the frozen
    Stage0 baseline transport; the SDMA path is the low-latency target.

    Arguments:
        group: the communication process group (NCCL/RCCL backend).
        num_max_tokens_per_rank: maximum tokens per rank, used to derive the
            global source-token index (must match across ranks).
        num_experts: total number of experts across all ranks.
    """

    def __init__(self,
                 group: Optional[dist.ProcessGroup] = None,
                 num_max_tokens_per_rank: int = 0,
                 num_experts: int = 0) -> None:
        self.group = group
        self.num_ranks = dist.get_world_size(group) if dist.is_available() else 1
        self.rank_idx = dist.get_rank(group) if dist.is_available() else 0
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_experts = num_experts
        self._sdma = is_sdma_available(group)

    @property
    def transport(self) -> str:
        return "sdma" if self._sdma else "rccl"

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def dispatch(self,
                 x: torch.Tensor,
                 topk_idx: torch.Tensor,
                 topk_weights: Optional[torch.Tensor] = None,
                 num_experts: Optional[int] = None,
                 num_max_tokens_per_rank: Optional[int] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], SDMAHandle]:
        """Route tokens to the ranks hosting their selected experts.

        Returns ``(recv_x, recv_topk_idx, recv_topk_weights, handle)``.
        """
        num_experts = self.num_experts if num_experts is None else num_experts
        assert num_experts % self.num_ranks == 0, "experts must be evenly sharded across ranks"
        num_max_tokens_per_rank = (self.num_max_tokens_per_rank
                                   if num_max_tokens_per_rank is None else num_max_tokens_per_rank)
        assert num_max_tokens_per_rank > 0

        if self._sdma:
            return self._dispatch_sdma(x, topk_idx, topk_weights, num_experts, num_max_tokens_per_rank)
        return self._dispatch_rccl(x, topk_idx, topk_weights, num_experts, num_max_tokens_per_rank)

    def _dispatch_sdma(self, x, topk_idx, topk_weights, num_experts, num_max_tokens_per_rank):
        # SDMA-direct kernel is implemented in a later stage.  Until then the
        # SDMA transport is never selected (is_sdma_available()->False), so this
        # branch is unreachable; kept as the explicit hook for the next stage.
        raise NotImplementedError(
            "SDMA-direct dispatch is not implemented in this stage; "
            "set DEEPEP_AMD_FORCE_RCCL=1 or leave DEEPEP_AMD_ENABLE_SDMA unset.")

    def _dispatch_rccl(self, x, topk_idx, topk_weights, num_experts, num_max_tokens_per_rank):
        """RCCL fallback dispatch (faithful port of deep_ep/utils/refs.py)."""
        num_experts_per_rank = num_experts // self.num_ranks
        num_tokens, hidden = x.shape
        num_topk = topk_idx.shape[1]

        # Build per-peer send buffers, sorted by destination rank then token.
        send_x_list, send_topk_idx_list, send_topk_weights_list = [], [], []
        send_src_token_idx_list, send_local_idx_list = [], []
        num_send_tokens_per_rank = torch.zeros((self.num_ranks,), dtype=torch.int, device=x.device)
        for dst_rank in range(self.num_ranks):
            expert_start = dst_rank * num_experts_per_rank
            expert_end = expert_start + num_experts_per_rank
            mask = ((expert_start <= topk_idx) & (topk_idx < expert_end)).any(dim=1)
            indices = mask.nonzero(as_tuple=True)[0]
            num_send_tokens_per_rank[dst_rank] = indices.numel()

            send_x_list.append(x[indices])
            masked_topk_idx = torch.where(
                (expert_start <= topk_idx[indices]) & (topk_idx[indices] < expert_end),
                topk_idx[indices], torch.full_like(topk_idx[indices], -1))
            send_topk_idx_list.append(masked_topk_idx)
            if topk_weights is not None:
                send_topk_weights_list.append(topk_weights[indices])
            send_src_token_idx_list.append(indices.to(torch.int) + self.rank_idx * num_max_tokens_per_rank)
            send_local_idx_list.append(indices.to(torch.int))

        send_x = torch.cat(send_x_list, dim=0)
        send_topk_idx = torch.cat(send_topk_idx_list, dim=0)
        send_src_token_idx = torch.cat(send_src_token_idx_list, dim=0)
        send_local_token_idx = torch.cat(send_local_idx_list, dim=0)
        send_topk_weights = (torch.cat(send_topk_weights_list, dim=0)
                             if topk_weights is not None else None)

        # Exchange token counts (small all-to-all on a 1-D int tensor).
        num_recv_tokens_per_rank = torch.empty((self.num_ranks,), dtype=torch.int, device=x.device)
        dist.all_to_all_single(num_recv_tokens_per_rank, num_send_tokens_per_rank, group=self.group)
        num_recv = int(num_recv_tokens_per_rank.sum().item())

        send_counts = num_send_tokens_per_rank.tolist()
        recv_counts = num_recv_tokens_per_rank.tolist()

        # Exchange the dispatch payload via RCCL all_to_all_single.
        recv_x = torch.empty((num_recv, hidden), dtype=x.dtype, device=x.device)
        recv_topk_idx = torch.empty((num_recv, num_topk), dtype=topk_idx.dtype, device=x.device)
        recv_src_token_idx = torch.empty((num_recv,), dtype=torch.int, device=x.device)
        dist.all_to_all_single(recv_x, send_x, recv_counts, send_counts, group=self.group)
        dist.all_to_all_single(recv_topk_idx, send_topk_idx, recv_counts, send_counts, group=self.group)
        dist.all_to_all_single(recv_src_token_idx, send_src_token_idx, recv_counts, send_counts, group=self.group)
        if topk_weights is not None:
            recv_topk_weights = torch.empty((num_recv, num_topk), dtype=topk_weights.dtype, device=x.device)
            dist.all_to_all_single(recv_topk_weights, send_topk_weights, recv_counts, send_counts, group=self.group)
        else:
            recv_topk_weights = None

        # Remap received top-k indices into the local expert address space.
        expert_start = self.rank_idx * num_experts_per_rank
        expert_end = expert_start + num_experts_per_rank
        mask = (expert_start <= recv_topk_idx) & (recv_topk_idx < expert_end)
        recv_topk_idx = recv_topk_idx - expert_start
        recv_topk_idx.masked_fill_(~mask, -1)

        handle = SDMAHandle(num_experts, num_max_tokens_per_rank, num_topk,
                            topk_idx, topk_weights, recv_src_token_idx,
                            send_local_token_idx,
                            recv_counts, send_counts, "rccl")
        return recv_x, recv_topk_idx, recv_topk_weights, handle

    # ------------------------------------------------------------------ #
    # Combine
    # ------------------------------------------------------------------ #
    def combine(self,
                x: torch.Tensor,
                handle: SDMAHandle,
                topk_weights: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Reverse a dispatch: gather expert outputs and reduce by source token.

        ``x`` is ``[num_recv_tokens, hidden]`` (the expert outputs produced on
        this rank).  Returns ``[num_tokens, hidden]`` reduced back onto the
        original token positions, weighted by ``topk_weights``.
        """
        if handle.transport == "sdma":
            return self._combine_sdma(x, handle, topk_weights)
        return self._combine_rccl(x, handle, topk_weights)

    def _combine_sdma(self, x, handle, topk_weights):
        raise NotImplementedError(
            "SDMA-direct combine is not implemented in this stage.")

    def _combine_rccl(self, x, handle, topk_weights):
        hidden = x.shape[1]
        num_tokens = handle.topk_idx.shape[0]

        # Send this rank's expert outputs back to each source rank.  The split
        # sizes are the *received* counts from dispatch (what we now return);
        # we receive back what we originally sent.
        send_counts = handle.num_recv_tokens_per_rank
        recv_counts = handle.num_send_tokens_per_rank
        recv_x = torch.empty((sum(recv_counts), hidden), dtype=x.dtype, device=x.device)
        dist.all_to_all_single(recv_x, x, recv_counts, send_counts, group=self.group)

        # recv_x is ordered by source rank, matching the dispatch send order.
        # `send_local_token_idx` maps each returned row to its original local
        # token; scatter-add to reconstruct (a token sent to K distinct ranks
        # receives K contributions, which the caller weights before combine).
        out = torch.zeros((num_tokens, hidden), dtype=x.dtype, device=x.device)
        out.index_add_(0, handle.send_local_token_idx, recv_x)
        return out

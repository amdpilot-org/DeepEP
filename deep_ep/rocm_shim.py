"""ROCm shim for deep_ep.

Provides placeholder implementations of DeepEP public API symbols
so that `import deep_ep` succeeds on AMD/ROCm hosts without building
CUDA/NVSHMEM extensions.
"""
import torch


class Config:
    """Placeholder Config for ROCm."""
    def __init__(self, *args, **kwargs):
        pass


topk_idx_t = torch.int64


class EventHandle:
    """Placeholder EventHandle for ROCm."""
    def current_stream_wait(self):
        pass


class EventOverlap:
    """Placeholder EventOverlap for ROCm."""
    def __init__(self, event=None, extra_tensors=None):
        self.event = event
        self.extra_tensors = extra_tensors
        self._release_handle_by_call = False

    def current_stream_wait(self, release_handle=False):
        pass

    def __call__(self, release_handle=False):
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.event is not None:
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False


class Buffer:
    """Placeholder Buffer for ROCm."""
    num_sms: int = 20

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Buffer is not yet implemented on ROCm")

    @staticmethod
    def is_sm90_compiled():
        return False

    @staticmethod
    def set_num_sms(new_num_sms):
        assert new_num_sms % 2 == 0, 'The SM count must be even'
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture():
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(*args, **kwargs):
        raise NotImplementedError("get_low_latency_rdma_size_hint is not yet implemented on ROCm")

    @staticmethod
    def get_dispatch_config(*args, **kwargs):
        raise NotImplementedError("get_dispatch_config is not yet implemented on ROCm")

    @staticmethod
    def get_combine_config(*args, **kwargs):
        raise NotImplementedError("get_combine_config is not yet implemented on ROCm")


class EPHandle:
    """Placeholder EPHandle for ROCm."""
    pass


class ElasticBuffer:
    """Placeholder ElasticBuffer for ROCm."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ElasticBuffer is not yet implemented on ROCm")

    @staticmethod
    def get_buffer_size_hint(*args, **kwargs):
        raise NotImplementedError("get_buffer_size_hint is not yet implemented on ROCm")

    @staticmethod
    def get_engram_storage_size_hint(*args, **kwargs):
        raise NotImplementedError("get_engram_storage_size_hint is not yet implemented on ROCm")

    @staticmethod
    def get_pp_buffer_size_hint(*args, **kwargs):
        raise NotImplementedError("get_pp_buffer_size_hint is not yet implemented on ROCm")

    @staticmethod
    def get_agrs_num_max_session_bytes(*args, **kwargs):
        raise NotImplementedError("get_agrs_num_max_session_bytes is not yet implemented on ROCm")

    @staticmethod
    def get_agrs_buffer_size_hint(*args, **kwargs):
        raise NotImplementedError("get_agrs_buffer_size_hint is not yet implemented on ROCm")

    @staticmethod
    def capture():
        return EventHandle()


def get_physical_domain_size():
    return 1, 1


def get_logical_domain_size():
    return 1, 1


def intranode_all_to_all(*args, **kwargs):
    """Placeholder intranode_all_to_all for ROCm."""
    raise NotImplementedError("intranode_all_to_all is not yet implemented on ROCm")


def internode_all_to_all(*args, **kwargs):
    """Placeholder internode_all_to_all for ROCm."""
    raise NotImplementedError("internode_all_to_all is not yet implemented on ROCm")

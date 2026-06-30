# For forward compatibility
try:
    # noinspection PyUnresolvedReferences
    from .event import EventHandle
except Exception:
    # On AMD/ROCm the C++ extension (deep_ep._C) is unavailable; the event
    # handle is only needed by the NVIDIA dispatch/combine path.
    EventHandle = None  # type: ignore[assignment,misc]

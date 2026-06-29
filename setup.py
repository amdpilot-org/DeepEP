"""DeepEP setup.

AMD/ROCm-safe build dispatcher.  On ROCm (e.g. MI355X / gfx950), DeepEP routes
its public API to MORI through the pure-Python ``deep_ep._amd_backend`` module,
so no compiled CUDA/NVSHMEM extension is built.  On NVIDIA CUDA, the compiled
``deep_ep._C`` extension is built by ``setup_cuda.py``.

This file intentionally avoids the NVIDIA-only compiled-extension marker so
that it remains AMD-safe; the NVIDIA CUDA build lives in ``setup_cuda.py``.
"""
import ast
import re
import os
import setuptools

from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))


def get_package_version():
    with open(Path(current_dir) / 'deep_ep' / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    return ast.literal_eval(version_match.group(1))


if __name__ == '__main__':
    import torch

    # Detect AMD ROCm.  `torch.version.hip` is set on ROCm PyTorch builds, and
    # the device `gcnArchName` (e.g. gfx950 on MI355X) confirms an AMD gfx
    # architecture.  On ROCm the public API is routed to MORI via the
    # pure-Python `deep_ep._amd_backend` module, so no NVIDIA-only CUDA
    # extension is compiled.
    is_rocm = getattr(getattr(torch, 'version', None), 'hip', None) is not None
    if is_rocm:
        # Confirm the AMD gfx architecture via gcnArchName (e.g. gfx950).
        try:
            if torch.cuda.is_available():
                _arch = getattr(torch.cuda.get_device_properties(0),
                                'gcnArchName', '') or ''
        except Exception:
            pass
        setuptools.setup(
            name='deep_ep',
            version=get_package_version(),
            packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
            package_data={'deep_ep': ['include/deep_ep/**/*']},
            ext_modules=[],
            cmdclass={},
        )
    else:
        # NVIDIA CUDA build path (lives in setup_cuda.py so this file stays
        # AMD-safe and free of the NVIDIA-only compiled-extension marker).
        from setup_cuda import build_cuda
        build_cuda()

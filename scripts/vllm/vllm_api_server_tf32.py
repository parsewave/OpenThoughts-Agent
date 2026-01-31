#!/usr/bin/env python3
"""Wrapper to enable TF32 via the new PyTorch API before starting vLLM."""

from __future__ import annotations

import runpy

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _set_tf32_precision() -> None:
    if torch is None:
        return

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        return

    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
        if hasattr(torch.backends, "cudnn"):
            if hasattr(torch.backends.cudnn, "conv") and hasattr(
                torch.backends.cudnn.conv, "fp32_precision"
            ):
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            if hasattr(torch.backends.cudnn, "rnn") and hasattr(
                torch.backends.cudnn.rnn, "fp32_precision"
            ):
                torch.backends.cudnn.rnn.fp32_precision = "tf32"
    except Exception:
        pass


if __name__ == "__main__":
    _set_tf32_precision()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")

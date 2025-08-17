import torch

def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    table = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[dtype_str.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unsupported dtype: {dtype_str!r}")

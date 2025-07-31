def str_to_torch_dtype(dtype_str):
    if dtype_str == "fp32":
        return "torch.float32"
    elif dtype_str == "fp16":
        return "torch.float16"
    elif dtype_str == "bf16":
        return "torch.bfloat16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}") 
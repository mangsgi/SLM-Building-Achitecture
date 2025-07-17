import torch

def get_best_device():
    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        print("✅ NVIDIA CUDA 디바이스 사용")
        return torch.device("cuda")

    # Intel GPU (IPEX)
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            print("✅ Intel GPU (XPU) 디바이스 사용 (IPEX)")
            return torch.device("xpu")
    except ImportError:
        pass

    # Apple M1/M2 (macOS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Apple MPS 디바이스 사용")
        return torch.device("mps")

    # Fallback: CPU
    print("⚠️ 사용 가능한 GPU가 없어 CPU 디바이스 사용")
    return torch.device("cpu")
import cupy as cp
import psutil


def get_current_ram_usage(
    use_gpu: bool = False, verbose: bool = False
) -> float:
    if use_gpu:
        current_ram_usage = (
            cp.get_default_memory_pool().used_bytes()
            / cp.get_default_memory_pool().total_bytes()
            * 100
        )
    else:
        current_ram_usage = psutil.virtual_memory().percent
    if verbose:
        using = "GPU" if use_gpu else "CPU"
        print(f"{using} RAM: {current_ram_usage}")
    return current_ram_usage

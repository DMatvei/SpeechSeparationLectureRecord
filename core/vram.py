import gc
import torch



# функции освобождения видеопамяти ===============================

def free_vram(label: str = "", verbose: bool = True) -> None:

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if verbose:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserv = torch.cuda.memory_reserved() / 1e9
            print(
                f'[VRAM after {label}] allocated={alloc:.2f} GB',
                f'reserved={reserv:.2f}GB', flush=True
            )


def get_device() -> str:
    return  "cuda:0" if torch.cuda.is_available() else 'cpu'

#  Полный объём видеопамяти. Если 0 - CPU
def total_vram_gb() -> float:

    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return  props.total_memory / 1e9
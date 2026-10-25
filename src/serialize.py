import numpy as np
import torch


def encode(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    a = tensor.detach().cpu().numpy()
    return a.tobytes(), list(a.shape), str(a.dtype)


def decode(data: bytes, shape: list[int], dtype: str) -> torch.Tensor:
    a = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
    return torch.from_numpy(a.copy())

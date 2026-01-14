from __future__ import annotations

from typing import Protocol, Dict, List

import numpy as np

from eztorch.typing import FloatArray


class HasParams(Protocol):
    def parameters(self) -> List[FloatArray]:
        ...

    def grads(self) -> List[FloatArray]:
        ...


class BaseModel:

    def parameters(self) -> List[FloatArray]:  # pragma: no cover - to be overridden
        raise NotImplementedError

    def grads(self) -> List[FloatArray]:  # pragma: no cover - to be overridden
        raise NotImplementedError

    def state_dict(self) -> Dict[str, FloatArray]:
        params = self.parameters()
        state: Dict[str, FloatArray] = {}
        for idx, p in enumerate(params):
            state[f"param_{idx}"] = np.array(p, copy=True)
        return state

    def load_state_dict(self, state: Dict[str, FloatArray]) -> None:
        params = self.parameters()
        for idx, p in enumerate(params):
            key = f"param_{idx}"
            if key not in state:
                raise KeyError(f"Missing parameter '{key}' in state_dict")
            src = state[key]
            if p.shape != src.shape:
                raise ValueError(f"Shape mismatch for {key}: expected {p.shape}, got {src.shape}")
            p[...] = src

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.state_dict())

    def load(self, path: str) -> None:
        data = np.load(path)
        state = {k: data[k] for k in data.files}
        self.load_state_dict(state)
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

    def train(self) -> None:
        self.training = True
        self._set_training_mode(True)

    def eval(self) -> None:
        self.training = False
        self._set_training_mode(False)

    def _set_training_mode(self, mode: bool) -> None:
        visited: set[int] = set()

        def walk(obj: object) -> None:
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if hasattr(obj, "training"):
                try:
                    setattr(obj, "training", mode)
                except Exception:
                    pass

            if isinstance(obj, dict):
                for value in obj.values():
                    walk(value)
                return

            if isinstance(obj, (list, tuple, set)):
                for value in obj:
                    walk(value)
                return

            if hasattr(obj, "__dict__"):
                for value in obj.__dict__.values():
                    walk(value)

        walk(self)

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
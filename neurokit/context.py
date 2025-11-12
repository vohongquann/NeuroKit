from typing import Any, List, Tuple

class Context:
    """
    Store temporary data during forward() to reuse in backward()
    """
    def __init__(self):
        self._saved_tensors: List[Any] = []
        self._saved_other: List[Any] = []
        self._dirty_indices: List[int] = []
        self.needs_input_grad: List[bool] = []

    # Main API
    def save_for_backward(self, *tensors: Any):
        """Save tensors/ndarrays for backward pass"""
        self._saved_tensors = list(tensors)

    def save_other(self, *objects: Any):
        """Save metadata like shape, stride, kernel..."""
        self._saved_other = list(objects)

    def mark_dirty(self, *tensor_indices: int):
        """Mark tensors that were modified in-place"""
        self._dirty_indices.extend(tensor_indices)

    def clear(self):
        """Free memory after backward pass"""
        self._saved_tensors.clear()
        self._saved_other.clear()
        self._dirty_indices.clear()

    # Access properties
    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return tuple(self._saved_tensors)

    @property
    def saved_other(self) -> Tuple[Any, ...]:
        return tuple(self._saved_other)

    @property
    def dirty_indices(self) -> Tuple[int, ...]:
        return tuple(self._dirty_indices)
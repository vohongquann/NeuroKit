from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from .context import Context
from .function import Function
import copy

class Tensor:
    """Core Tensor object supporting autograd."""

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        grad_fn: Optional[Function] = None,
        parents: Optional[List["Tensor"]] = None,
        ctx: Optional[Context] = None,
    ):
        if isinstance(data, Tensor):
            if requires_grad is False and data.requires_grad:
                requires_grad = data.requires_grad
            grad_fn = grad_fn or data.grad_fn
            parents = parents or data.parents
            ctx = ctx or data.ctx
            data = data.data.copy()

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.grad_fn = grad_fn
        self.parents = parents or []
        self.ctx = ctx
        self.is_leaf = (grad_fn is None)

    # Autograd
    def backward(self, grad: Optional[np.ndarray] = None):
        """Backpropagate gradient to parent tensors."""

        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("Gradient can only be automatically computed for scalar tensors.")
            grad = np.ones_like(self.data, dtype=np.float32)

        # Validate gradient shape - Operations must handle broadcasting
        if grad.shape != self.data.shape:
            raise RuntimeError(
                f"Gradient shape {grad.shape} doesn't match tensor shape {self.data.shape}. "
                f"Each operation must properly reduce gradients in its backward() method."
            )

        # Accumulate gradient
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad = self.grad + grad

        # Stop if leaf node or no gradient function
        if self.is_leaf or self.grad_fn is None:
            return

        # Backpropagate to parents
        grads = self.grad_fn.backward(self.ctx, grad)
        if not isinstance(grads, (tuple, list)):
            grads = (grads,)

        for parent, g in zip(self.parents, grads):
            if parent is not None and parent.requires_grad and g is not None:
                parent.backward(g)

        # Clear context after backward pass
        if self.ctx:
            self.ctx.clear()


    # Helper
    def zero_grad(self):
        """Clear previous gradients."""
        self.grad = None

    def detach(self) -> "Tensor":
        """Return a new tensor without gradient tracking."""
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data

    
    # Python operator overloading
    def __add__(self, other):
        from .ops import Add
        return Add.apply(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .ops import Subtract
        return Subtract.apply(self, other)

    def __rsub__(self, other):
        from .ops import Subtract
        return Subtract.apply(other, self)

    def __mul__(self, other):
        from .ops import Multiply
        return Multiply.apply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from .ops import Divide
        return Divide.apply(self, other)

    def __rtruediv__(self, other):
        from .ops import Divide
        return Divide.apply(other, self)

    def __pow__(self, exponent):
        from .ops import Power
        return Power.apply(self, exponent)

    def __neg__(self):
        from .ops import Negate
        return Negate.apply(self)

    def __matmul__(self, other):
        from .ops import MatMul
        return MatMul.apply(self, other)

    def __rmatmul__(self, other):
        from .ops import MatMul
        return MatMul.apply(other, self)

    def __getitem__(self, key):
        from .ops import GetItem
        return GetItem.apply(self, key)
    
    def __del__(self):
        if self.ctx:
            self.ctx.clear()

    # Extended math functions
    def sum(self):
        from .ops import Sum
        return Sum.apply(self)

    def mean(self):
        from .ops import Mean
        return Mean.apply(self)

    def exp(self):
        from .ops import Exp
        return Exp.apply(self)

    def log(self):
        from .ops import Log
        return Log.apply(self)

    # Utilities
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def reshape(self, *shape) -> "Tensor":
        from .ops import Reshape
        return Reshape.apply(self, shape)

    def __repr__(self):
        return f"Tensor(data=\n{self.data}, shape={self.data.shape}, requires_grad={self.requires_grad})"
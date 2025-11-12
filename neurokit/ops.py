import numpy as np
from typing import Tuple
from .context import Context
from .function import Function

# Binary Ops
class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output


class Subtract(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output


class Multiply(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a


class Divide(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output / b, -grad_output * a / (b ** 2)


class Power(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, p: float) -> np.ndarray:
        ctx.save_for_backward(a, p)
        return a ** p

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        a, p = ctx.saved_tensors
        return grad_output * p * (a ** (p - 1)),


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output


# Unary Ops
class Negate(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return -grad_output,


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        (out,) = ctx.saved_tensors
        return grad_output * out,


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        (a,) = ctx.saved_tensors
        return grad_output / a,


class GetItem(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, key) -> np.ndarray:
        ctx.save_for_backward(a.shape, key)
        return a[key]

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        shape, key = ctx.saved_tensors
        grad = np.zeros(shape, dtype=np.float32)
        grad[key] = grad_output
        return grad,


# Reduction Ops 
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a.shape)
        return np.sum(a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        (shape,) = ctx.saved_tensors
        return grad_output * np.ones(shape, dtype=np.float32),


class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a.shape)
        return np.mean(a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        (shape,) = ctx.saved_tensors
        size = np.prod(shape)
        return grad_output * np.ones(shape, dtype=np.float32) / size,
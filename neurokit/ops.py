import numpy as np
from typing import Tuple, Union
from .context import Context
from .function import Function
from .utils import broadcast_backward

# Binary Ops
class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_other(a.shape, b.shape)
        return a + b

    @staticmethod
    @broadcast_backward
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output

class Subtract(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_other(a.shape, b.shape)
        return a - b

    @staticmethod
    @broadcast_backward
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output

class Multiply(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        ctx.save_other(a.shape, b.shape)
        return a * b

    @staticmethod
    @broadcast_backward
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a

class Divide(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        ctx.save_other(a.shape, b.shape)
        return a / b

    @staticmethod
    @broadcast_backward
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output / b, -grad_output * a / (b ** 2)

class Power(Function):
    metadata_flag = True

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, p: Union[float, np.ndarray]) -> np.ndarray:
        ctx.save_for_backward(a, p)
        return np.array(a ** p)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        a, p = ctx.saved_tensors
        return grad_output * p * (a ** (p - 1)), None

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        ctx.save_other(a.shape, b.shape)
        return a @ b

    @staticmethod
    @broadcast_backward
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output

# Unary Ops
class Negate(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return -grad_output

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        (out,) = ctx.saved_tensors
        return grad_output * out

class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        (a,) = ctx.saved_tensors
        return grad_output / a

# Reduction Ops
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_other(a.shape)
        return np.array(np.sum(a, dtype=np.float32))
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        (shape,) = ctx.saved_other
        return grad_output * np.ones(shape, dtype=np.float32)

class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_other(a.shape)
        return np.array(np.mean(a, dtype=np.float32))
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        (shape,) = ctx.saved_other
        size = np.prod(shape)
        return (grad_output * np.ones(shape, dtype=np.float32)) / size
    
class GetItem(Function):
    metadata_flag = True

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, key) -> np.ndarray:
        ctx.save_other(a.shape, key)
        result = a[key]
        return np.asarray(result, dtype=np.float32)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        shape, key = ctx.saved_other
        grad = np.zeros(shape, dtype=np.float32)
        grad[key] = grad_output
        return (grad,) 

class Reshape(Function):
    metadata_flag = True

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
        ctx.save_other(a.shape)
        result = np.reshape(a, new_shape)
        return np.asarray(result, dtype=np.float32)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        (original_shape,) = ctx.saved_other
        return (np.reshape(grad_output, original_shape),)  
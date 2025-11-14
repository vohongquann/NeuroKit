import numpy as np

from .tensor import Tensor
from .nn import Module
from .function import Function


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0
        ctx.save_for_backward(mask)
        return np.maximum(x, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        return grad_output * mask


class ReLU(Module):
    """ReLU activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return ReLUFunction.apply(x)
    
    def __repr__(self):
        return 'ReLU()'

class SigmoidFunction(Function):
    
    @staticmethod
    def forward(ctx, x):
        output = 1.0 / (1.0 + np.exp(-x))
        ctx.save_for_backward(output)  
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_output = ctx.saved_tensors[0]
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
        return grad_input

class Sigmoid(Module):
    """Sigmoid activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return SigmoidFunction.apply(x)
    
    def __repr__(self):
        return 'Sigmoid()'

class TanhFunction(Function):
    @staticmethod
    def forward(ctx, x):
        output = np.tanh(x)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tanh_output = ctx.saved_tensors[0]
        # d/dx tanh(x) = 1 - tanh(x)^2
        grad_input = grad_output * (1 - tanh_output ** 2)
        return grad_input


class Tanh(Module):
    """Tanh activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return TanhFunction.apply(x)
    
    def __repr__(self):
        return 'Tanh()'

class LeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        mask = x > 0
        ctx.save_for_backward(mask)
        ctx.save_other(alpha)
        
        output = x.copy()
        output[~mask] *= alpha
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        alpha = ctx.saved_other[0]
        
        grad_input = grad_output.copy()
        grad_input[~mask] *= alpha
        
        return grad_input, None  # None for alpha gradient

class LeakyReLU(Module):
    """Leaky ReLU activation layer."""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        return LeakyReLUFunction.apply(x, self.alpha)
    
    def __repr__(self):
        return f'LeakyReLU(alpha={self.alpha})'

class SoftmaxFunction(Function):
    """Softmax activation with numerical stability"""
    
    @staticmethod
    def forward(ctx, x: np.ndarray, axis: int = -1) -> np.ndarray:
        # Numerical stability: subtract max
        axis = int(axis)
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max
        
        # Compute exp and normalize
        exp_x = np.exp(x_shifted)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        output = exp_x / sum_exp
        
        # Save for backward
        ctx.save_for_backward(output)
        ctx.save_other(axis)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: np.ndarray) -> tuple:
        softmax_output = ctx.saved_tensors[0]
        axis = ctx.saved_other[0]
        
        # Jacobian computation: softmax * (grad - sum(grad * softmax))
        sum_term = np.sum(grad_output * softmax_output, axis=axis, keepdims=True)
        grad_input = softmax_output * (grad_output - sum_term)
        
        return grad_input, None  # None for axis

class Softmax(Module):
    """Softmax activation layer"""
    
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis
    
    def forward(self, x: Tensor) -> Tensor:
        # Đảm bảo truyền đúng numpy array
        return SoftmaxFunction.apply(x, self.axis)
    
    def __repr__(self):
        return f'Softmax(axis={self.axis})'
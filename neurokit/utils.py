import numpy as np
from typing import Tuple, Callable
from functools import wraps

def unbroadcast(grad: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduce gradient to match original shape before broadcasting.
    
    Example:
        grad: (4, 4), original: (4,) → (4,)
        grad: (4, 4), original: (1, 4) → (1, 4)
    """
    # Sum over leading dimensions that were added
    ndim_add = grad.ndim - len(original_shape)
    for _ in range(ndim_add):
        grad = grad.sum(axis=0)
    
    # Sum over dimensions where original was 1
    for i in range(len(original_shape)):
        if original_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad.reshape(original_shape)


def broadcast_backward(backward_fn: Callable) -> Callable:
    """
    Decorator to automatically unbroadcast gradients in backward pass.
    
    Usage:
        @staticmethod
        @broadcast_backward
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            return grad_output * b, grad_output * a
    
    The decorator will:
    1. Get original shapes from ctx.saved_other (must be saved in forward)
    2. Apply backward function
    3. Unbroadcast each gradient to match original shape
    """
    @wraps(backward_fn)
    def wrapper(ctx, grad_output, *args, **kwargs):
        # Get original shapes saved in forward
        if not hasattr(ctx, 'saved_other') or not ctx.saved_other:
            raise RuntimeError(
                "broadcast_backward requires shapes saved in ctx.saved_other. "
                "Add 'ctx.save_other(a.shape, b.shape, ...)' in forward()."
            )
        
        original_shapes = ctx.saved_other
        
        # Call original backward
        grads = backward_fn(ctx, grad_output, *args, **kwargs)
        
        # Ensure grads is tuple/list
        if not isinstance(grads, (tuple, list)):
            grads = (grads,)
        
        # Unbroadcast each gradient
        unbroadcasted_grads = []
        for grad, shape in zip(grads, original_shapes):
            if grad is None:
                unbroadcasted_grads.append(None)
            else:
                unbroadcasted_grads.append(unbroadcast(grad, shape))
        
        # Return in original format
        return tuple(unbroadcasted_grads) if len(unbroadcasted_grads) > 1 else unbroadcasted_grads[0]
    
    return wrapper
import numpy as np
from .tensor import Tensor
from .nn import Module
from .function import Function
from .context import Context

class MSELossFunction(Function):
    """Mean Squared Error Loss: (1/N) * Σ(pred - target)²"""
    
    @staticmethod
    def forward(ctx: Context, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        # Validate shapes
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: pred {predictions.shape}, target {targets.shape}")
        
        # Compute difference and save for backward
        diff = predictions - targets
        ctx.save_for_backward(diff)
        ctx.save_other(diff.size)  # Total elements count
        
        # Compute MSE
        return np.array(np.mean(diff ** 2))
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        """
        Returns gradients for (predictions, targets).
        Note: targets gradient is None because loss doesn't depend on target gradients.
        """
        diff, N = ctx.saved_tensors[0], ctx.saved_other[0]
        grad_predictions = (2.0 / N) * diff * grad_output
        grad_targets = None  # Targets are constant labels, no gradient needed
        return grad_predictions, grad_targets

class MSELoss(Module):
    """MSE Loss module"""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return MSELossFunction.apply(predictions, targets)
    
    def __repr__(self):
        return 'MSELoss()'

class CrossEntropyLossFunction(Function):
    """
    Cross Entropy Loss with Softmax for multi-class classification.
    Input: logits (raw scores), targets (class indices)
    Output: scalar loss
    """
    
    @staticmethod
    def forward(ctx: Context, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        batch_size, num_classes = logits.shape
        
        # Validate targets
        if targets.shape != (batch_size,):
            raise ValueError(f"Targets shape must be ({batch_size},), got {targets.shape}")
        
        # Ensure targets are integers
        if not np.issubdtype(targets.dtype, np.integer):
            targets = targets.astype(np.int64)
        
        # Validate target range
        if np.any(targets < 0) or np.any(targets >= num_classes):
            raise ValueError(f"Target indices must be in [0, {num_classes}), got min={targets.min()}, max={targets.max()}")
        
        # Stable softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        logits_shifted = logits - logits_max
        exp_logits = np.exp(logits_shifted)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp
        
        # Get correct class probabilities
        batch_indices = np.arange(batch_size)
        correct_class_probs = probs[batch_indices, targets]
        
        # Negative log likelihood
        epsilon = 1e-7
        log_probs = np.log(correct_class_probs + epsilon)
        nll = -log_probs
        
        # Average loss
        loss = np.mean(nll)
        
        # Save for backward
        ctx.save_for_backward(probs, targets)
        ctx.save_other(batch_size)
        
        return np.array(loss)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        """
        Returns gradients for (logits, targets).
        Note: targets gradient is None because they are class labels (integers).
        """
        probs, targets = ctx.saved_tensors
        batch_size = ctx.saved_other[0]
        
        # Gradient: softmax - one_hot(targets)
        grad_logits = probs.copy()
        batch_indices = np.arange(batch_size)
        grad_logits[batch_indices, targets] -= 1
        
        # Average and apply chain rule
        grad_logits = grad_logits / batch_size * grad_output
        grad_targets = None  # Targets are discrete labels, no gradient
        
        return grad_logits, grad_targets

class CrossEntropyLoss(Module):
    """Cross Entropy Loss module."""
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return CrossEntropyLossFunction.apply(logits, targets)
    
    def __repr__(self):
        return 'CrossEntropyLoss()'
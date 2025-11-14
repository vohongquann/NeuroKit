import numpy as np
from typing import List, Dict, Optional
from .nn import Parameter

class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, parameters: List[Parameter], lr: float):
        """
        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.state: Dict[int, Dict] = {}  # State for each parameter
    
    def zero_grad(self):
        """Clear gradients of all parameters"""
        for param in self.parameters:
            param.zero_grad()
    
    def step(self):
        """Perform single optimization step"""
        raise NotImplementedError
    
    def _get_state(self, param_id: int) -> Dict:
        """Get or create state dict for a parameter"""
        if param_id not in self.state:
            self.state[param_id] = {}
        return self.state[param_id]


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(
        self, 
        parameters: List[Parameter], 
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        """
        Args:
            lr: Learning rate
            momentum: Momentum factor (0 = no momentum)
            weight_decay: L2 penalty coefficient
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
    
    def step(self):
        """Update parameters using SGD"""
        for param in self.parameters:
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                param_id = id(param)
                state = self._get_state(param_id)
                
                if 'velocity' not in state:
                    state['velocity'] = np.zeros_like(param.data)
                
                velocity = state['velocity']
                velocity[:] = self.momentum * velocity + grad
                
                if self.nesterov:
                    # Nesterov momentum: look ahead
                    grad = grad + self.momentum * velocity
                else:
                    grad = velocity
            
            # Update parameters
            param.data = param.data - self.lr * grad


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)"""
    
    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Args:
            lr: Learning rate
            betas: (beta1, beta2) coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: L2 penalty coefficient
        """
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # Timestep for bias correction
    
    def step(self):
        """Update parameters using Adam"""
        self.t += 1
        
        for param in self.parameters:
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Get or initialize state
            param_id = id(param)
            state = self._get_state(param_id)
            
            if 'm' not in state:
                state['m'] = np.zeros_like(param.data)  # First moment
                state['v'] = np.zeros_like(param.data)  # Second moment
            
            m, v = state['m'], state['v']
            
            # Update biased first and second moments
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0
    ):
        """
        Args:
            lr: Learning rate
            alpha: Smoothing constant
            eps: Term added to denominator for numerical stability
            weight_decay: L2 penalty coefficient
            momentum: Momentum factor
        """
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
    
    def step(self):
        """Update parameters using RMSprop"""
        for param in self.parameters:
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Get or initialize state
            param_id = id(param)
            state = self._get_state(param_id)
            
            if 'square_avg' not in state:
                state['square_avg'] = np.zeros_like(param.data)
                if self.momentum > 0:
                    state['momentum_buffer'] = np.zeros_like(param.data)
            
            square_avg = state['square_avg']
            
            # Update running average of squared gradients
            square_avg[:] = self.alpha * square_avg + (1 - self.alpha) * (grad ** 2)
            
            # Compute update
            if self.momentum > 0:
                buf = state['momentum_buffer']
                buf[:] = self.momentum * buf + grad / (np.sqrt(square_avg) + self.eps)
                param.data = param.data - self.lr * buf
            else:
                param.data = param.data - self.lr * grad / (np.sqrt(square_avg) + self.eps)
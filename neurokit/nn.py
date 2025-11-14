from typing import Tuple, Iterator
from .tensor import Tensor
import numpy as np  

class Parameter(Tensor):
    def __init__(self, data):
        Tensor.__init__(self, data, requires_grad=True)
    
    def __repr__(self):
        return f"Parameter containing: data={self.data}, shape={self.data.shape}, requires_grad={self.requires_grad}"

class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self.training_mode = True

        # Save sub-modules (layers)
        self._modules = dict()

        # Save parameters of this module
        self._parameters = dict()
    
    def forward(self, *args, **kargs):
        raise NotImplementedError("Sub class must implemented forward")
    
    def __call__(self, *args, **kargs):
        # Allows call modules as function: layer(x) alternative for layer.forward(x)
        return self.forward(*args, **kargs)

    def __setattr__(self, name, value):
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        
        if params is None:
            object.__setattr__(self, "_parameters", {})
            object.__setattr__(self, "_modules", {})
            params = self.__dict__["_parameters"]
            modules = self.__dict__["_modules"]
        if isinstance(value, Parameter):
            params[name] = value
        elif isinstance(value, Module):
            modules[name] = value

        object.__setattr__(self, name, value)
    
    def parameters(self) -> Iterator[Parameter]:
        # Yield own parameters
        for param in self._parameters.values():
            yield param
        
        # Recursively yield sub-module parameters
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """
        Yield (name, parameter) pairs.
        """
        # Yield own parameters
        for name, param in self._parameters.items():
            yield name, param

        # Recursively yield sub-module parameters with prefix
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                yield f"{module_name}.{param_name}", param

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True):
        self.training_mode = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        self.train(mode=False)
        return self

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1.0 / np.sqrt(in_features)
        weight_data = np.random.uniform(
            -k, k, size=(in_features, out_features)
        ).astype(np.float32)

        self.weight = Parameter(weight_data)
        if bias:
            bias_data = np.random.uniform(
                -k, k, size=(out_features,)
            ).astype(np.float32)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None
    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor (batch_size, in_features)
        y: Tensor (batch_size, out_features)

        1. Matrix multiplication: x @ weight
           Shape: (batch, in) @ (in, out) = (batch, out)

        2. Add bias (if exists): + bias
           Shape: (batch, out) + (out,) = (batch, out)
           Broadcasting handles shape automatically
        """
        outputs = x @ self.weight
        if self.bias:
            outputs = outputs + self.bias
        return outputs
    
    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
    
    def __repr__(self):
        self.extra_repr()
        return f'Linear({self.extra_repr()})'

        
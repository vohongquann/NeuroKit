import numpy as np
from .context import Context
from .tensor import Tensor
from typing import Union, Tuple, List

class Function:
    @staticmethod
    def forward(ctx: Context, *args):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs) -> Union["Tensor", Tuple["Tensor", ...]]:
        # Normalize inputs to Tensor objects
        tensor_inputs: List[Tensor] = [
            x if isinstance(x, Tensor) else Tensor(x) for x in inputs 
        ]

        # Determine if the output requires gradient
        requires_grad = any(t.requires_grad for t in tensor_inputs)

        # Create context for saving intermediates
        ctx = Context()
        ctx.needs_input_grad = [t.requires_grad for t in tensor_inputs]

        # Forward computation
        outputs = cls.forward(ctx, *[t.data for t in tensor_inputs])

        # Wrap outputs into Tensor
        def wrap_output(array: np.ndarray) -> "Tensor":
            return Tensor(
                data=np.asarray(array, dtype=np.float32),
                requires_grad=requires_grad,
                grad_fn=cls,
                parents=tensor_inputs,
                ctx=ctx,
            )
        
        if isinstance(outputs, np.ndarray):
            return wrap_output(outputs)
        
        elif isinstance(outputs, (tuple, list)):
            return tuple(wrap_output(output) for output in outputs)
        
        else:
            raise TypeError("Forward output must be np.ndarray or tuple/list of np.ndarray")
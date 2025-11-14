import numpy as np
from typing import Union, Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor import Tensor

from .context import Context

class Function:
    """Base class for differentiable operations"""
    metadata_flag = False

    @staticmethod
    def forward(ctx: Context, *args):
        """Compute output from inputs. Must be implemented by subclass."""
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs):
        """Compute input gradients from output gradients. Must be implemented by subclass."""
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs) -> Union["Tensor", Tuple["Tensor", ...]]:
        """Execute forward pass and build computational graph."""
        from .tensor import Tensor

        if cls.metadata_flag:
            tensor_input = inputs[0] if isinstance(inputs[0], Tensor) else Tensor(inputs[0])
            metadata = inputs[1:]  # Another
            tensor_inputs = [tensor_input]
        else:
            # Normalize inputs to Tensor objects
            tensor_inputs = [x if isinstance(x, Tensor) else Tensor(x) for x in inputs]
            metadata = []

        # Determine if the output requires gradient
        requires_grad = any(t.requires_grad for t in tensor_inputs)

        # Create context for saving intermediates
        ctx: Optional[Context] = None

        if requires_grad:
            ctx = Context()
            ctx.needs_input_grad = [tensor.requires_grad for tensor in tensor_inputs]
        
        # Foward computation
        forward_args = [tensor.data for tensor in tensor_inputs] + list(metadata)
        outputs = cls.forward(
            ctx if ctx else Context(), 
            *forward_args
        )

        if isinstance(outputs, np.ndarray):
            output_list = [outputs]
            is_single = True
        elif isinstance(outputs, (tuple, list)):
            for output in outputs:
                if not isinstance(output, np.ndarray):
                    raise TypeError("All forward outputs must be np.ndarray")
            output_list = list(outputs)
            is_single = False
        else:
            raise TypeError(f"Forward output must be np.ndarray or tuple/list of np.ndarray, got {type(outputs)}")
        
        # Wrap output into Tensor
        wrap_output_list = []
        for array in output_list:
            tensor = Tensor(
                data=np.asarray(array, dtype=np.float32),
                requires_grad=requires_grad,
                grad_fn=cls,
                parents=tensor_inputs,
                ctx=ctx
            )
            wrap_output_list.append(tensor)
        return wrap_output_list[0] if is_single else tuple(wrap_output_list)
    



        

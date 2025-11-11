import pickle
import numpy as np
from .layers import *
from .optimizers import *
from .network import *

def clip_gradients(gradients, max_norm):
    # Compute L2 norm of the gradients
    total_norm = np.sqrt(np.sum(gradients**2))
    # Calculate scaling factor to clip gradients if norm exceeds max_norm
    clip_coef = total_norm / (max_norm + 1e-15)
    # If norm is larger than max_norm, rescale the gradients
    if clip_coef > 1.0:
        return gradients / clip_coef
    return gradients

def batch_iterator(X, y, batch_size=32, shuffle=True):
    # Create indices and shuffle if required
    len_samples = len(X)
    indices = np.arange(len_samples)
    if shuffle:
        np.random.shuffle(indices)
    # Generate mini-batches of size batch_size
    for start in range(0, len_samples, batch_size):
        end = min(start + batch_size, len_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]

def _serialize_layer(layer):
    # Convert a layer object into a serializable dictionary format
    # Includes layer type, initialization parameters, and current state (weights, buffers, etc.)
    if isinstance(layer, Linear):
        return {
            'type': 'Linear',
            'init_kwargs': {
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'bias': layer.bias is not None  # Indicate if bias is used
            },
            'state': {
                'weights': layer.weights,  # Weight matrix
                'bias': layer.bias if layer.bias is not None else None  # Bias vector
            }
        }
    if isinstance(layer, BatchNorm1D):
        return {
            'type': 'BatchNorm1D',
            'init_kwargs': {
                'num_features': layer.num_features,  # Number of features
                'momentum': layer.momentum,        # Momentum for running stats
                'eps': layer.eps,                  # Small value for numerical stability
                'affine': layer.affine             # Whether to learn gamma/beta
            },
            'state': {
                'gamma': layer.gamma,              # Scale parameter
                'beta': layer.beta,                # Shift parameter
                'running_mean': layer.running_mean,  # Moving average of mean
                'running_var': layer.running_var     # Moving average of variance
            }
        }
    if isinstance(layer, BatchNorm2D):
        return {
            'type': 'BatchNorm2D',
            'init_kwargs': {
                'num_channels': layer.num_channels,  # Number of channels in input
                'momentum': layer.momentum,
                'eps': layer.eps,
            },
            'state': {
                'gamma': layer.gamma,
                'beta': layer.beta,
                'running_mean': layer.running_mean,
                'running_var': layer.running_var
            }
        }
    if isinstance(layer, Dropout):
        return {
            "type": "Dropout", 
            "init_kwargs": {
                "dropout_rate": layer.dropout_rate  # Probability of dropping units
            },
            "state": {
                "training": getattr(layer, "training", True)  # Store training/inference mode
            }
        }
    if isinstance(layer, ReLULayer):
        return {
            "type": "ReLULayer", 
            "init_kwargs": {},   # No parameters needed
            "state": {}          # No internal state
        }
    if isinstance(layer, LeakyReLULayer):
        return {
            "type": "LeakyReLULayer", 
            "init_kwargs": {"alpha": 0.01},  # Negative slope coefficient
            "state": {}  # Activation function with no state
        }
    if isinstance(layer, SigmoidLayer):
        return {
            "type": "SigmoidLayer", 
            "init_kwargs": {}, 
            "state": {}  # No parameters; computes 1 / (1 + exp(-x))
        }
    if isinstance(layer, TanhLayer):
        return {
            "type": "TanhLayer", 
            "init_kwargs": {}, 
            "state": {}  # Hyperbolic tangent activation, stateless
        }
    if isinstance(layer, SoftmaxLayer):
        return {
            "type": "SoftmaxLayer", 
            "init_kwargs": {}, 
            "state": {}  # Normalizes input into probability distribution
        }
    # Fallback for unrecognized layers
    return {
        'type': type(layer).__name__,  # Layer class name
        "init_kwargs": {},             # No known initialization args
        "state": {}                    # No state saved
    }

def _deserialize_layer(desc):
    # Reconstruct a layer from its serialized dictionary representation
    type_ = desc["type"]           # Get layer type
    kwargs = desc.get("init_kwargs", {})  # Get initialization arguments
    state = desc.get("state", {})         # Get internal state (weights, buffers, etc.)

    if type_ == "Linear":
        layer = Linear(**kwargs)
        if "weights" in state:
            layer.weights = state["weights"]  # Restore weight matrix
        if "bias" in state:
            layer.bias = state["bias"]        # Restore bias vector
        return layer

    elif type_ == "BatchNorm1D":
        layer = BatchNorm1D(**kwargs)
        layer.gamma = state["gamma"]              # Restore scale parameter
        layer.beta = state["beta"]                # Restore shift parameter
        layer.running_mean = state["running_mean"]  # Restore moving average of mean
        layer.running_var = state["running_var"]    # Restore moving average of variance
        return layer

    elif type_ == "BatchNorm2D":
        layer = BatchNorm2D(**kwargs)
        layer.gamma = state["gamma"]
        layer.beta = state["beta"]
        layer.running_mean = state["running_mean"]
        layer.running_var = state["running_var"]
        return layer

    elif type_ == "Dropout":
        layer = Dropout(**kwargs)
        layer.training = state.get("training", True)  # Restore training mode safely
        return layer

    elif type_ == "ReLULayer":
        return ReLULayer()  # Stateless activation

    elif type_ == "SigmoidLayer":
        return SigmoidLayer()  # No parameters, just forward pass

    elif type_ == "TanhLayer":
        return TanhLayer()  # Hyperbolic tangent activation

    elif type_ == "LeakyReLULayer":
        # Use provided alpha if available, otherwise default to 0.01
        return LeakyReLULayer(**kwargs) if "alpha" in kwargs else LeakyReLULayer()

    elif type_ == "SoftmaxLayer":
        return SoftmaxLayer()  # Normalizes input to probabilities

    else:
        raise ValueError(f"Unknown layer type: {type_}")  # Handle unsupported types

def _serialize_optimizer(optimizer):
    # Serialize optimizer into a dictionary containing its type and internal state
    # Used for saving training state 
    
    if isinstance(optimizer, SGD):
        return {
            "type": "SGD",
            "learning_rate": optimizer.learning_rate  # Only learning rate, no state
        }
    
    elif isinstance(optimizer, Momentum):
        return {
            "type": "Momentum",
            "learning_rate": optimizer.learning_rate,
            "momentum": optimizer.momentum,
            "velocities": optimizer.velocities  # Save velocity buffer for each parameter
        }
    
    elif isinstance(optimizer, RMSProp):
        return {
            "type": "RMSProp",
            "learning_rate": optimizer.learning_rate,
            "decay": optimizer.decay,           # Decay rate for moving average
            "eps": optimizer.eps,               # Small value for numerical stability
            "cache": optimizer.cache            # Running average of squared gradients
        }
    
    elif isinstance(optimizer, Adagrad):
        return {
            "type": "Adagrad",
            "learning_rate": optimizer.learning_rate,
            "eps": optimizer.eps,
            "G": optimizer.G                    # Sum of squared gradients (accumulated)
        }
    
    elif isinstance(optimizer, Adam):
        return {
            "type": "Adam",
            "learning_rate": optimizer.learning_rate,
            "beta1": optimizer.beta1,           # Exp. decay rate for first moment
            "beta2": optimizer.beta2,           # Exp. decay rate for second moment
            "eps": optimizer.eps,               # Epsilon for numerical stability
            "t": optimizer.t,                   # Time step (for bias correction)
            "m": optimizer.m,                   # First moment estimate (mean)
            "v": optimizer.v                    # Second moment estimate (uncentered variance)
        }
    
    else:
        # Fallback for unknown optimizers
        return {
            "type": type(optimizer).__name__    # Save optimizer class name
        }

def _deserialize_optimizer(state):
    # Reconstruct an optimizer from its serialized state
    # Used when loading a saved training state 
    type_ = state["type"]
    
    if type_ == "SGD":
        optimizer = SGD(learning_rate=state["learning_rate"])
        # SGD has no internal state to restore
    
    elif type_ == "Momentum":
        optimizer = Momentum(
            learning_rate=state["learning_rate"],
            momentum=state["momentum"]
        )
        optimizer.velocities = state.get("velocities", {})  # Restore velocity buffers
    
    elif type_ == "RMSProp":
        optimizer = RMSProp(
            learning_rate=state["learning_rate"],
            decay=state["decay"],
            eps=state.get("eps", 1e-8)  # Default eps if not saved
        )
        optimizer.cache = state.get("cache", {})  # Restore squared gradient cache
    
    elif type_ == "Adagrad":
        optimizer = Adagrad(
            learning_rate=state["learning_rate"],
            eps=state.get("eps", 1e-8)
        )
        optimizer.G = state.get("G", {})  # Restore accumulated squared gradients
    
    elif type_ == "Adam":
        optimizer = Adam(
            learning_rate=state["learning_rate"],
            beta1=state["beta1"],
            beta2=state["beta2"],
            eps=state.get("eps", 1e-8)
        )
        optimizer.t = state.get("t", 0)      # Time step (for bias correction)
        optimizer.m = state.get("m", {})     # First moment estimate
        optimizer.v = state.get("v", {})     # Second moment estimate
    
    else:
        raise ValueError(f"Unknown optimizer type: {type_}")  # Handle unsupported types

    return optimizer

def save_checkpoint(model, optimizer, file_path):
    # Save model and optimizer state to a file for later restoration
    data = {
        'layers': [_serialize_layer(layer) for layer in model.layers],  # Serialize all layers
        'optimizer': _serialize_optimizer(optimizer)  # Serialize optimizer state
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)  # Write serialized data to file

def load_checkpoint(filepath, model_type=None):
    # Load a saved checkpoint containing model and optimizer state
    with open(filepath, 'rb') as f:
        data = pickle.load(f)  # Read and deserialize saved data

    if model_type == 'NeuralNetwork':
        model = NeuralNetwork()  # Create a new neural network instance

        # Rebuild each layer from serialized data and add to model
        for layer_data in data['layers']:
            model.add_layer(_deserialize_layer(layer_data))

        # Reconstruct optimizer if available
        if data.get('optimizer'):
            optimizer = _deserialize_optimizer(data['optimizer'])
        else:
            optimizer = None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, optimizer
 
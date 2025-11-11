import numpy as np
from .activations import *
from .loss import *

class Layer:
    # Base class for layers

    def forward(self, x):
        raise NotImplementedError  # Must override in subclass

    def backward(self, grad_output):
        raise NotImplementedError  # Must override in subclass

    def update_params(self, optimizer):
        pass  # Only param layers (e.g. Dense) need to override

    def zero_grad(self):
        pass  # Only layers with grads need to override

class Linear(Layer):
    # Dense / Fully connected layer

    def __init__(self, input_size, output_size, init_type= 'xavier', bias= True):
        self.input_size =input_size
        self.output_size = output_size
        self.bias_flag = bias

        # Weight initialization
        if init_type.lower() == 'xavier':
            bound = np.sqrt(6 / (self.input_size + self.output_size))
        elif init_type.lower() == 'he':
            bound = np.sqrt(2 / self.input_size)
        else:  # default: small random uniform
            bound = 0.01
        self.weights = np.random.uniform(-bound, bound, (self.input_size, self.output_size))

        if bias:
            self.bias = np.zeros(output_size)
        else:
            self.bias = None

        # Gradients
        self.weights_grad = np.zeros_like(self.weights)
        if bias:
            self.bias_grad = np.zeros_like(self.bias)

        # Cache for backward pass
        self.input_cache = None
    
    def forward(self, x):
        # Foward pass
        self.input_cache = x.copy() 

        output = np.dot(x, self.weights)
        if self.bias_flag:
            output += self.bias
        return output
    
    def backward(self, grad_output):
        # Gradient w.r.t. weights: dL / dW = X^T * grad_output
        self.weights_grad += np.dot(self.input_cache.T, grad_output)

        # Gradient w.r.t bias: dl / db = sum(grad_output)
        if self.bias_flag:
            self.bias_grad += np.sum(grad_output, axis= 0)

        # Gradient w.r.t input: dl / dX = grad_output * W^T
        grad_input = np.dot(grad_output, self.weights.T)

        return grad_input
    
    def update_params(self, optimizer):
        # Update parameters by optimizer
        optimizer.update(self.weights, self.weights_grad)
        if self.bias_flag:
            optimizer.update(self.bias, self.bias_grad)

    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights_grad)
        if self.bias_flag:
            self.bias_grad = np.zeros_like(self.bias_grad)

class Activation(Layer):
    def __init__(self, activation_function, derivative_function):
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        self.input_cache = None

    def forward(self, x):
        self.input_cache = x.copy()
        return self.activation_function(x)
    
    def backward(self, grad_output):
        return grad_output * self.derivative_function(self.input_cache)
    
class ReLULayer(Activation):
    def __init__(self):
        super().__init__(relu, relu_derivative)

class SigmoidLayer(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_derivative)

class TanhLayer(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_derivative)

class LeakyReLULayer(Activation):
    def __init__(self, alpha= 0.01):
        super().__init__(
            lambda x: leaky_relu(x, alpha),
            lambda x: leaky_relu_derivative(x, alpha)
        )

class SoftmaxLayer(Activation):
    def __init__(self):
        self.output_cache = None

    def forward(self, x):
        output = softmax(x)
        self.output_cache = output.copy()
        return output

    def backward(self, grad_output):
        # Standard softmax gradient for general case
        # Note: When used with CrossEntropy, this may be redundant
        # as CrossEntropy backward already includes softmax derivative
        # s = softmax(z) 
        # ds_i / dz_j: s_i * (1 - s_i) if i = j 
        #            :-s_i * s_j       if i >< j 
        # Or write in Jacobian 
        # J_ij = s_i * (δ_ij - s_j) = diag(s) - s sᵀ
        # dL / dz = Jᵀ · dL / dŷ

        s = self.output_cache
        len_sample = self.output_cache.shape[0]

        # dL / dz has same shape as softmax output
        grad_input = np.zeros_like(s)
        for i in range(len_sample):
            s_i = s[i].reshape(-1, 1) # Column vector shape: (C, 1)
            # Jacobian matrix: diag(s) - s sᵀ       
            jacobian = np.diagflat(s_i) - np.dot(s_i, s_i.T)
            # Jacobian with gradient output 
            grad_input[i] = np.dot(jacobian, grad_output[i])
        return grad_input
    
# class MaxPool2D:
#     """2D Max Pooling Layer"""
#     def __init__(self, pool_size=2, stride=None):
#         self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
#         self.stride = stride if stride is not None else self.pool_size
#         self.input_cache = None
#         self.mask = None
        
#     def forward(self, x):
#         """Forward pass for max pooling"""
#         self.input_cache = x.copy()
#         N, C, H, W = x.shape
#         pool_h, pool_w = self.pool_size
        
#         # Calculate output dimensions
#         out_h = (H - pool_h) // self.stride[0] + 1
#         out_w = (W - pool_w) // self.stride[1] + 1
        
#         # Initialize output and mask
#         out = np.zeros((N, C, out_h, out_w))
#         self.mask = np.zeros_like(x)
        
#         for n in range(N):
#             for c in range(C):
#                 for i in range(out_h):
#                     for j in range(out_w):
#                         h_start = i * self.stride[0]
#                         h_end = h_start + pool_h
#                         w_start = j * self.stride[1]
#                         w_end = w_start + pool_w
                        
#                         # Find max in the pooling window
#                         pool_region = x[n, c, h_start:h_end, w_start:w_end]
#                         max_val = np.max(pool_region)
#                         out[n, c, i, j] = max_val
                        
#                         # Create mask for backpropagation
#                         max_mask = (pool_region == max_val)
#                         self.mask[n, c, h_start:h_end, w_start:w_end] = max_mask
                        
#         return out
    
#     def backward(self, grad_output):
#         """Backward pass for max pooling"""
#         N, C, out_h, out_w = grad_output.shape
#         grad_input = np.zeros_like(self.input_cache)
#         pool_h, pool_w = self.pool_size
        
#         for n in range(N):
#             for c in range(C):
#                 for i in range(out_h):
#                     for j in range(out_w):
#                         h_start = i * self.stride[0]
#                         h_end = h_start + pool_h
#                         w_start = j * self.stride[1]
#                         w_end = w_start + pool_w
                        
#                         # Distribute gradient only to max locations
#                         grad_input[n, c, h_start:h_end, w_start:w_end] += (
#                             grad_output[n, c, i, j] * self.mask[n, c, h_start:h_end, w_start:w_end]
#                         )
                        
#         return grad_input
    
#     def update_params(self, optimizer):
#         pass  # No parameters to update
    
#     def zero_grad(self):
#         pass  # No gradients to zero

# class Conv2D:
#     """2D Convolution Layer"""
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride
#         self.padding = padding
#         self.bias_flag = bias
        
#         # Initialize weights using He initialization
#         fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
#         bound = np.sqrt(2.0 / fan_in)
#         self.weights = np.random.normal(0, bound, 
#                                       (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        
#         if bias:
#             self.bias = np.zeros(out_channels)
#         else:
#             self.bias = None
            
#         # Gradients
#         self.weights_grad = np.zeros_like(self.weights)
#         if bias:
#             self.bias_grad = np.zeros_like(self.bias)
            
#         # Cache for backward pass
#         self.input_cache = None
        
#     def forward(self, x):
#         """Forward pass for convolution"""
#         # x shape: (N, C, H, W)
#         self.input_cache = x.copy()
        
#         N, C, H, W = x.shape
#         FN, FC, FH, FW = self.weights.shape
        
#         # Add padding
#         if self.padding > 0:
#             x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
#                             mode='constant', constant_values=0)
#         else:
#             x_padded = x
            
#         # Calculate output dimensions
#         out_h = (H + 2 * self.padding - FH) // self.stride + 1
#         out_w = (W + 2 * self.padding - FW) // self.stride + 1
        
#         # Initialize output
#         out = np.zeros((N, FN, out_h, out_w))
        
#         # Convolution operation
#         for n in range(N):  # batch
#             for f in range(FN):  # output channels
#                 for i in range(out_h):
#                     for j in range(out_w):
#                         h_start = i * self.stride
#                         h_end = h_start + FH
#                         w_start = j * self.stride
#                         w_end = w_start + FW
                        
#                         # Element-wise multiplication and sum
#                         out[n, f, i, j] = np.sum(x_padded[n, :, h_start:h_end, w_start:w_end] * self.weights[f])
                        
#                         if self.bias_flag:
#                             out[n, f, i, j] += self.bias[f]
                            
#         return out
    
#     def backward(self, grad_output):
#         """Backward pass for convolution"""
#         x = self.input_cache
#         N, C, H, W = x.shape
#         FN, FC, FH, FW = self.weights.shape
        
#         # Add padding to input
#         if self.padding > 0:
#             x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
#                             mode='constant', constant_values=0)
#         else:
#             x_padded = x
            
#         # Initialize gradients
#         grad_x_padded = np.zeros_like(x_padded)
#         grad_weights = np.zeros_like(self.weights)
        
#         if self.bias_flag:
#             grad_bias = np.zeros_like(self.bias)
        
#         _, _, out_h, out_w = grad_output.shape
        
#         for n in range(N):
#             for f in range(FN):
#                 for i in range(out_h):
#                     for j in range(out_w):
#                         h_start = i * self.stride
#                         h_end = h_start + FH
#                         w_start = j * self.stride
#                         w_end = w_start + FW
                        
#                         # Gradient w.r.t. weights
#                         grad_weights[f] += x_padded[n, :, h_start:h_end, w_start:w_end] * grad_output[n, f, i, j]
                        
#                         # Gradient w.r.t. input
#                         grad_x_padded[n, :, h_start:h_end, w_start:w_end] += self.weights[f] * grad_output[n, f, i, j]
                        
#                         # Gradient w.r.t. bias
#                         if self.bias_flag:
#                             grad_bias[f] += grad_output[n, f, i, j]
        
#         # Remove padding from input gradient
#         if self.padding > 0:
#             grad_x = grad_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
#         else:
#             grad_x = grad_x_padded
            
#         # Store gradients
#         self.weights_grad += grad_weights
#         if self.bias_flag:
#             self.bias_grad += grad_bias
            
#         return grad_x
    
#     def update_params(self, optimizer):
#         optimizer.update(self.weights, self.weights_grad)
#         if self.bias_flag:
#             optimizer.update(self.bias, self.bias_grad)
    
#     def zero_grad(self):
#         self.weights_grad = np.zeros_like(self.weights_grad)
#         if self.bias_flag:
#             self.bias_grad = np.zeros_like(self.bias_grad)

class Conv2D:
    """Optimized 2D Convolution Layer using im2col and matrix multiplication"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Initialize weights using He initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = np.sqrt(2.0 / fan_in)
        self.weights = np.random.normal(0, bound, 
                                      (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        
        if bias:
            self.bias = np.zeros(out_channels)
        else:
            self.bias = None
            
        # Gradients
        self.weights_grad = np.zeros_like(self.weights)
        if bias:
            self.bias_grad = np.zeros_like(self.bias)
            
        # Cache for backward pass
        self.input_cache = None
        self.col_cache = None
        
    def im2col(self, x, kernel_h, kernel_w, stride, padding):
        """
        Convert input tensor to column matrix for efficient convolution
        This transforms the convolution operation into a matrix multiplication
        """
        N, C, H, W = x.shape
        
        # Add padding
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                            mode='constant', constant_values=0)
        else:
            x_padded = x
            
        # Calculate output dimensions
        out_h = (H + 2 * padding - kernel_h) // stride + 1
        out_w = (W + 2 * padding - kernel_w) // stride + 1
        
        # Create column matrix
        col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
        
        for j in range(kernel_h):
            j_lim = j + stride * out_h
            for i in range(kernel_w):
                i_lim = i + stride * out_w
                col[:, :, j, i, :, :] = x_padded[:, :, j:j_lim:stride, i:i_lim:stride]
        
        # Reshape to (N * out_h * out_w, C * kernel_h * kernel_w)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        
        return col, out_h, out_w
    
    def col2im(self, col, input_shape, kernel_h, kernel_w, stride, padding):
        """
        Convert column matrix back to input tensor format for backpropagation
        """
        N, C, H, W = input_shape
        out_h = (H + 2 * padding - kernel_h) // stride + 1
        out_w = (W + 2 * padding - kernel_w) // stride + 1
        
        # Reshape column back to 6D tensor
        col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
        
        # Initialize padded input gradient
        if padding > 0:
            img = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
        else:
            img = np.zeros((N, C, H, W))
        
        # Accumulate gradients
        for j in range(kernel_h):
            j_lim = j + stride * out_h
            for i in range(kernel_w):
                i_lim = i + stride * out_w
                img[:, :, j:j_lim:stride, i:i_lim:stride] += col[:, :, j, i, :, :]
        
        # Remove padding if applied
        if padding > 0:
            return img[:, :, padding:-padding, padding:-padding]
        else:
            return img
    
    def forward(self, x):
        """Optimized forward pass using im2col and matrix multiplication"""
        # x shape: (N, C, H, W)
        self.input_cache = x.copy()
        
        N, C, H, W = x.shape
        FN, FC, FH, FW = self.weights.shape
        
        # Convert input to column matrix
        col, out_h, out_w = self.im2col(x, FH, FW, self.stride, self.padding)
        self.col_cache = col  # Cache for backward pass
        
        # Reshape weights to (out_channels, in_channels * kernel_h * kernel_w)
        weights_col = self.weights.reshape(FN, -1)
        
        # Matrix multiplication: (out_channels, kernel_size) @ (kernel_size, N*out_h*out_w)
        out = weights_col @ col.T  # Shape: (out_channels, N*out_h*out_w)
        
        # Add bias if present
        if self.bias_flag:
            out = out + self.bias.reshape(-1, 1)
        
        # Reshape output to (N, out_channels, out_h, out_w)
        out = out.reshape(FN, N, out_h, out_w).transpose(1, 0, 2, 3)
        
        return out
    
    def backward(self, grad_output):
        """Optimized backward pass using cached column matrix"""
        x = self.input_cache
        N, C, H, W = x.shape
        FN, FC, FH, FW = self.weights.shape
        
        # Reshape grad_output to (out_channels, N*out_h*out_w)
        grad_output_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(FN, -1)
        
        # Gradient w.r.t. weights using matrix multiplication
        # weights_grad = grad_output @ input_col^T
        weights_col_grad = grad_output_reshaped @ self.col_cache  # Shape: (out_channels, kernel_size)
        self.weights_grad += weights_col_grad.reshape(self.weights.shape)
        
        # Gradient w.r.t. bias
        if self.bias_flag:
            self.bias_grad += np.sum(grad_output_reshaped, axis=1)
        
        # Gradient w.r.t. input using matrix multiplication
        # input_col_grad = weights^T @ grad_output
        weights_col = self.weights.reshape(FN, -1)
        col_grad = weights_col.T @ grad_output_reshaped  # Shape: (kernel_size, N*out_h*out_w)
        col_grad = col_grad.T  # Shape: (N*out_h*out_w, kernel_size)
        
        # Convert column gradients back to input gradients
        grad_x = self.col2im(col_grad, x.shape, FH, FW, self.stride, self.padding)
        
        return grad_x
    
    def update_params(self, optimizer):
        optimizer.update(self.weights, self.weights_grad)
        if self.bias_flag:
            optimizer.update(self.bias, self.bias_grad)
    
    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights_grad)
        if self.bias_flag:
            self.bias_grad = np.zeros_like(self.bias_grad)

class MaxPool2D:
    """Ultra-optimized Max Pooling using numpy stride tricks (experimental)"""
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        self.input_cache = None
        self.mask_cache = None
    
    def forward(self, x):
        """Ultra-fast forward pass using stride tricks"""
        from numpy.lib.stride_tricks import sliding_window_view
        
        self.input_cache = x.copy()
        N, C, H, W = x.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        
        # Calculate output dimensions
        out_h = (H - pool_h) // stride_h + 1
        out_w = (W - pool_w) // stride_w + 1
        
        # Use sliding window view for efficient pooling
        # This creates a view of all pooling windows without copying data
        windows = sliding_window_view(x, (pool_h, pool_w), axis=(2, 3))
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]
        
        # Find max values across pooling dimensions
        out = np.max(windows, axis=(4, 5))
        
        # Create mask for backprop (simplified version)
        self.mask_cache = np.zeros_like(x)
        
        # This is still O(n) but more efficient than before
        windows_flat = windows.reshape(N, C, out_h, out_w, -1)
        max_indices = np.argmax(windows_flat, axis=4)
        
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        max_idx = max_indices[n, c, i, j]
                        max_h, max_w = np.unravel_index(max_idx, (pool_h, pool_w))
                        self.mask_cache[n, c, h_start + max_h, w_start + max_w] = 1
        
        return out
    
    def backward(self, grad_output):
        """Same backward pass as regular MaxPool2D"""
        N, C, out_h, out_w = grad_output.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        
        grad_input = np.zeros_like(self.input_cache)
        
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        h_end = h_start + pool_h
                        w_start = j * stride_w
                        w_end = w_start + pool_w
                        
                        grad_input[n, c, h_start:h_end, w_start:w_end] += (
                            grad_output[n, c, i, j] * self.mask_cache[n, c, h_start:h_end, w_start:w_end]
                        )
                        
        return grad_input
    
    def update_params(self, optimizer):
        pass
    
    def zero_grad(self):
        pass

class CNN:
    """Complete CNN Model"""
    def __init__(self):
        self.layers = []
        self.training = True
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def update_params(self, optimizer):
        for layer in self.layers:
            layer.update_params(optimizer)
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

class Dropout(Layer):
    def __init__(self, dropout_rate= 0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training and self.dropout_rate > 0:
            # Create dropout mask 
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
            return x * self.mask / (1 - self.dropout_rate)
        return x.copy()
    
    def backward(self, grad_output):
        if self.training and self.dropout_rate > 0:
            return grad_output * self.mask / (1 - self.dropout_rate)
        return grad_output
    
    def set_training(self, training):
        self.training = training

class Flatten:
    """Flatten layer to convert 4D tensor to 2D"""
    def __init__(self):
        self.input_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
    def update_params(self, optimizer):
        pass
    
    def zero_grad(self):
        pass

class BatchNorm1D(Layer):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True,
                 gamma=1.0, beta=0.0):
        self.num_features = num_features
        self.affine = affine

        # Exponentially weighted average
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        if self.affine:
            self.gamma = gamma * np.ones(num_features)  # scale
            self.beta = beta * np.ones(num_features)    # shift
        else:
            # Non-learnable parameters for non-affine mode
            self.gamma = np.ones(num_features)
            self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Gradients (only meaningful if affine=True)
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

        # Cache for backward pass 
        self.input_cache = None
        self.normalized_cache = None
        self.var_cache = None
        self.mean_cache = None
        self.training = True

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0, ddof=0)  # Use population variance

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)

            # Cache for backward pass 
            self.input_cache = x.copy()
            self.normalized_cache = x_norm.copy()  # Fixed variable name
            self.var_cache = batch_var.copy()
            self.mean_cache = batch_mean.copy()
        else: 
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        return self.gamma * x_norm + self.beta

    def backward(self, grad_output):
        if not self.training:
            # For inference mode, gradient flows through normalization only
            return grad_output * self.gamma / np.sqrt(self.running_var + self.eps)
        
        batch_size = self.input_cache.shape[0]
        x_centered = self.input_cache - self.mean_cache
        std_inv = 1.0 / np.sqrt(self.var_cache + self.eps)

        # Gradients for gamma and beta (only update if affine=True)
        if self.affine:
            self.gamma_grad += np.sum(grad_output * self.normalized_cache, axis=0)
            self.beta_grad += np.sum(grad_output, axis=0)

        # Gradients w.r.t normalized input 
        grad_norm = grad_output * self.gamma

        # Gradient w.r.t variance
        grad_var = -np.sum(grad_norm * x_centered, axis=0) * (std_inv ** 3) / 2

        # Gradient w.r.t mean
        grad_mean = (-np.sum(grad_norm * std_inv, axis=0) + 
                    grad_var * (-2.0 * np.sum(x_centered, axis=0) / batch_size))

        # Gradients w.r.t input 
        grad_input = (grad_norm * std_inv + 
                     2.0 * grad_var * x_centered / batch_size + 
                     grad_mean / batch_size)

        return grad_input
    
    def update_params(self, optimizer):
        if self.affine:
            optimizer.update(self.gamma, self.gamma_grad)
            optimizer.update(self.beta, self.beta_grad)

    def zero_grad(self):
        self.gamma_grad = np.zeros_like(self.gamma_grad)
        self.beta_grad = np.zeros_like(self.beta_grad)

    def set_training(self, training):
        self.training = training

class BatchNorm2D(Layer):
    # Batch normalization for 2d feature maps: N, C, H, W
    def __init__(self, num_channels, eps= 1e-15, momentum= 0.1):
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters per channel
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)

        # Running statistic per channel 
        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)

        # Gadient
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

        # Cache for backward pass
        self.input_cache = None
        self.norm_cache = None
        self.var_cache = None
        self.training = True

    def forward(self, x):
        # Check the shape of x
        assert (x.ndim == 4 and x.shape[1] == self.num_channels), "BatchNorm2D expects input of shape (N, C, H, W)"

        if self.training:
            axes = (0, 2, 3) # over N, H, W (per-channel stats)
            batch_mean = np.mean(x, axis= axes)
            batch_var = np.var(x, axis= axes)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            denom = np.sqrt(batch_var + self.eps)
            x_hat = (x - batch_mean[None, :, None, None]) / denom[None, :, None, None]

            # Cache for backward
            self.input_cache = x.copy()
            self.norm_cache = x_hat.copy()
            self.var_cache = batch_var.copy()
        else:
            denom = np.sqrt(self.running_var + self.eps)
            x_hat = (x - self.running_mean[None, :, None, None]) / denom[None, :, None, None]
        return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]
    
    def backward(self, grad_output):
        # grad_output: (N, C, H, W)
        if not self.training:
            # During eval, treat BN as affine with fixed stats
            return grad_output * (self.gamma[None, :, None, None] / np.sqrt(self.running_var[None, :, None, None] + self.eps))
        x = self.input_cache
        x_hat = self.norm_cache
        var = self.var_cache

        N, C, H, W = x.shape
        axes = (0, 2, 3)

        m = N * H * W 

        # Gradients for gamma and beta
        self.beta_grad += np.sum(grad_output, axis= axes)
        self.gamma_grad += np.sum(grad_output * x_hat, axis= axes)

        # Gradient for normalization
        grad_norm = grad_output * self.gamma[None, :, None, None]
        std_inv = 1.0 / np.sqrt(var + self.eps)
        std_inv_b = std_inv[None, :, None, None]

        mu = np.mean(x, axis= axes)
        x_centered = x - mu[None, :, None, None]

        grad_var = - np.sum(grad_norm * x_centered, axis= axes) * (std_inv ** 3) / 2
        grad_mean = - (np.sum(grad_norm * std_inv_b, axis= axes) + 2 * grad_var * np.sum(x_centered, axis= axes) / m)
        grad_input = (grad_norm * std_inv_b + 2 * x_centered * grad_var[None, :, None, None] / m + grad_mean[None, :, None, None] / m)
        return grad_input

    def update_params(self, optimizer):
        optimizer.update(self.gamma, self.gamma_grad)
        optimizer.update(self.beta, self.beta_grad)

    def zero_grad(self):
        self.gamma_grad = np.zeros_like(self.gamma_grad)
        self.beta_grad = np.zeros_like(self.beta_grad)

    def set_training(self):
        self.training = True

class SoftmaxCrossEntropyLoss:
    
    def __init__(self):
        self.cache = None  

    def forward(self, logits, targets):
        N = logits.shape[0]  # Number of samples

        # Numerically stable softmax 
        # Shift logits by max for stability: exp(x - max) avoids overflow
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (N, C)

        # Compute cross-entropy loss 
        # Clip probabilities to prevent log(0)
        probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)

        if targets.ndim == 1:  # Class indices
            # Extract log probability of true class
            log_probs = -np.log(probs_clipped[np.arange(N), targets])  # (N,)
        else:  # One-hot encoded
            # -sum(y * log(p)) for each sample
            log_probs = -np.sum(targets * np.log(probs_clipped), axis=1)  # (N,)

        loss = np.mean(log_probs)  # Average over batch

        #  Cache for backward pass 
        self.cache = (probs, targets)
        return loss

    def backward(self):
        assert self.cache is not None, "Must call forward() before backward()"
        probs, targets = self.cache
        N = probs.shape[0]

        # Gradient: (probs - targets) / N
        # But targets may be class indices or one-hot
        if targets.ndim == 1:
            # Convert class indices to one-hot for gradient computation
            grad = probs.copy()  # (N, C)
            grad[np.arange(N), targets] -= 1  # p_i - 1 for correct class
        else:
            grad = probs - targets  # (N, C)

        # Normalize by batch size because loss was averaged
        return grad / N

    def __call__(self, logits, targets):
        return self.forward(logits, targets)
import numpy as np

# Activation function 
def relu(x):
    # Rectified Linear Unit activation function
    return np.maximum(0, x)

def relu_derivative(x):
    # Derivative of the ReLU function
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    # Sigmoid activation function
    x_clipped = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    # Derivative of the Sigmoid function
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    # Tanh activation function 
    return np.tanh(x)

def tanh_derivative(x):
    # Derivative of the Tanh function
    return 1 - np.tanh(x) ** 2

def leaky_relu(x, alpha= 0.01):
    # Leaky ReLu activation function
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha= 0.01):
    # Derivative of the Leaky ReLU function
    return np.where(x > 0, 1, alpha)

def softmax(x):
    # Softmax activation function
    x_shifted = x - np.max(x, axis= -1, keepdims= True)
    x_clipped = np.clip(x_shifted, -500, 500)
    exp_x = np.exp(x_clipped)
    return exp_x / np.sum(exp_x, axis= -1, keepdims= True)
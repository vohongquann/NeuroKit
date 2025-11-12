import numpy as np

# NN Layer ====================================================================================================
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.training = True

    def add_layer(self, layer):
        # Add a layer to the network
        self.layers.append(layer)

    def forward(self, x):
        # Forward pass through the network
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        # Backward pass through the network
        grad = grad_output # Use grad to store gradient of last layer
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    def update_params(self, optimizer):
        # Update all parameters in all layers
        for layer in self.layers:
            layer.update_params(optimizer)
        
    def zero_grad(self):
        # Zero all gradients in all layers
        for layer in self.layers:
            layer.zero_grad()

    def train(self):
        # Set to training mode
        self.training = True

    def eval(self): 
        # Set to evaluation mode
        self.training = False

# CNN layer =========================================================================================
class CNN:
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
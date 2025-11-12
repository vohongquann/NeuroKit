import numpy as np

# Note: Use in-place to update parameters of network 
# (If params =  params + .. -> crate new variable in function -> useless)
class SGD: 
    # Stochastic Gradient Descent optimizer
    def __init__(self, learning_rate= 0.001):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        # θ_{t+1} = θ_t - η * g_t
        params -= self.learning_rate * gradients

class Momentum:
    def __init__(self, learning_rate= 0.001, momentum= 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, params, gradients):
        params_id = id(params)

        if params_id not in self.velocities:
            self.velocities[params_id] = np.zeros_like(params)
        
        # Update velocities 
        # v_t = μ * v_{t-1} - α * g_t
        # θ_{t+1} = θ_t + v_t
        self.velocities[params_id] = self.momentum * self.velocities[params_id] - self.learning_rate * gradients

        # Update param
        params += self.velocities[params_id]
        
class RMSProp: 
    # Root Mean Square Propagation optimizer
    def __init__(self, learning_rate= 0.001, decay= 0.9, eps= 1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.eps = eps
        self.cache = {}
    
    def update(self, params, gradients):
        params_id = id(params)

        if params_id not in self.cache:
            self.cache[params_id] = np.zeros_like(params)

        # Update cache: E[g^2]_t
        # E[g^2]_t = β * E[g^2]_{t-1} + (1 - β) * g_t^2
        # θ_{t+1} = θ_t - η / (√E[g^2]_t + ε) ⊙ g_t
        self.cache[params_id] = self.decay * self.cache[params_id] + (1 - self.decay) * (gradients ** 2)

        # Update parameter 
        # # Adaptive learning_rate: scale down if grad large, up if small
        params -= self.learning_rate * gradients / (np.sqrt(self.cache[params_id]) + self.eps)

class Adagrad:
    def __init__(self, learning_rate= 0.001, eps= 1e-8):
        self.learning_rate = learning_rate
        self.eps = eps
        self.G = {}

    def update(self, params, gradients):
        params_id = id(params)

        if params_id not in self.G:
            self.G[params_id] = np.zeros_like(params) # G_0 = 0

        # G_t = G_{t-1} + g_t²
        self.G[params_id] += gradients ** 2 
        # θ_{t+1} = θ_t - η ⋅ g_t / √(G_t + ε)
        params -= self.learning_rate * gradients / np.sqrt(self.G[params_id] + self.eps)

class Adam:
    def __init__(self, learning_rate= 0.001, beta1= 0.9, beta2= 0.999, eps= 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0 #time step

        # Moment 
        self.m = {} # First moment
        self.v = {} # Second moment

        # Track if step() has been called this iteration
        self._step_called = False

    def step(self):
        if not self._step_called:
            self.t += 1
            self._step_called = True
    
    def zero_step(self):
        # Reset step flag (call this at the beginning of each optimization iteration)
        self._step_called = False

    def update(self, params, gradients):
        params_id = id(params)

        if params_id not in self.m or params_id not in self.v:
            self.m[params_id] = np.zeros_like(params)  # m_0 = 0
            self.v[params_id] = np.zeros_like(params)  # v_0 = 0

        if not self._step_called:
            self.step()

        # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        self.m[params_id] = self.beta1 * self.m[params_id] + (1 - self.beta1) * gradients

        # v_t = β₂ * v_{t-1} + (1 - β₂) * (g_t)^2
        self.v[params_id] = self.beta2 * self.v[params_id] + (1 - self.beta2) * (gradients ** 2)

        # \hat{m}_t = m_t / (1 - β₁^t)
        # \hat{v}_t = v_t / (1 - β₂^t)
        m_hat = self.m[params_id] / (1 - self.beta1 ** self.t)  # \hat{m}_t
        v_hat = self.v[params_id] / (1 - self.beta2 ** self.t)  # \hat{v}_t

        # θ_{t+1} = θ_t - η ⋅ \hat{m}_t / (√\hat{v}_t + ε)
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


        
import numpy as np

def mse_loss(predictions, targets):
    # Mean square error loss
    predictions = predictions.flatten()
    targets = targets.flatten()
    return ((predictions - targets ) ** 2).mean()

def mse_loss_derivative(predictions, targets):
    # Derivative of mean square error loss
    len_sample = predictions.shape[0]
    return 2 * (predictions - targets) / len_sample

def cross_entropy_loss(predictions, targets):
    # Clip predictions to prevent log(0)
    predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)

    if targets.ndim == 1:  # Class indices: targets = [0, 1, 2, ...]
        len_samples = predictions.shape[0]
        # Select the predicted probability for the correct class
        correct_class_probs = predictions_clipped[np.arange(len_samples), targets]
        return -np.log(correct_class_probs).mean()
    else:  # One-hot encoded targets
        return -(targets * np.log(predictions_clipped)).sum(axis=1).mean()
    
def cross_entropy_loss_derivative(predictions, targets):
    # Derivative of cross-entropy loss
    len_sample = predictions.shape[0]

    if targets.ndim == 1:
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(len_sample), targets] = 1 
        targets = one_hot
    return (predictions - targets) / len_sample




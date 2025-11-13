
import numpy as np


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Mean Squared Error

    Args:
        y_true: numpy array of shape (n_samples, n_target_variables)
        y_pred: numpy array of shape (n_samples, n_target_variables)

    Returns:
        float: NMSE
    """
    assert isinstance(y_true, np.ndarray), f"y_true is not a numpy array: {type(y_true)}"
    assert isinstance(y_pred, np.ndarray), f"y_pred is not a numpy array: {type(y_pred)}"
    assert y_true.shape == y_pred.shape, f"y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}"
    epsilon = 1e-10
    return np.sum((y_true - y_pred) ** 2) / (np.sum(y_true ** 2) + epsilon)

def fitness(nmse: float, complexity: int) -> float:
    '''Fitness function to balance between NMSE and complexity

    Args:
        nmse: float, normalized mean squared error
        complexity: int, complexity of the model
        lam: float, regularization parameter, controls the trade-off between NMSE and complexity
        L: float, maximum complexity of the model

    Returns:
        float: fitness value
    '''
    lam = 1.0 # coefficient of complexity in the fitness score
    L = 200 # maximum complexity in the fitness score
    return 1/(1 + nmse) + lam*np.exp(-complexity/L)
import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # 1. Convert given python list into numpy array
    x = np.asarray(x, dtype=np.float32)

    # 2. Calculate sigmoid = 1 / (1 + e^-x) 
    return 1 / (1 + np.exp(-x))
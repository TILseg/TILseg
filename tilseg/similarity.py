import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def image_similarity(mask1, mask2):
    """
    calculates mean squared error and between two arrays 
    and generates the image difference.

    Parameters
    ----
    mask1 (np.ndarray): array of first image
    mask2 (np.ndarray): array of second image

    Returns:
    mse (float): mean squared error
    diff (np.ndarray): image difference as numpy array.
    """

    # calculate mse
    mse = mean_squared_error(mask1.flatten(), mask2.flatten())

    # calculate absolute difference 
    abs_diff = np.abs(mask1 - mask2)
    # # threshold the result, wrapped around 255s replaced by 1s
    diff = np.where(abs_diff > 1, 1, abs_diff)

    return mse, diff  
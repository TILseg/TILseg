import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error

def image_similarity(array1, array2):
    """
    calculates mean squared error and between two arrays 
    and generates the image difference.

    Parameters
    ----
    array1 (np.ndarray): array of first image.
    array2 (np.ndarray): array of second image.

    Returns:
    mse (float): mean squared error
    diff (np.ndarray): image difference as numpy array.
    """

   # calculate mse
    mse = mean_squared_error(array1, array2)

    # Compute mean squared error
    mse = np.mean(squared_diff)    

    return mse, diff

# usage:
if __name__ == "__main__":
    img1_path = "image1.jpg"
    img2_path = "image2.jpg"

    mse, ssim_score, diff = image_similarity(img1_path, img2_path)
    print ("MSE:", mse)
    print("SSIM Similarity Score:", ssim_score)

    # save or display difference image
    Image.fromarray(diff).save("difference_image.jpg")
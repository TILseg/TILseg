import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from PIL import Image

def image_similarity(img1_path, img2_path):
    """
    calculates mean squared error and strcutural similarity index between two images 
    and generates the image difference.

    Parameters
    ----
    img1_path (str): path to first image.
    img2_path (str): path to second image.

    Returns:
    mse (float): mean squared error
    ssim score (float): SSIM similarity score between two images.
    diff (np.ndarray): image difference as numpy array.
    """

    # load images, convert to greyscale, then to  floating-point arrays
    img1 = img_as_float(np.array(Image.open(img1_path).convert('L')))
    img2 = img_as_float(np.array(Image.open(img2_path).convert('L')))

   # calculate mse
    mse = np.sum((img1_path.astype("float") - img2_path.astype("float")) ** 2 )
    mse /= float(img1_path.shape[0] * img1_path.shape[1])

    # calculate SSIM and difference array
    ssim_score, diff = ssim(img1, img2, full=True)

    # scales pixel values to [0,255] from [0,1] and converts to integers
    diff = (diff * 255).astype(np.uint8)

    return mse, ssim_score, diff

# usage:
if __name__ == "__main__":
    img1_path = "image1.jpg"
    img2_path = "image2.jpg"

    mse, ssim_score, diff = image_similarity(img1_path, img2_path)
    print ("MSE:", mse)
    print("SSIM Similarity Score:", ssim_score)

    # save or display difference image
    Image.fromarray(diff).save("difference_image.jpg")
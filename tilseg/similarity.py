# function takes two image paths as input, calculates the SSIM similarity score between them, 
# and generates an image difference

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from PIL import Image

def image_similarity(img1_path, img2_path):
    """
    calculate the similarity score between two images using SSIM metric and generate the image difference.

    Parameters
    ----
    img1_path (str): path to first image.
    img2_path (str): path to second image.

    Returns:
    ssim score (float): SSIM similarity score between two images.
    diff (np.ndarray): image difference as numpy array.
    """

    # load images
    img1 = img_as_float(np.array(Image.open(img1_path).convert('L')))
    img2 = img_as_float(np.array(Image.open(img2_path).convert('L')))

    # calculate SSIM
    ssim_score, diff = ssim(img1, img2, full=True)

    # convert difference image to uint8 for visualization
    diff = (diff * 255).astype(np.uint8)

    return ssim_score, diff

# usage:
if __name__ == "__main__":
    img1_path = "image1.jpg"
    img2_path = "image2.jpg"

    similarity_score, difference_image = image_similarity(img1_path, img2_path)
    print("SSIM Similarity Score:", similarity_score)

    # save or display difference image
    Image.fromarray(difference_image).save("difference_image.jpg")
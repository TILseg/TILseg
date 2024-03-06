import unittest
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tilseg.similarity import image_similarity 

class TestImageSimilarity(unittest.TestCase):
    def setUp(self):
        # creates two simple test images before each test
        # black image
        self.img1 = np.zeros((100, 100), dtype=np.uint8)  
        # white image
        self.img2 = np.ones((100, 100), dtype=np.uint8) * 255
        self.img1_path = "test_img1.jpg"
        self.img2_path = "test_img2.jpg"
        Image.fromarray(self.img1).save(self.img1_path)
        Image.fromarray(self.img2).save(self.img2_path)

    def tearDown(self):
        # cleans up test images after each test
        import os
        os.remove(self.img1_path)
        os.remove(self.img2_path)

    def test_image_similarity(self):
        # test on two images
        ssim_score, diff, mse = image_similarity(self.img1_path, self.img2_path)

        # SSIM score for two different images should be 0
        self.assertAlmostEqual(ssim_score, 0.0, places=2)  
        # MSE (Mean Squared Error)should be the same
        expected_mse = np.mean((self.img1.astype("float") - self.img2.astype("float")) ** 2)
        self.assertAlmostEqual(mse, expected_mse, places=2)
        # Assert difference image shape
        self.assertEqual(diff.shape, self.img1.shape)


        # test on same image
        ssim_score, diff, mse = image_similarity(self.img1_path, self.img2_path)
        # SSIM score for two identical images should be 1
        self.assertAlmostEqual(ssim_score, 1.0, places=2)  
        # MSE for two identical images should be 0
        self.assertAlmostEqual(mse, 0.0, places=2)
        # difference image should be all 0s
        self.assertTrue(np.array_equal(diff, np.zeros_like(diff)))  

    def test_image_similarity_invalid_paths(self):
        # invalid paths provided
        with self.assertRaises(FileNotFoundError): 
        image_similarity("invalid_path1.jpg", "invalid_path2.jpg")

if __name__ == '__main__':
    unittest.main()

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import PIL
import openslide
import os
from skimage import io

def pred_and_cluster(model, dir_path):

    '''
    model: model that will be used to predict the patch.
    dir_path: path of the directory that has patches that needs to be predicted and clustered.
              This should be probably be performed on tissue_images_file_patch subdirectory created in extract_svs_img function below
    
    '''
    # Iterating over every patch in the directory
    for file in os.listdir(dir_path):
        
        # Creating a directory with the same file name (without extenstion)
        # Passing if such a directory already exists
        try:
            os.mkdir(file[:-4])
        except:
            pass
        
        # Makes sure that the model is training for 8 clusters
        # if len(model.cluster_centers_) == 8:
        #     pass
        # else:
        #     raise ValueError("The model must be trained for 8 clusters")
        
        # Reads the current patch into a numpy uint8 array 
        pred_patch = plt.imread(os.path.join(dir_path, file))
        # Linearizes the array for R, G, and B separately and normalizes
        # The result is an N X 3 array where N=height*width of the patch in pixels
        pred_patch_n = np.float32(pred_patch.reshape((-1, 3))/255.)
        # Predicting the index/labels of the clusters on the fitted model from 'model' function
        # The result is an N X 3 array where N=height*width of the patch in pixels
        # Each value shows the label of the cluster that pixel belongs to
        labels = model.predict(pred_patch_n)
        # creates a copy of the coordinates of the cluster centers in the RGB space
        # The results is 8X3 numpy array
        overlay_center = np.copy(model.cluster_centers_)
        # created a numpy uint8 array of the background image- this is just the H&E patch without any normalization
        back_img = np.uint8(np.copy(pred_patch))
        # Reassigning the cluster centers of the RGB space to custom colors for visual effects
        # Essentially creating new RGB coordinates for each cluster center
        overlay_center[0] = np.array([255, 102, 102])/255. #Light Red
        overlay_center[1] = np.array([153, 255, 51])/255. #Light Green
        overlay_center[2] = np.array([0, 128, 255])/255. #Light Blue
        overlay_center[3] = np.array([0, 255, 255])/255. #Cyan
        # overlay_center[4] = np.array([178, 102, 255])/255. #Light Purple
        # overlay_center[5] = np.array([95, 95, 95])/255. #Grey
        # overlay_center[6] = np.array([102, 0, 0])/255. #Maroon
        # overlay_center[7] = np.array([255, 0, 127])/255. #Bright Pink

        # Iterating over each cluster centroid
        for i in range(len(overlay_center)):
            # Creating a copy of the linearized and normalized RGB array
            seg_img = np.copy(pred_patch_n)
            # The left-hand side is a mask that accesses all pixels that belong to cluster 'i'
            # The ride hand side replaces the RGB values of each pixel with the RGB value of the corresponding custom-chosen RGB values for each cluster
            seg_img[labels.flatten() == i] = overlay_center[i] 
            # The left-hand side is a mask that accesses all pixels that DO NOT belong to cluster 'i'
            # The ride hand side replaces the RGB values of each pixel with white color
            # Therefor every pixel except for those in cluster 'i' will be white
            seg_img[labels.flatten() != i] = np.array([255, 255, 255])/255.
            # Reshapes the image with cluster 'i' identified to the original picture shape
            seg_img = seg_img.reshape(pred_patch.shape)
            # Saves the image as filename_segmented_#.jpg with 1,000 dots per inch printing resolution
            # Thus there will be 8 images identifying each cluster from each patch
            plt.imsave(os.path.join(file[:-4], '_segmented_'+str(i)+'.jpg'), seg_img, dpi=1000)
            # Reversing the normalization of the RGB values of the image with the cluster
            seg_img = np.uint8(seg_img*255.)
            # cv2.addWeighted is a function that allows us to overlay one image on top of another and adjust their 
            # alpha (transparency) so that the two can blended/overlayed and both still be clearly visible
            # overlay_img is the image where the segmented image consisting of isolated cluster is overlayed over the 
            # original H&E image
            overlay_img = cv2.addWeighted(back_img, 0.4, seg_img, 0.6, 0)/255.
            # Saves the overlayed image as filename_overlay_#.jpg with 1,000 dots per inch printing resolution
            # Thus there will be 8 overlayed images identifying each cluster from each patch
            plt.imsave(os.path.join(file[:-4], '_overlay_'+str(i)+'.jpg'), overlay_img, dpi=1000)
        # Make an image containing all the clusters in one
        # Also reshapes the image with all clusters identified to the original picture shape
        # Don't quite understand how this line would work without indexing error but I get what it is trying to do
        all_cluster = overlay_center[labels.flatten()].reshape(pred_patch.shape)
        # Saves the image as filename_all_cluster.jpg with 1,000 dots per inch printing resolution
        # Thus there will be 1 image identifying all clusters on the same image from each patch
        plt.imsave(os.path.join(file[:-4], '_all_cluster.jpg'), all_cluster, dpi=1000)
        # Overlaying the complete cluster:
        # Reversing the normalization of the RGB values of the image with the all the clusters
        seg_img = np.uint8(np.copy(all_cluster)*255.)
        # overlay_img is the image where the segmented image consisting of all the isolated clusters 
        # is overlayed over the original H&E image
        overlay_img = cv2.addWeighted(back_img, 0.6, seg_img, 0.4, 0)
        # Saves the overlayed image as filename_full_overlay.jpg with 1,000 dots per inch printing resolution
        # Thus there will be 1 fully overlayed image identifying all the clusters from each patch
        plt.imsave(os.path.join(file[:-4], '_full_overlay.jpg'), overlay_img, dpi=1000)

    return None
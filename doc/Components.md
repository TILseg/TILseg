#### PREPROCESSING
##### What it does: 
Takes the image file of the whole slide image (WSI) and separates it into smaller patches which are then sampled and used for creating the superpatch.
Inputs: WSIs, usually .svs or .ndpi


**Subcomponents**
##### Patch Creation
Divides the whole slide image(s) (WSI) into smaller sections called “patches”
1. open_slide: opens a slide and returns OpenSlide object and slide’s dimensions
   	Input:
   	Output:




#### MODEL SELECTION
##### What it does:
Utilizes various functions to score and select clustering algorithms and their hyperparameters based on three metrics: Silhouette scores, Calinski Harabasz scores, and Davies Bouldin scores. Also contains optimization functions for kmeans, dbscan, birch, and optics models.
**Subcomponents**



#### REFINED KMEANS
##### What it does:
**Subcomponents**



#### IMAGE SEGMENTATION
##### What it does:
Includes functions that fit and score clustering models as well as segment TILs in H&E stained patches within an image, or images. This should be done after using the tilseg.model_selection module to select and optimize hyperparameters of a model.

**Subcomponents**



#### CLUSTER PROCESSING
##### What it does:
Uses a clustering model developed in modules above to generate a variety of information from the clusters generated which can be used for subsequent analysis. The output is a series of images which represent the original image and relevant overlays of the determined clusters. Additionally, based on the clusters, data from filtered cell clusters will be compiled into a CSV. Immune cell groups are identified using the contour functionality from OpenCV. The implemented filters are based on area and roundness of the derived contours.

**Subcomponents**

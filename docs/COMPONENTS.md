## Summary of Components

1. Patch Generator 
    - Access Whole Slide Image 
    - Turn Image into Patches
    - Filter Patches - Call Function 2
    - Save Patches based on background pixel count
<br/>

2. Image Filter
    - Access Patch
    - Implement trained model for background removal
    - Output background pixel count
<br/>

3. Superpatch Generator (collection of patches from different H&E images)
    - Access saved patches
    - Append to form superpatches
    - Output directory of superpatches
<br/>

4. KMeans Model Fitter to Superpatch
    - Access saved superpatches
    - Fit KMeans clustering model to superpatch
    - Output clustering model
<br/>

4. Clustering Scorer
    - Access saved model or fit a new model
    - Access saved patch of interest
    - Perform clustering to obtain cluster labels
    - Score the clustering
    - Output scores
<br/>

5. TILs Segmentor
    - Access saved model or fit new model
    - Acess patches of interest from input directory
    - Perform clustering on all to obtain cluster labels
    - Filter clustered objects to identify TILs
    - Output TIL counts in each patch
    - Save transformed images containing cluster and TILs information
<br/>

6. Immune Contour Generator
    - Access labeled images with clusters
    - Generate contours based on pixel location and cluster
    - Filter contours based on size and roundness
    - Output filtered contours identified as immune cells
<br/>

7. Immune Cluster Identifier
    - Access cell groups and relevant statistics
    - Identify immune cell cluster based on analysis of cell group statistics
    - Output immune cell groups from all clusters
<br/>

8. Cluster Image Generator
    - Access labeled images with clusters assigned to pixels
    - Generate image files converting clusters to RGB values
    - Output overlaid images and cluster mask images
<br/>

9. Contour Data Generator
    - Access filtered contours identified as immune clusters
    - Use OpenCV to calculate feature values for each filtered contour
    - Optionally generate and output filtered contour overlaid image and mask
    - Output comma seperated value file with features calculated for filtered contours

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

4. Cluster Model Fitter  
    - Access saved superpatches
    - Fit clustering model to superpatch
    - Output clustering model
<br/>

5. Cluster Predictor
    - Apply generic clustering model
    - Output labeled image with clusters assigned to pixels 
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

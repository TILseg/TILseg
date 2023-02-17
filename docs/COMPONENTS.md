Component 1: Patch Generator 
Access Whole Slide Image 
Turn Image into Patches
Filter Patches - Call Function 2
Save Patches based on background pixel count

Component 2: Image Filter
Access Patch
Implement trained model for background removal
Output background pixel count

Component 3: Superpatch Generator (collection of patches from different H&E images)
Access saved patches
Append to form superpatches
Output directory of superpatches

Component 4: Cluster Model Fitter  
Access saved superpatches
Fit clustering model to superpatch
Output clustering model

Component 5: Cluster Predictor
Apply generic clustering model
Output labeled image with clusters assigned to pixels 

Component 6: Cell Group Generator
Access labeled images with clusters
Generate groups based on pixel location and cluster 
Generate statistics for derived groups (size, circularity etc.)

Component 7: Immune Cluster Identifier
Access cell groups and relevant statistics
Identify immune cell cluster based on analysis of cell group statistics
Output immune cell groups from all clusters

Component 8: Overlay Generator
Access labeled image with clusters assigned to pixels
Generate image files converting clusters to RGB values
Output overlayed image and cluster location image

Component 9: Statistical Summary Generator
Access statistics for cell clusters
Output well-formated summary and relevant data
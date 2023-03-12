## Use Cases
1. User wants to visualize immune cell clusters from H&E stained images
    - User inputs: H&E stained image(s)
    - User receives: Image with overlay that indifies cluster location and size
<br/>

2. User wants to quantify immune cell clusters using H&E stained images
    - User inputs: H&E stained image(s)
    - User receives: Count of filtered immune cell clusters and optionally feature data (including perimeter, area, circularity etc.)
<br/>

3. User wants to segment images into various tissue types
    - User inputs: H&E stained image(s)
    - User receives: Image with overlay that identifies epthilium, stroma, and glass tissue types
<br/>

4. User wants to implement a clustering model for different types of images
    - User inputs: N/A 
    - User receives: Pipeline for clustering on whole slide image 
<br/>

5. User wants guidance on H&E stained image analysis for diagnosis  
    - User inputs: H&E stained image(s)
    - User receives: Locations of high immune cell density via overlaid immune cell clusters
<br/>

6. User wants to quickly count immune cell clusters using H&E stained images
    - User inputs: H&E stained image(s)
    - User receives: Count of filtered immune cell clusters
    - Application Note: Image post-processing contains boolean options for outputs, minimizing calculation time if only TIL counts are desired

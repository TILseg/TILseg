### USER STORY: RESEARCHERS

As a data science researcher in the Mittal lab, I want to use machine learning models to analyze stained images of breast cancer tissue and distinguish between various immune cell types. I want the software to be well-documented and organized such that additional features can be implemented in future iterations.
  - Use Case 1: General
      - Input:
      - Output:
  - Use Case 2: Mittal Lab
      - Input:
      - Output:



### USER STORY: CLINICIAN

I am a clinician who has had a vast experience with looking at scans of H&E stained breast tissue. After receiving the images from the biopsy I took, I want to run the images of my patient’s breast cancer tissue through a program and analyze them to give a diagnosis that will follow a more personalized medicine approach. To be effective, the software should be easy to use for a non-data science user and produce quantitative results. I am not experienced with technical skills and my priority is giving the patient a timely diagnosis, so I value a simple interface.
  - Use Case 1: Retrieving Original + Clustered Images 
      - Input: Selection of corresponding patient + H&E images to be added
      - Output: Confirmation of uploaded imagesRetrieve Quantitative Results from Images
  - Use Case 2: Retrieve Quantitative Results from Images
      - Input: User selects the image they want to look at
      - Output: Identification + quantification of immune cells clusters
  - Use Case 3: Run statistical testing on samples



### USER STORY: HEALTHCARE ADMINISTRATOR
As a healthcare administrator, I want to use this tool across the many hospitals in our system. I would like this tool to be scalable and able to take in images from a variety of patients while simultaneously producing results that provide a diagnosis that is not going to vary widely between different images (samples). I am informed on the medical science involved, but I do not have much of a background in data science. 
  - Use Case 1: 
  - Use Case 2:



### USER STORY: LAB TECHNICIANS 
I am a lab technician working with samples of breast cancer tissue. Since I am not deeply familiar with technology and am handling many patient and doctor needs, I require a user-friendly platform where I can easily upload stained images from a biopsy and apply the software to generate the images a clinician would need to look at.
  - Use Case 1: Inputting Images/Scans
      - Input: Selection of corresponding patient + H&E images to be added
      - Output: Confirmation of uploaded images
  - Use Case 2: Applying the Model to the Stained Images
      - Input: H&E stained image(s)
      - Output: Images and overlays with colored masks of the clusters



  ### USER STORY: PATIENT

  I am a patient who has recently been screened for breast cancer. I would like this tool to provide me with an accurate diagnosis on my progression. I know nothing about the science behind cancer and how it spreads, and minimal background in data science. But I would like to be involved in this process in any way possible to relieve anxiety. Having some sort of quantitative score to measure how accurate this diagnosis may be would be very helpful for me.



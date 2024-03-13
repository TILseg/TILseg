### USER STORY: RESEARCHERS

As a data science researcher, I want to use machine learning models to analyze stained images of breast cancer tissue and distinguish between various immune cell types. I want the software to be well-documented and organized such that additional features can be implemented in future iterations and accross multiple labs. I do have some background within data science, and am proficient to advanced in this field.
  - Use Case 1: General lab use
      - Input: H&E stained images
      - Output: Identification + quantification of immune cells clusters to be compared with clinician diagnosis
  - Use Case 2: Mittal Lab
      - Input: H&E stained images, with the intent on comparing to previous data
      - Output: Both qualitative and quantitative difference in this method and previous methods used within the lab



### USER STORY: CLINICIAN

I am a clinician who has had a vast experience with looking at scans of H&E stained breast tissue. After receiving the images from the biopsy I took, I want to run the images of my patient’s breast cancer tissue through a program and analyze them to give a diagnosis that will follow a more personalized medicine approach. To be effective, the software should be easy to use for a non-data science user and produce quantitative results. I am not experienced with technical skills and my priority is giving the patient a timely diagnosis, so I value a simple interface.
  - Use Case 1: Retrieving Original + Clustered Images 
      - Input: Selection of corresponding patient + H&E images to be added
      - Output: Confirmation of uploaded imagesRetrieve Quantitative Results from Images
  - Use Case 2: Retrieve Quantitative Results from Images
      - Input: User selects the image they want to look at
      - Output: Identification + quantification of immune cells clusters
  - Use Case 3: Run statistical testing on samples (future use case; our software currently cannot handle this)
      - Input: Identification + quantification of immune cells clusters information from previous runs
      - Output: Statisical difference between clustering information accross multiple images. This helps to provide information as to how much more "consistent" this software is than clinicians in accurately diagnosing patients.



### USER STORY: HEALTHCARE ADMINISTRATOR
As a healthcare administrator, I want to use this tool across the many hospitals in our system. I would like this tool to be scalable and able to take in images from a variety of patients while simultaneously producing results that provide a diagnosis that is not going to vary widely between different images (samples). I am informed on the medical science involved, but I do not have much of a background in data science. 
  - Use Case 1: User wants to ensure diagnoses within a single hospital over a time period is consistent
      - Inputs: H&E stained images form each month in a year for multiple doctors in a hospital
      - Outputs: Identification + quantification of immune cells clusters (comparison of the diagnosis form this information and what was diagnosed in real time)
  - Use Case 2: User wants to ensure diagnoses within a collection of hospitals over a time period is consistent
      - Inputs: H&E stained images form each month in a year for a single doctor in each respective hospital
      - Outputs: Identification + quantification of immune cells clusters (comparison of the diagnosis form this information and what was diagnosed in real time)



### USER STORY: LAB TECHNICIANS 
I am a lab technician working with samples of breast cancer tissue. Since I am not deeply familiar with technology and am handling many patient and doctor needs, I require a user-friendly platform where I can easily upload stained images from a biopsy and apply the software to generate the images a clinician would need to look at.
  - Use Case 1: Inputting Images/Scans
      - Input: Selection of corresponding patient + H&E images to be added
      - Output: Confirmation of uploaded images
  - Use Case 2: Applying the Model to the Stained Images
      - Input: H&E stained image(s)
      - Output: Images and overlays with colored masks of the clusters



  ### USER STORY: PATIENT

  I am a patient who has recently been screened for breast cancer. I would like this tool to provide me with an accurate diagnosis on my progression. I know nothing about the science behind cancer and how it spreads, and minimal background in data science. But I would like to be involved in this process in any way possible to relieve anxiety. Having a quick quantitative score to measure how accurate this diagnosis may be would be very helpful for me.
  - Use Case 1: Patient wants confirmation that their diagnosis was accurate
      - Input: H&E stained image (from the patient)
      - Output: Quantitative score showing confidence in diagnosis

Note: In a world where  personalized medicine is increasingly relevant we see this as a growing use case. Many individuals would like to be more informed on how and why they are treated the way they are within medicine. This software can also help within this realm.


# DEEP-LEARNING-BASED-HUMAN-GAIT-PATTERN-ANALYSIS-FOR-CLASSIFICATION-OF-SUSPECTS
This research presents a novel deep neural network-based system for gait-based person identification, specifically designed for forensic applications within a selected pool of suspects. The proposed approach utilizes an Artificial Neural Network (ANN) to identify potential criminals in evidence videos, focusing on distinguishing gait patterns. A comprehensive set of 54 gait parameters, incorporating visibility scores, is defined, and calculated on a per-cycle basis to enrich the training dataset.

## Final UI for precdicting the criminal among suspects.



## Steps involved:
##### 1.Prepare the dataset
##### 2.keypoint extraction
##### 3.Gait parameter calculation
##### 4.Feature vector modelling
##### 5.ANN model
##### 6.UI for predicting the criminal

## Prepare the dataset
Multi-camera setup was employed, capturing footage from both side and front angles.
![Screenshot from 2023-07-24 16-28-52](https://github.com/cbkings/DEEP-LEARNING-BASED-HUMAN-GAIT-PATTERN-ANALYSIS-FOR-CLASSIFICATION-OF-SUSPECTS/assets/46423501/2b51ef29-3a8c-4d13-8026-49b83b47f047)

### Keypoint extraction
Extraction of coordinates of joint points using the MediaPipe library.
![output_with_landmarks](https://github.com/cbkings/DEEP-LEARNING-BASED-HUMAN-GAIT-PATTERN-ANALYSIS-FOR-CLASSIFICATION-OF-SUSPECTS/assets/46423501/57548bd7-a177-4ee7-b6fd-9400da96332f)

### Gait Parameter Calculation
Deriving static and dynamic gait parameters from the extracted keypoints, including the calculation of parameters per gait cycle.

![image](https://github.com/cbkings/DEEP-LEARNING-BASED-HUMAN-GAIT-PATTERN-ANALYSIS-FOR-CLASSIFICATION-OF-SUSPECTS/assets/46423501/d65722bb-9d32-4bec-9608-3af9ee265c57)

### Feature Vector Construction

![Screenshot from 2023-07-24 16-50-32](https://github.com/cbkings/DEEP-LEARNING-BASED-HUMAN-GAIT-PATTERN-ANALYSIS-FOR-CLASSIFICATION-OF-SUSPECTS/assets/46423501/2e27f0ab-3a66-4585-9c82-6960dfbca100)

### ANN model
During the training phase, we divided the available data into a training set and a testing set. Specifically, we recorded a total of 6 videos for each individual, reserving one video for testing purposes, and using the remaining 5 videos for model training.
For both the side-facing and front-facing camera perspectives, we trained separate models with and without visibility scores. The models utilized the Rectified Linear Unit (ReLU) and Sigmoid activation functions, while the loss function employed was sparse categorical cross-entropy.

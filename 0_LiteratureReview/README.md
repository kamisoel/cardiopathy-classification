# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: *"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation 
and Diagnosis: Is the Problem Solved?"*

  - **[Bernard et al. (2018)](https://doi.org/10.1109/tmi.2018.2837502)**
  - **Objective**: Introduction of the “Automatic Cardiac Diagnosis Challenge” dataset (ACDC) and summary
  of the challenges' results.
  - **Methods**: The presented studies on the automatic diagnosis task all used feature based approaches based
  on the segmentation masks
  - **Outcomes**: Accuracy of 96% for automatic diagnosis
  - **Relation to the Project**: This article introduced the used dataset and summaries the results 
  obtained by several teams. While it is more focused on the segmentation part of the ACDC challenge, it still
  serves as a great starting point on the classification task.

- **Source 2**: *"Densely Connected Fully Convolutional Network for Short-Axis Cardiac Cine MR Image 
Segmentation and Heart Diagnosis Using Random Forest"*

  - **[Khened, Alex, & Krishnamurthi (2018).](https://doi.org/10.1007/978-3-319-75541-0_15)**
  - **Objective**: The study presents a submission on the ACDC challenge for both the segmentation and
  the automatic diagnosis tasks.
  - **Methods**: For the diagnosis task a random forest was employed using features calculated from the
  segmentation masks (e.g. volumes)
  - **Outcomes**: Accuracy of 90%
  - **Relation to the Project**: The authors present an approach that is similar to the chosen baseline
  approach with similar results.

- **Source 3**: *"Automatic Cardiac Disease Assessment on cine-MRI via Time-Series Segmentation and 
Domain Specific Features"*

  - **[Isensee et al. (2017)](https://arxiv.org/abs/1707.00587)**
    - **Objective**: The authors present an integrated segmentation and disease classification pipeline based on the
    ACDC challenge 
  - **Methods**: 
  For the classification task, information is extracted from the segmented time-series in 
  form of comprehensive features handcrafted to reflect diagnostic clinical procedures. 
  Based on these features an ensemble of heavily regularized multilayer perceptrons (MLP) 
  and a random forest classifier to predict the pathologic target class are trained. 
  - **Outcomes**: Accuracy of 92%
  - **Relation to the Project**: The authors present an approach that is similar to the chosen baseline
  approach. In this study patient's anthropometric features (e.g weight, body height) and dynamic features
  based on the whole cardiac cycle are employed as well, which are not available in the kaggle challenge.
  Still similar results were obtained.

- **Source 4**: *"Med3D: Transfer Learning for 3D Medical Image Analysis"*
  - **[Chen., Ma. & Zheng (2019)](https://arxiv.org/abs/1904.00625)**
  - **Objective**: The MedicalNet project provides a series of 3D-ResNet pre-trained models and corresponding 
transfer-learning training code based on a large dataset with diverse modalities, target organs, and pathologies
  - **Methods**: A series of 3D-ResNet were pretrained on a multi-class segmentation task
  - **Outcomes**: SOTA results for fine-tuned segmentation tasks
  - **Relation to the Project**: Since diagnosis of the wanted diseases are highly typically based on the segmentation
  it is reasonable to assume that segmentation could give useful features for classification.
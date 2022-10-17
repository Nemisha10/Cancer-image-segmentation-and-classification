# Cancer image segmentation and classification (ML-project)
This project deals with cancer image segmentation and classification

Problem Statement
-------------------------------------
The script attempts at classification of histopathological cancer images – images that can be used to identify onset of the said disease. The images used for the case study are of colon and lung cancer. The image dataset consists of 5 different image records. In lung tissue images, one set consists of normal lung tissue images and the other 2 sets correspond to specific types of lung cancer types. In the case of colon tissue, one set consists of normal, healthy colon tissue and the other set is of cases with colon cancer. 
In the first part, the images are segmented to identify the areas of interest in the images that can be used to identify and classify the images into different categories. This is an exercise in clustering/grouping and segmenting. The normal and segmented images of the tissues are compared to better understand how unmasked images are masked prior to classification. 
The data is used to train a deep learning model, specifically CNN to identify and classify an image into any of the 5 different kinds of cancer images. The accuracy of the model is then compared with other pre-trained models to assess the accuracy and precision obtained from the model.




ToolKit:
------------
itertools
pandas
numpy
tensorflow
Matplotlib.pyplot
glob
sklearn
seaborn
skimage.color
cv2
tensorflow.keras
tensorflow.keras.preprocessing.image
tensorflow.keras.applications.efficientnet
tensorflow.keras.layers
Tensorflow.keras.models







Dataset Used:
--------------------
https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af


URL: https://github.com/tampapath/lung_colon_image_set/
License: No license specified, the work may be protected by copyright.
Terms: Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019


Methods Used:
--------------------


Data generation and augmentation:
1. ImageDataGenerator module from Keras to augment the data


Modeling:
1. Convolutional Neural Networks based on Keras has been used for the classification of the images. 
2. Segmentation of the images has been carried out using K-means clustering to create masking.
3. Pre-trained model values for simple CNN, ResNet (CNN with 50 layers) and InceptionV3 (CNN for Image analysis) has been used for comparison with the model trained by the script to validate the results.


Model Validation:
1. The accuracy and loss metrics has been compared to better understand the performance of the model. This comparison is done across test and train dataset as well as against the number of cycles that the model was trained for.
2. Precision, Recall and F1-score of the classification model was also compared to analyze the model results and ensure higher value of recall to minimise false negatives.
3. Confusion matrix was plotted to understand the extent of misclassifications and to identify the higher misclassified sections
4. Accuracy of the test dataset for different models were also compared against the number of epochs ( cycles of updation in neural networks)

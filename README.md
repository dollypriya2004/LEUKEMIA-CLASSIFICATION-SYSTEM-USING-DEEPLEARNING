Leukemia Classification System using Deep Learning
This project is a deep learning-based system for detecting leukemia from blood smear images. It uses a fine-tuned ResNet18 convolutional neural network (CNN) to classify blood smear images as Leukemia or Healthy. The system provides a web application for users to upload images and get instant classification results.

Project Overview
The Leukemia Classification System automates the process of detecting leukemia in blood smear images using deep learning. The system is built using Flask for the web application, PyTorch for the model, and ResNet18 as the backbone architecture for the CNN model.

Key Features:
High Accuracy: Achieves 99.65% accuracy in classifying blood smear images.

Real-Time Prediction: Users can upload images and get classification results instantly.

Interpretability: Grad-CAM is used for visualizing the regions of the image that contribute to the classification decision.

Web Interface: Simple, easy-to-use UI for uploading images and viewing results.

Technologies Used
Flask: Web framework for creating the application and handling HTTP requests.

PyTorch: Framework for building, training, and deploying deep learning models.

ResNet18: Pretrained Convolutional Neural Network used for image classification.

Grad-CAM: Used for visualizing which areas of the input image are important for model predictions.

HTML/CSS: For frontend UI design.

JavaScript: For dynamic interactivity in the web application.

Dataset
The model was trained on a dataset of 13,000 high-resolution labeled blood smear images, split into:

ALL (Acute Lymphoblastic Leukemia): 7,000 images for training

HEM (Healthy): 3,000 images for training

Test Set: 2,000 ALL images and 1,000 HEM images.

Image Preprocessing:
Resized to 224x224 pixels.

Normalized using ImageNet's mean and standard deviation.

Augmented with techniques such as rotation, flipping, and contrast adjustment.

Model Architecture
The system uses ResNet18, a lightweight convolutional neural network, for binary classification. The model was fine-tuned on the dataset, achieving high accuracy in classifying images into Leukemia and Healthy classes.

Optimizer: Adam

Loss Function: Cross-Entropy Loss

Training Epochs: 50

Batch Size: 32

Accuracy: 99.65%

Results
The model achieved the following results on the test set:

Accuracy: 99.65%

Precision: 99.65%

Recall: 99.65%

F1-Score: 99.65%

The Grad-CAM visualization helped identify the most significant features in the image, confirming the model's focus on relevant areas for leukemia detection.

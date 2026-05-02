# Bubble_detection_deeplearning
OMR Bubble Detection — VGG19 Classifier
A deep learning model that classifies Optical Mark Recognition (OMR) answer sheet bubbles into four states: filled, crossed, default, and invalid. Achieved 98% training accuracy and 97% validation accuracy.

Overview
This project addresses a common challenge in automated exam grading: accurately detecting how a bubble has been marked on an OMR sheet. The pipeline combines a DCGAN-based synthetic data generator with a fine-tuned VGG19 classifier to produce a robust, production-ready model — even when original labeled data is limited.

Classes
Label    Description
filled   Bubble is fully shaded/darkened
crossed  Bubble is marked with an X or cross
default  Bubble is empty 
unmarked invalidBubble is ambiguously or incorrectly marked
Model Performance

Split Accuracy 
Training 98% 
Validation 97%
Evaluation outputs include:

Per-class precision, recall, and F1-score (classification report)
Confusion matrices for both validation and test sets
Per-class bar chart (Precision / Recall / F1)

Architecture
Backbone: VGG19 (pretrained on ImageNet)
Convolutional feature layers are frozen (transfer learning)
Only the final fully connected classifier layer is retrained for 4-class output
Input size: 224 × 224 × 3 (RGB)
Output: softmax over 4 classes

Project Structure :
omr-detect/
├── omr-detect2.ipynb          # Full pipeline notebook
├── vgg19_best.pt              # Best model checkpoint (PyTorch)
├── vgg19_model.h5             # Model weights (HDF5 format)
├── vgg19_classification_report.csv
├── vgg19_confusion_matrix_val.png
├── vgg19_confusion_matrix_test.png
├── vgg19_per_class_prf_test.png
└── gan_loss_<class>.png       # GAN training loss plots per class

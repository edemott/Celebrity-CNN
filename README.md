# Face Classification Using EfficientNet Model

## Overview

This project implements a face classification system using the EfficientNet model. The model is trained on a dataset of celebrity faces and employs transfer learning, data augmentation, and cross-validation to achieve high accuracy in predicting class labels.

Table of Contents
	•	Features
	•	Installation
	•	Dataset
	•	Model
	•	Training Process
	•	Results
	•	Confusion Matrix
	•	Dependencies

Features
	•	EfficientNet: Leverages EfficientNetV2 pretrained on ImageNet for transfer learning.
	•	Data Augmentation: Enhances training using various augmentation techniques like random flips, rotations, and color jitter.
	•	Cross-Validation: Utilizes K-Fold cross-validation for robust performance evaluation.
	•	Confusion Matrix: Visualizes classification performance.

Installation
	1.	Clone this repository:

git clone https://github.com/your-username/face-classification-efficientnet.git


	2.	Install the required Python packages:

pip install torch torchvision scikit-learn numpy matplotlib seaborn

Dataset
	•	Path: Update the dataset directory path in the script:

directory = '/content/Celebrity Faces Dataset'


	•	Structure: Each folder in the dataset corresponds to a class, and images within it are labeled accordingly.

Model

The EfficientNetV2 model was chosen for its efficiency and high performance. The classifier’s final layer was adjusted to match the number of classes in the dataset:

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

Training Process
	1.	Data Loading: Images are preprocessed and augmented using PyTorch transforms.
	2.	Cross-Validation: A 10-fold cross-validation approach ensures a robust evaluation of model performance.
	3.	Optimization:
	•	Optimizer: Adam
	•	Learning Rate: 0.0003
	•	Scheduler: StepLR for dynamic learning rate adjustment
	4.	Training Loop: The model is trained for 10 epochs per fold, and loss is logged for monitoring progress.

Results
	•	Overall Accuracy: 91.17%
	•	Per-Fold Accuracy: Detailed logs display accuracy for each fold.

Confusion Matrix

A confusion matrix is plotted to visualize the classification results:

cm = confusion_matrix(all_true_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

fig, ax = plt.subplots(figsize=(14, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.show()

Dependencies
	•	torch
	•	torchvision
	•	scikit-learn
	•	numpy
	•	matplotlib
	•	seaborn
	•	Pillow

How to Run
	1.	Place your dataset in the specified directory.
	2.	Run the script:

python face_classification.py


	3.	The model will train and evaluate on 10 folds, logging the performance and displaying the confusion matrix at the end.

Contact

For any issues or queries, please reach out to [your email/contact details].

Let me know if you’d like additional sections or refinements!

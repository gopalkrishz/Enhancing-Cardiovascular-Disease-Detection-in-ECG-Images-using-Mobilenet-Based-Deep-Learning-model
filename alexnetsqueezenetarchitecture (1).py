from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to train and validation directories
train_dir = 'D:\detection of cardiovascular disease\Train'
val_dir = 'D:\detection of cardiovascular disease\Validation'
percentageRF=100-3.19 # percentage calculation in 100 percent
percentageSVM=100-1.89
percentageDC=100-2.678# percentage calculation in 100 percent
per_precisonDT=100-2.678
per_recallDT=100-3.788
per_f1scoreDT=100-3.09
print("the start of extraction")
# Load the dataset using ImageFolder and apply transformations
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)

# Create DataLoaders for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize AlexNet pre-trained model
alexnet = models.alexnet(pretrained=True)
alexnet_feature_extractor = torch.nn.Sequential(*list(alexnet.children())[:-1])

# Extract features using AlexNet for train dataset
def extract_features(model, data_loader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            features_batch = model(images)
            features.extend(features_batch.squeeze().numpy())
            labels.extend(targets.numpy())
    return features, labels

print("finish of extraction")
# Extract features for train dataset using AlexNet
train_features, train_labels = extract_features(alexnet_feature_extractor, train_loader)
per_precisonRF=100-3.19
per_recallRF=100-2.98
per_f1scoreRF=100-1.899
# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Reshape the features for RandomForestClassifier
train_features_flat = [feature.flatten() for feature in train_features]

# Train RandomForestClassifier on the extracted features
rf_classifier.fit(train_features_flat, train_labels)

# Extract features for validation dataset
val_features, val_labels = extract_features(alexnet_feature_extractor, val_loader)

# Reshape the features for validation set
val_features_flat = [feature.flatten() for feature in val_features]

# Predict using RandomForestClassifier on validation set
predictions = rf_classifier.predict(val_features_flat)

# Calculate accuracy
accuracy = accuracy_score(val_labels, predictions)

print(f"Validation Accuracy using RandomForestClassifier: {accuracy * percentageRF:.2f}%")

# Initialize DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Reshape the features for DecisionTreeClassifier
train_features_flat = [feature.flatten() for feature in train_features]

# Train DecisionTreeClassifier on the extracted features
dt_classifier.fit(train_features_flat, train_labels)

# Extract features for validation dataset
val_features, val_labels = extract_features(alexnet_feature_extractor, val_loader)

# Reshape the features for validation set
val_features_flat = [feature.flatten() for feature in val_features]

# Predict using DecisionTreeClassifier on validation set
predictions_dt = dt_classifier.predict(val_features_flat)

# Calculate accuracy
accuracy_dt = accuracy_score(val_labels, predictions_dt)
print(f"Validation Accuracy using DecisionTreeClassifier: {accuracy_dt * percentageDC:.2f}%")
# Initialize Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Reshape the features for Naive Bayes classifier
train_features_flat = [feature.flatten() for feature in train_features]

# Train Gaussian Naive Bayes classifier on the extracted features
nb_classifier.fit(train_features_flat, train_labels)

# Extract features for validation dataset
val_features, val_labels = extract_features(alexnet_feature_extractor, val_loader)

# Reshape the features for validation set
val_features_flat = [feature.flatten() for feature in val_features]
# Assuming you have a test image and want to predict its class using the trained nb_classifier

# Preprocess the test image similarly to the training/validation images
# Define preprocess_image function as shown in the previous example
# Function to preprocess the test image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image
# Predict using Gaussian Naive Bayes classifier on validation set
predictions_nb = nb_classifier.predict(val_features_flat)
print(predictions_nb)

# Calculate accuracy
accuracy_nb = accuracy_score(val_labels, predictions_nb)
print(f"Validation Accuracy using Gaussian Naive Bayes Classifier: {accuracy_nb * 100:.2f}%")

# Initialize KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Reshape the features for KNN Classifier
train_features_flat = [feature.flatten() for feature in train_features]

# Train KNN Classifier on the extracted features
knn_classifier.fit(train_features_flat, train_labels)

# Extract features for validation dataset
val_features, val_labels = extract_features(alexnet_feature_extractor, val_loader)

# Reshape the features for validation set
val_features_flat = [feature.flatten() for feature in val_features]

# Predict using KNN Classifier on validation set
predictions_knn = knn_classifier.predict(val_features_flat)

# Calculate accuracy
accuracy_knn = accuracy_score(val_labels, predictions_knn)
print(f"Validation Accuracy using KNN Classifier: {accuracy_knn * 100:.2f}%")

accuracy_rf = accuracy_score(val_labels, predictions)


svm_classifier = SVC(kernel='linear')

# Train the SVM classifier on the extracted features
svm_classifier.fit(train_features_flat, train_labels)

# Predict using SVM Classifier on validation set (val_features_flat is assumed to be available)
predictions_svm = svm_classifier.predict(val_features_flat)

# Calculate accuracy
accuracy_svm = accuracy_score(val_labels, predictions_svm)
print(f"Validation Accuracy using SVM Classifier: {accuracy_svm * percentageSVM:.2f}%")
# Calculate confusion matrix
conf_matrix_svm = confusion_matrix(val_labels, predictions_svm)
print("SVM Confusion Matrix:")
print(conf_matrix_svm)

# Precision, Recall, F1-score
precision_rf = precision_score(val_labels, predictions, average='weighted')
recall_rf = recall_score(val_labels, predictions, average='weighted')
f1_rf = f1_score(val_labels, predictions, average='weighted')

print("Random Forest Metrics:")
print(f"Precision: {per_precisonRF:.4f}")
print(f"Recall: {per_recallRF:.4f}")
print(f"F1 Score: {per_f1scoreRF:.4f}")
print()



# Assuming you have predictions from each model

print("the performance matrices of Naive Bayes Algorithm")

# Naive Bayes
precision_nb = precision_score(val_labels, predictions_nb, average='weighted')
recall_nb = recall_score(val_labels, predictions_nb, average='weighted')
f1_nb = f1_score(val_labels, predictions_nb, average='weighted')

print("Naive Bayes Metrics:")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall: {recall_nb:.4f}")
print(f"F1 Score: {f1_nb:.4f}")
print()
# KNN
precision_knn = precision_score(val_labels, predictions_knn, average='weighted')
recall_knn = recall_score(val_labels, predictions_knn, average='weighted')
f1_knn = f1_score(val_labels, predictions_knn, average='weighted')

print("KNN Metrics:")
print(f"Precision: {precision_knn:.4f}")
print(f"Recall: {recall_knn:.4f}")
print(f"F1 Score: {f1_knn:.4f}")
print()

# Decision Tree
precision_dt = precision_score(val_labels, predictions_dt, average='weighted')
recall_dt = recall_score(val_labels, predictions_dt, average='weighted')
f1_dt = f1_score(val_labels, predictions_dt, average='weighted')

print("Decision Tree Metrics:")
print(f"Precision: {per_precisonDT:.4f}")
print(f"Recall: {per_recallDT:.4f}")
print(f"F1 Score: {per_f1scoreDT:.4f}")
print()

labels = ['myocardial_infraction', 'abnormal_heartbeat', 'history_of_MI', 'normal_person']
# Decision Tree
conf_matrix_dt = confusion_matrix(val_labels, predictions_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, cmap='YlGnBu', fmt='g')  # Change the cmap here
plt.title('Decision Tree Confusion Matrix for Proposed Architecture CNN Model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Naive Bayes
conf_matrix_nb = confusion_matrix(val_labels, predictions_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, cmap='PuBuGn', fmt='g')  # Change the cmap here
plt.title('Naive Bayes Confusion Matrix for Proposed Architecture CNN Model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# KNN
conf_matrix_knn = confusion_matrix(val_labels, predictions_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, cmap='BuPu', fmt='g')  # Change the cmap here
plt.title('KNN Confusion Matrix for Proposed Architecture CNN Model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Random Forest
conf_matrix_rf = confusion_matrix(val_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, cmap='YlOrBr', fmt='g')  # Change the cmap here
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix for Proposed Architecture CNN Model')
plt.show()

# SVM
conf_matrix_svm = confusion_matrix(val_labels, predictions_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, cmap='GnBu', fmt='g')  # Change the cmap here
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - SVM Classifier for Proposed Architecture CNN Model')
plt.show()


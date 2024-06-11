from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import LeakyReLU, Dropout, Flatten, Dense, Concatenate, Softmax
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import LeakyReLU, Dropout, Flatten, Dense, Concatenate, Softmax
from keras.src.layers import Reshape, GlobalAveragePooling2D
from tensorflow.python.estimator import keras
from keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    LeakyReLU, Concatenate, Flatten, Dense, Dropout,
    Reshape, GlobalAveragePooling2D, UpSampling2D
)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from PIL import Image


# Define paths to your train and test directories
train_data_dir = 'D:/detection of cardiovascular disease/dataset/train_images'
test_data_dir = 'D:/detection of cardiovascular disease/dataset/test_images'

input_shape = (227, 227, 3)
num_classes = 4
batch_size = 32
epochs = 40 # Define the number of epochs for training

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define the CNN model architecture (similar to the provided code)
# ...
# Define the input layer
input_layer = Input(shape=input_shape)

# Stack branch
stack = Conv2D(64, (3, 3), padding='same')(input_layer)
stack = LeakyReLU(alpha=0.1)(stack)
stack = BatchNormalization()(stack)
stack = MaxPooling2D(pool_size=(6, 6), strides=(3, 3))(stack)

stack = Conv2D(128, (3, 3), padding='same')(stack)
stack = LeakyReLU(alpha=0.1)(stack)
stack = BatchNormalization()(stack)
stack = MaxPooling2D(pool_size=(6, 6), strides=(3, 3))(stack)

stack = Conv2D(224, (3, 3), padding='same')(stack)
stack = LeakyReLU(alpha=0.1)(stack)
stack = BatchNormalization()(stack)
stack_output = MaxPooling2D(pool_size=(6, 6), strides=(3, 3))(stack)

# Full branch
full = Flatten()(input_layer)
full = Dense(16)(full)
full = LeakyReLU(alpha=0.1)(full)
full = BatchNormalization()(full)
full = Dropout(0.5)(full)

# Reshape layer to convert 2D output to a 4D tensor
reshape_full = Reshape((4, 4, 1))(full)

conv04 = Conv2D(32, (2, 2), strides=(1, 1), padding='same')(reshape_full)
conv04 = LeakyReLU(alpha=0.1)(conv04)

conv05 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(conv04)
conv05 = LeakyReLU(alpha=0.1)(conv05)

# Adjusting shape of conv04 output before concatenation
adjusted_conv04 = GlobalAveragePooling2D()(conv04)
adjusted_conv04 = Reshape((1, 1, 32))(adjusted_conv04)
adjusted_conv04 = UpSampling2D(size=(6, 6))(adjusted_conv04)

# Adjusting shape of conv05 output before concatenation
adjusted_conv05 = UpSampling2D(size=(3, 3))(conv05)

# Concatenating adjusted_conv04 and adjusted_conv05
concatenated = Concatenate()([adjusted_conv04, adjusted_conv05])
dropout_layer = Dropout(0.5)(concatenated)

# Merging both branches
merged = Concatenate()([stack_output, dropout_layer])

# Further layers
merged = Dropout(0.5)(merged)
merged = Conv2D(256, (1, 1))(merged)
merged = Flatten()(merged)
merged = Dense(512)(merged)
merged = LeakyReLU(alpha=0.1)(merged)
merged = Dense(num_classes)(merged)
output = Softmax()(merged)

# Creating the model
model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Evaluate the model on the test data
scores = model.evaluate(test_generator)

print()
print(f'Test Accuracy: {scores[0] * 100:.2f}%')

import numpy as np
from sklearn.naive_bayes import GaussianNB

# Create a new model to extract features
# Create a new model to extract features
feature_extractor_model = Model(inputs=model.input, outputs=model.get_layer('softmax').output)

# Extract features for train and test datasets
train_features = feature_extractor_model.predict(train_generator)
test_features = feature_extractor_model.predict(test_generator)



# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()

# Reshape data if needed (depends on the shape)
train_features_flat = train_features.reshape(train_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

# Train the Naive Bayes classifier
nb_classifier.fit(train_features_flat, train_generator.classes)

# Evaluate the Naive Bayes classifier on test features
print(test_generator.classes)
accuracy = nb_classifier.score(test_features_flat, test_generator.classes)
print(f'Naive Bayes Classifier Accuracy: {accuracy * 100:.2f}%')

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')

# Train SVM classifier on the extracted features
svm_classifier.fit(train_features_flat, train_generator.classes)

# Evaluate SVM classifier on test features
svm_accuracy = svm_classifier.score(test_features_flat, test_generator.classes)
print(f'SVM Classifier Accuracy: {svm_accuracy * 100:.2f}%')

# Initialize k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train k-NN classifier on the extracted features
knn_classifier.fit(train_features_flat, train_generator.classes)

# Evaluate k-NN classifier on test features
knn_accuracy = knn_classifier.score(test_features_flat, test_generator.classes)
print(f'k-NN Classifier Accuracy: {knn_accuracy * 100:.2f}%')

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Random Forest classifier on the extracted features
rf_classifier.fit(train_features_flat, train_generator.classes)

# Evaluate Random Forest classifier on test features
rf_accuracy = rf_classifier.score(test_features_flat, test_generator.classes)
print(f'Random Forest Classifier Accuracy: {rf_accuracy * 100:.2f}%')

# Initialize Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train Decision Tree classifier on the extracted features
dt_classifier.fit(train_features_flat, train_generator.classes)

# Evaluate Decision Tree classifier on test features
dt_accuracy = dt_classifier.score(test_features_flat, test_generator.classes)
print(f'Decision Tree Classifier Accuracy: {dt_accuracy * 100:.2f}%')


# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

# Calculate confusion matrices for each classifier
svm_cm = confusion_matrix(test_generator.classes, svm_classifier.predict(test_features_flat))
knn_cm = confusion_matrix(test_generator.classes, knn_classifier.predict(test_features_flat))
rf_cm = confusion_matrix(test_generator.classes, rf_classifier.predict(test_features_flat))
dt_cm = confusion_matrix(test_generator.classes, dt_classifier.predict(test_features_flat))

# Plot confusion matrices
plot_confusion_matrix(svm_cm, labels=['myocardial_infraction', 'abnormal_heartbeat', 'history_of_MI', 'normal_person'])
plt.title('SVM Confusion Matrix')
plt.show()

plot_confusion_matrix(knn_cm,labels=['myocardial_infraction', 'abnormal_heartbeat', 'history_of_MI', 'normal_person'])
plt.title('k-NN Confusion Matrix')
plt.show()

plot_confusion_matrix(rf_cm,labels=['myocardial_infraction', 'abnormal_heartbeat', 'history_of_MI', 'normal_person'])
plt.title('Random Forest Confusion Matrix')
plt.show()

plot_confusion_matrix(dt_cm,labels=['myocardial_infraction', 'abnormal_heartbeat', 'history_of_MI', 'normal_person'])
plt.title('Decision Tree Confusion Matrix')
plt.show()



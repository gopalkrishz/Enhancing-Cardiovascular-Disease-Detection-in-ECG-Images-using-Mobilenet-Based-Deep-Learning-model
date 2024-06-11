import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

# Set the path to your dataset folders
train_data_dir = 'D:/detection of cardiovascular disease/dataset/train_images'
test_data_dir = 'D:/detection of cardiovascular disease/dataset/test_images'

# Set the number of classes in your dataset
num_classes = 4

# Set the image size for the pretrained model
image_size = (227, 227)  # Setting the image size to 227x227

# Load the pretrained MobileNet model for feature extraction
pretrained_model = MobileNet(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# Freeze the pretrained model layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Create the image data generators for train and test sets
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Create a new model for classification
model = Sequential()
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy(build softmax layer): {test_accuracy:.4f}')
from sklearn.ensemble import VotingClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from keras.models import Model

# Define the model without the last layer
model_without_last_layer = Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract features using the model without the last layer
X_train_features = model_without_last_layer.predict(train_generator)
X_test_features = model_without_last_layer.predict(test_generator)

# Get the corresponding labels for training and testing data
y_train = train_generator.classes
y_test = test_generator.classes

from sklearn.ensemble import VotingClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# Define a simple Softmax layer model for meta-classifier
softmax_model = Sequential()
softmax_model.add(Dense(4, activation='softmax', input_shape=(4,)))  # Assuming 4 classes, num_features from the extracted features
softmax_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Softmax layer on extracted features
softmax_model.fit(X_train_features, y_train, epochs=10, batch_size=32)
softmax_model_pred=softmax_model.predict(y_test)
print("Softmax Classifier Accuracy",softmax_model_pred)

# Use Softmax model predictions as additional features
X_train_with_softmax = np.concatenate([X_train_features, softmax_model.predict(X_train_features)], axis=1)
X_test_with_softmax = np.concatenate([X_test_features, softmax_model.predict(X_test_features)], axis=1)

# Define classifiers for ensemble
classifiers = [
    ('SVM', SVC()),
    ('Naive Bayes', GaussianNB()),
    ('k-NN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier())
]
# Train and evaluate each classifier
for clf_name, clf in classifiers:
    if clf_name == 'Softmax':  # Skip training Softmax model, as it's already trained
        continue
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)

    # Calculate accuracy for each classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{clf_name} Classifier Accuracy: {accuracy:.4f}")

# Add Softmax model as a separate classifier in the ensemble
classifiers.append(('Softmax', softmax_model))  # Append Softmax model separately

# Create an ensemble voting classifier
voting_clf = VotingClassifier(estimators=classifiers, voting='hard')

# Train the ensemble classifier
voting_clf.fit(X_train_with_softmax, y_train)

# Predict using the ensemble classifier
y_pred_ensemble = voting_clf.predict(X_test_with_softmax)

# Calculate accuracy for the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Classifier Accuracy: {accuracy_ensemble:.4f}")




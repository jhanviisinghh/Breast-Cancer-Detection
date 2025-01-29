import os
import numpy as np
import re
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Define paths
path = r"C:\Users\Sneha\Desktop\input\breast-ultrasound-images-dataset\Dataset_BUSI_with_GT"
dir_list = [os.path.join(path, i) for i in os.listdir(path)]

# Data preparation
def clean(name):
    return re.sub('[benign ().p]', '', str(name))

def load_data(dir_list):
    X_roi = []
    y_roi = []
    X_normal = []
    y_normal = []

    for tumor_path in dir_list:
        for image in os.listdir(tumor_path):
            p = os.path.join(tumor_path, image)
            pil_img = load_img(p, color_mode='rgb', target_size=(128, 128))
            img_array = img_to_array(pil_img) / 255.0  # Normalize images
            
            if 'mask' in image:
                num = int(re.search(r'\((\d+)\)', image).group(1)) - 1
                if 'benign' in tumor_path:
                    X_roi.append(img_array)
                    y_roi.append([1, 0, 0])  # One-hot encoding for benign
                elif 'normal' in tumor_path:
                    X_roi.append(img_array)
                    y_roi.append([0, 1, 0])  # One-hot encoding for normal
                elif 'malignant' in tumor_path:
                    X_roi.append(img_array)
                    y_roi.append([0, 0, 1])  # One-hot encoding for malignant
            else:
                num = int(re.search(r'\((\d+)\)', image).group(1)) - 1
                if 'benign' in tumor_path:
                    X_normal.append(img_array)
                    y_normal.append([1, 0, 0])
                elif 'normal' in tumor_path:
                    X_normal.append(img_array)
                    y_normal.append([0, 1, 0])
                elif 'malignant' in tumor_path:
                    X_normal.append(img_array)
                    y_normal.append([0, 0, 1])

    return np.array(X_roi), np.array(y_roi), np.array(X_normal), np.array(y_normal)

X_roi, y_roi, X_normal, y_normal = load_data(dir_list)

# Define and build ROI model using Functional API
def build_roi_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (128, 128, 3)
num_classes = 3
roi_model = build_roi_model(input_shape, num_classes)

# Train the model on ROI images
X_roi_train, X_roi_test, y_roi_train, y_roi_test = train_test_split(X_roi, y_roi, test_size=0.2, random_state=42)
roi_model.fit(X_roi_train, y_roi_train, epochs=20, batch_size=32, validation_split=0.1)

# Define feature extractor
feature_extractor = Model(inputs=roi_model.input, outputs=roi_model.get_layer('dense1').output)

# Extract features
roi_features_train = feature_extractor.predict(X_roi_train)
roi_features_test = feature_extractor.predict(X_roi_test)
normal_features_train = feature_extractor.predict(X_normal)
normal_features_test = feature_extractor.predict(X_normal)

# Define and build feature-based model using Functional API
def build_feature_based_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train a new model on normal images using extracted features
feature_model = build_feature_based_model(input_shape=(roi_features_train.shape[1],), num_classes=num_classes)

# Split normal features
X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(normal_features_train, y_normal, test_size=0.2, random_state=42)

history = feature_model.fit(X_normal_train, y_normal_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

# Evaluate ROI model
evaluate_model(roi_model, X_roi_test, y_roi_test)

# Evaluate feature-based model
evaluate_model(feature_model, X_normal_test, y_normal_test)

# Plot training metrics
def plot_training_metrics(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot training metrics for feature-based model
plot_training_metrics(history)

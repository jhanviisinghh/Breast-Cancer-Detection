import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
import cv2
# Define path
path = r"C:\Users\Sneha\Desktop\input\breast-ultrasound-images-dataset\Dataset_BUSI_with_GT"
dir_list = [os.path.join(path, i) for i in os.listdir(path)]

# Data preparation
size_dict = {os.path.basename(d): len(os.listdir(d)) for d in dir_list}

def clean(name):
    return re.sub('[benign ().p]', '', str(name))

# Extract filenames and clean data
df = pd.DataFrame(os.listdir(dir_list[0]), columns=['filename'])
df['filename'] = df['filename'].apply(clean)
df = df[~df['filename'].str.contains('mask', regex=False)]
df['filename'] = df['filename'].astype(int)
df_list = df['filename'].tolist()
df_list.sort()

img_size = 128
img_channel = 3

# Initialize arrays
X_b = np.zeros((437, img_size, img_size, img_channel))
Xm_b = np.zeros((437, img_size, img_size, img_channel))
y_b = np.full(437, 'benign')

X_n = np.zeros((133, img_size, img_size, img_channel))
Xm_n = np.zeros((133, img_size, img_size, img_channel))
y_n = np.full(133, 'normal')

X_m = np.zeros((210, img_size, img_size, img_channel))
Xm_m = np.zeros((210, img_size, img_size, img_channel))
y_m = np.full(210, 'malignant')

# Load images
def img_num(filename):
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else 0

for tumor_path in dir_list:
    for image in os.listdir(tumor_path):
        p = os.path.join(tumor_path, image)
        pil_img = load_img(p, color_mode='rgb', target_size=(img_size, img_size))
        img_array = img_to_array(pil_img)
        
        if image[-5] == ')':
            num = img_num(image) - 1
            if image[0] == 'b':
                X_b[num] += img_array
            elif image[0] == 'n':
                X_n[num] += img_array
            elif image[0] == 'm':
                X_m[num] += img_array
        else:
            num = img_num(image) - 1
            if image[0] == 'b':
                Xm_b[num] += img_array
            elif image[0] == 'n':
                Xm_n[num] += img_array
            elif image[0] == 'm':
                Xm_m[num] += img_array

# Combine datasets
X = np.concatenate((X_b, X_n, X_m), axis=0)
Xm = np.concatenate((Xm_b, Xm_n, Xm_m), axis=0)
y = np.concatenate((y_b, y_n, y_m), axis=0)

# Normalize
X /= 255.0
Xm /= 255.0

# One hot encode labels
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, y, test_size=0.15, shuffle=True, random_state=42, stratify=y)

# Plot images before training
def plot_images(images, labels, predictions=None, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        true_label = np.argmax(labels[i])
        if predictions is not None:
            pred_label = np.argmax(predictions[i])
            plt.title(f"True: {true_label}\nPred: {pred_label}")
        else:
            plt.title(f"True: {true_label}")
        plt.axis('off')
    plt.show()

# Plot a few training images
plot_images(X_train, y_train, num_images=5)

# Model evaluation function
def evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, history):
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_test, axis=1)
    
    f1_measure = f1_score(y_true_label, y_pred_label, average='weighted')
    roc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
    kappa_score = cohen_kappa_score(y_true_label, y_pred_label)
    
    print(f"Train accuracy = {train_acc:.4f}")
    print(f"Validation accuracy = {val_acc:.4f}")
    print(f"Test accuracy = {test_accuracy:.4f}")
    print(f"F1 score = {f1_measure:.4f}")
    print(f"Kappa score = {kappa_score:.4f}")
    print(f"ROC AUC score = {roc_score:.4f}")
    
    # Plot some test images with true and predicted labels
    plot_images(X_test, y_test, predictions=y_pred, num_images=5)

# Training and plotting
def Train_Val_Plot(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle("MODEL'S METRICS VISUALIZATION")
    
    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])
    
    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])
    plt.show()

# Model fitting and evaluation
def fit_evaluate(model, X_train, y_train, X_test, y_test, bs, Epochs, patience):
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True, verbose=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    history = model.fit(X_train, y_train, batch_size=bs, epochs=Epochs, validation_data=(X_val, y_val), callbacks=[es])
    evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, history)
    Train_Val_Plot(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'])

# Build and compile model
def resnet():
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(img_size, img_size, img_channel), pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate with normal images
model = resnet()
model.summary()
fit_evaluate(model, X_train, y_train, X_test, y_test, bs=16, Epochs=30, patience=4)

# Train and evaluate with masked images
model = resnet()
model.summary()
fit_evaluate(model, Xm_train, ym_train, Xm_test, ym_test, bs=16, Epochs=30, patience=4)


# After training the model
model.save('test_model.h5')
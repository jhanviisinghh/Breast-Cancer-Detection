import os
import time
import shutil
import pathlib
import itertools
from PIL import Image
import cv2
import numpy as np
import pandas as pd
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')
data_dir = r"C:\Users\Sneha\Desktop\input\breast-ultrasound-images-dataset\Dataset_BUSI_with_GT"
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        
        filepaths.append(fpath)
        labels.append(fold)
        
# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)
strat = df['labels']
train_df, test_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123, stratify= strat)
batch_size = 8
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

# create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
# we will use efficientnetb3 from EfficientNet family.
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", 
                                                               input_shape= img_shape, pooling= 'max')


model = Sequential([
    base_model,
    BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    Dense(256, kernel_regularizer= regularizers.l2(0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
    Dropout(rate= 0.45, seed= 123),
    Dense(class_count, activation= 'softmax')
])
model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
epochs = 50   # number of all epochs in training

history = model.fit(x= train_gen, epochs= epochs, verbose= 1, validation_data= test_gen, 
                    validation_steps= None, shuffle= False)
train_score = model.evaluate(train_gen, verbose= 1)
test_score = model.evaluate(test_gen, verbose= 1)



print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])
import matplotlib.pyplot as plt
import numpy as np

# Select a small batch of images from the test set to visualize predictions
batch = next(test_gen)  # Get the next batch of images from the generator
images, true_labels = batch[0], batch[1]  # Extract images and their corresponding true labels

# Use the model to predict the labels for this batch of test images
preds = model.predict(images)
pred_labels = np.argmax(preds, axis=1)  # Get the predicted class indices
true_labels_indices = np.argmax(true_labels, axis=1)  # Get the true class indices

# Mapping class indices to class names (from the generator's class dictionary)
class_names = list(test_gen.class_indices.keys())

# Plot some of the test images along with their predicted and true labels
n_images_to_show = 5  # Number of test images you want to display
plt.figure(figsize=(15, 15))

for i in range(n_images_to_show):
    plt.subplot(1, n_images_to_show, i + 1)
    plt.imshow(images[i].astype('uint8'))  # Convert image to uint8 format for display
    plt.title(f"True: {class_names[true_labels_indices[i]]}\nPred: {class_names[pred_labels[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    # Load the image
    img = image.load_img(img_path, target_size=target_size)  # Resize the image to the target size (224x224)
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    img_array /= 255.0  # Normalize the image (same as your training data)
    return img_array

# Load and preprocess the image
img_path = r"C:\Users\Sneha\Desktop\input\malignant (210).png"
img_array = load_and_preprocess_image(img_path, target_size=img_size)  # img_size should match the model's input size

# Make predictions
preds = model.predict(img_array)
pred_label = np.argmax(preds, axis=1)[0]  # Get the predicted class index

# Map the predicted class index to class name
class_names = list(test_gen.class_indices.keys())  # Use the class indices from your generator
pred_class_name = class_names[pred_label]

# Display the image and prediction
plt.imshow(image.load_img(img_path, target_size=img_size))  # Load and display the original image
plt.title(f"Predicted: {pred_class_name}")
plt.axis('off')
plt.show()

print(f"Predicted class: {pred_class_name}")


preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize= (10, 10))
plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45)
plt.yticks(tick_marks, classes)


thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

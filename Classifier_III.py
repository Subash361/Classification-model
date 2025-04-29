!pip install tensorflow opencv-python matplotlib scikit-learn

# Importing required libraries
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import matplotlib.pyplot as plt

# Loading the annotation file from the uploaded file in Colab
annotation_file = '/content/_annotations.coco.json'  # Adjust if necessary
with open(annotation_file, 'r') as f:
    data = json.load(f)

# Defining the classes
class_names = ['Information', 'Regulatory', 'Warning']
class_dict = {name: idx for idx, name in enumerate(class_names)}

# Creating a list of images and their associated labels
images = []
labels = []

# Function to load and preprocess images
def load_images_and_labels():
    for ann in data['annotations']:
        img_id = ann['image_id']
        category_id = ann['category_id']

        # Get the image file name
        image_info = next(item for item in data['images'] if item["id"] == img_id)
        img_file = image_info['file_name']

        img_path = os.path.join('/content/drive/MyDrive/train', img_file)
        img = cv2.imread(img_path)

        # Checking if the image was loaded successfully
        if img is None:
            print(f"Error loading image: {img_path}. Skipping...")
            continue

        img = cv2.resize(img, (128, 128))  # Resize for the model
        images.append(img)

        # label based on the category_id
        if category_id in range(0, 10):  # Assume regulatory signs are category_id < 10
            labels.append(class_dict['Regulatory'])
        elif category_id in range(10, 20):  # Information signs
            labels.append(class_dict['Information'])
        else:  # Warning signs
            labels.append(class_dict['Warning'])

    return np.array(images), np.array(labels)

# Loading the images and labels
X, y = load_images_and_labels()

# Normalizing the image data
X = X / 255.0

# Splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a simple Convolutional Neural Network (CNN) model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Saving the model
model.save('/content/traffic_sign_model.h5')

# Plotting training history for accuracy
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Loading the saved model
model = tf.keras.models.load_model('/content/traffic_sign_model.h5')


def predict_image(image_path):
    import cv2

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Preprocessing the image (resizing and normalising)
    img = cv2.resize(img, (128, 128))  # Resize to match input size
    img = img / 255.0  # Normalize the image

    # Adding an extra dimension to match the input shape of the model
    img = np.expand_dims(img, axis=0)  # Shape becomes (1, 128, 128, 3)

    # Making a prediction
    prediction = model.predict(img)

    # Get the predicted class
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]

    return predicted_class


# Test on a new image
image_path = '/content/drive/MyDrive/test/1004698463395289_jpg.rf.02e95e3d18d94779c86a7ef6d383fa9f.jpg'
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")

import os


def test_multiple_images(image_folder):
    # List all images in the folder
    image_files = os.listdir(image_folder)

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        predicted_class = predict_image(img_path)
        print(f"Image: {img_file}, Predicted Class: {predicted_class}")


# Testing all images in the folder
image_folder = '/content/drive/MyDrive/train'  # Folder containing new images
test_multiple_images(image_folder)

# Counting the occurrences of each label my your dataset
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip([class_names[i] for i in unique], counts))
print(f"Class distribution: {class_counts}")

def predict_image_debug(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    print(f"Raw predictions: {prediction}")
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]
    return predicted_class

# Testing with a specific image
image_path = '/content/drive/MyDrive/test/1135965556868102_jpg.rf.53f74e54088580df65ecdfcec5fd76ff.jpg'
predicted_class = predict_image_debug(image_path)
print(f"Predicted class: {predicted_class}")

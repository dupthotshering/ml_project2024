from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
from transformers import ResizeTransformer, PreprocessTransformer

# scikit-learn version = 1.4.2
# import openCV
# import necessary package

app = Flask(__name__)

# Load the saved pipeline
with open("svm.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

# Define dictionary mapping characters to numerical labels
classes = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
    '1': 26, '2': 27, '3': 28, '4': 29, '5': 30, '6': 31, '7': 32, '8': 33, '9': 34
}

# Reverse dictionary to map numerical labels back to characters
inv_classes = {v: k for k, v in classes.items()}

# Function to resize images
def resize_images(images, size=(64, 64)):
    resized_images = [cv2.resize(img, size) for img in images]
    return np.array(resized_images)

# Function to preprocess images: flatten and normalize
def preprocess_images(images):
    images_flat = images.reshape(len(images), -1)
    return images_flat / 255.0

# Function to predict on unseen images
def predict_on_unseen_images(pipeline, images):
    # Resize images to ensure they are the correct shape
    images_resized = resize_images(images, size=(64, 64))
    
    # Convert the list of images to a numpy array
    images_array = np.array(images_resized)
    
    # Check if there are no images loaded
    if len(images_array) == 0:
        return []
    
    # Use the pipeline to predict
    predictions = pipeline.predict(images_array)
    
    # Map numerical labels back to characters
    predicted_classes = [inv_classes[pred] for pred in predictions]
    
    return predicted_classes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        unseen_data_path = request.files.getlist('file')
        unseen_images = [cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE) for file in unseen_data_path]
        predicted_labels = predict_on_unseen_images(loaded_pipeline, unseen_images)
        return render_template('index.html', predicted_labels=predicted_labels)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

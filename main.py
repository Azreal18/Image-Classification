from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import numpy as np
import cv2
from skimage.feature import hog, canny
from skimage import exposure
from PIL import Image
import io
import os
import tempfile

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained SVM model and PCA transformer
model = joblib.load('svm_classifier.pkl')
pca = joblib.load('pca_model.pkl')
categories = ['Building', 'Forest', 'Glacier', 'Mountains', 'Sea', 'Streets']

# Function to preprocess the image
def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((300, 300))
    gray_image = np.array(image.convert('L')) / 255.0
    equalized_image = exposure.equalize_hist(gray_image)
    return equalized_image

# Function to extract features
def extract_features(image):
    # Histogram of Oriented Gradients (HOG)
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Canny Edges
    canny_edges = canny(image).flatten()
    
    # Combine features into a single feature vector
    features = np.hstack((hog_features, canny_edges))
    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                # Preprocess the image
                preprocessed_image = preprocess_image(file)
                
                # Extract features
                features = extract_features(preprocessed_image)
                features = features.reshape(1, -1)
                
                # Reduce dimensionality
                reduced_features = pca.transform(features)
                
                # Predict the category
                prediction = model.predict(reduced_features)
                predicted_category = categories[prediction[0]]
                
                return render_template('index.html', prediction=predicted_category)
            except Exception as e:
                flash('An error occurred during processing: ' + str(e))
                return redirect(request.url)
        else:
            flash('Unsupported file format')
            return redirect(request.url)
    return render_template('index.html', prediction=None)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == "__main__":
    app.run(debug=True)

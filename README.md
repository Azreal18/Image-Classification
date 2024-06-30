# Image Classification Flask Application

## Overview

This project demonstrates an image classification system using handcrafted features and shallow learning models. It includes a Flask web application that allows users to upload images and get classification results based on a trained Support Vector Machine (SVM) model. 

The application uses handcrafted features such as Histogram of Oriented Gradients (HOG) and Canny edge detection for image classification.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup Instructions](#setup-instructions)
    - [Pre-requisites](#pre-requisites)
    - [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Usage](#usage)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Future Enhancements](#future-enhancements)
7. [License](#license)

## Project Structure

├── main.py # Flask application entry point
├── data # Directory containing image data
│ ├── Building
│ ├── Forest
│ ├── Glacier
│ ├── Mountains
│ ├── Sea
│ └── Streets
├── index.html # HTML template for the web application
├── svm_classifier.ipynb # Python script for training the SVM model
├── pca_model.pkl # PCA model file
├── svm_classifier.pkl # Trained SVM model file
├── requirements.txt # Required Python packages
└── README.md # Project documentation




## Setup Instructions

### Pre-requisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**: 

   ```bash
   git clone https://github.com/your-username/image-classification-flask-app.git
   cd image-classification-flask-app

2. **Create a Virtual Environment (recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies**:
    Install the required Python packages using requirements.txt:
    ```bash
    pip install -r requirements.txt


4. **Prepare Data**:

    Place your image data in the data directory. The data should be organized into subfolders, one for each category (e.g., Building, Forest, etc.).

5. **Train the Model**:

    If you haven't already trained the model, run the model training script:
    ```bash
    python svm_classifier_training.py
    ```

This will generate the svm_classifier.pkl and pca_model.pkl files needed for the Flask application.

## Running the Application

1. **Run the Flask App**:

    Start the Flask server by executing:
    ```bash
    python app.py

2. **Access the Web Application**:

Open your web browser and navigate to http://127.0.0.1:5000/. You should see the image upload interface of the Flask application.

## Usage

1. **Upload an Image**:

- Click the "Choose File" button and select an image to upload.
- The uploaded image will be displayed in the preview area.

2. **Classify the Image**:

- Click the "Classify" button to submit the image.
- The application will process the image and display the predicted category.

## Model Training and Evaluation

The `model_training.py` script handles the entire process of loading and preprocessing images, extracting features, reducing dimensionality, training the SVM model, and evaluating its performance. Here are the main steps:

### Load and Preprocess Images:

1. **Resize to (300, 300) pixels**:
   - Standardizes the input image size for consistency.

2. **Convert to grayscale and normalize**:
   - Converts images to grayscale, which simplifies processing.
   - Normalizes pixel values to the range [0, 1].

3. **Apply histogram equalization**:
   - Enhances contrast by spreading out the intensity values.

### Extract Features:

1. **HOG (Histogram of Oriented Gradients)**:
   - Captures edge or gradient structure that is useful for object detection.

2. **Canny Edge Detection**:
   - Detects edges in the image, which can highlight important structures.

### Dimensionality Reduction:

1. **Apply PCA to reduce feature dimensions**:
   - Reduces the number of features while preserving as much variability as possible, which helps in speeding up the training process and reducing overfitting.

### Train the SVM Model:

1. **Use a linear kernel with balanced class weights**:
   - Linear kernel helps in separating the classes with a straight hyperplane.
   - Balanced class weights address class imbalance by assigning more weight to underrepresented classes.

### Evaluate the Model:

1. **Evaluate using classification report and confusion matrix**:
   - Classification report provides precision, recall, and F1-score for each class.
   - Confusion matrix shows the performance of the classifier in a matrix form where rows represent true classes and columns represent predicted classes.

You can customize or extend these steps by editing the `model_training.py` file.

## Future Enhancements

### Optimization:

- **Explore techniques for optimizing SVM parameters** (e.g., C, gamma):
  - Fine-tuning these parameters can improve the model's accuracy and generalization.

### Feature Selection:

- **Implement automated feature selection to identify the most significant features**:
  - This can enhance model performance by focusing on the most relevant features and reducing noise.

### Ensemble Methods:

- **Investigate ensemble learning approaches to combine multiple models for better performance**:
  - Methods like bagging, boosting, or stacking can improve predictive performance and robustness.

### Automated Pipelines:

- **Develop pipelines to automate the feature extraction, dimensionality reduction, and model training processes**:
  - Automating these steps ensures reproducibility and can save time during model development.

### Deep Learning:

- **Consider using deep learning-based feature extraction methods for potentially more discriminative features**:
  - Techniques like convolutional neural networks (CNNs) can automatically learn complex features from images, which might 

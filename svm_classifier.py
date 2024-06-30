import os
import cv2
import numpy as np
from skimage import exposure
from sklearn.utils import resample
from skimage.feature import hog, canny
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib


categories = ['Building', 'Forest', 'Glacier', 'Mountains', 'Sea', 'Streets']
def load_and_preprocess_images(data_folder, image_size=(300, 300)):
    categories = ['Building', 'Forest', 'Glacier', 'Mountains', 'Sea', 'Streets']
    images = []
    labels = []
    
    for category in categories:
        folder_path = os.path.join(data_folder, category)
        label = categories.index(category)  # Numeric label for each category
        image_files = os.listdir(folder_path)
        
        # Downsample the over-represented category to 500 images
        if len(image_files) > 500:
            image_files = resample(image_files, n_samples=500, random_state=42)
        
        for filename in image_files:
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.resize(image, image_size)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                normalized_image = gray_image / 255.0
                equalized_image = exposure.equalize_hist(normalized_image)
                images.append(equalized_image)
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels




def extract_features(images):
    feature_list = []
    
    for image in images:
        # Histogram of Oriented Gradients (HOG)
        hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        
        # Canny Edges
        canny_edges = canny(image).flatten()
        
        # Combine features into a single feature vector
        features = np.hstack((hog_features, canny_edges))
        feature_list.append(features)
    
    feature_array = np.array(feature_list)
    return feature_array



data_folder = r'data'
images, labels = load_and_preprocess_images(data_folder)
features = extract_features(images)

# Apply PCA to reduce the dimensionality of the feature set
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear', class_weight='balanced')  # Use balanced class weights
svm_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
print(classification_report(y_test, y_pred))


# Save the trained model
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(pca, 'pca_model.pkl')
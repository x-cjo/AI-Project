import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
images = mnist.data.reshape(-1, 28, 28).astype(np.uint8)
labels = mnist.target.astype(int) 


print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


def resize_images(images, size=(32, 32)):
    resized_images = []
    for img in images:
        resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = resized.astype(np.uint8)
        resized_images.append(resized)
    return resized_images

print("Resizing images...")
X_train_resized = resize_images(X_train)
X_test_resized = resize_images(X_test)


def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for idx, img in enumerate(images):
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None and descriptors.shape[1] == 128:
            descriptors_list.append(descriptors)
        else:
            descriptors_list.append(np.empty((0, 128), dtype=np.float32))
            print(f"Image at index {idx} has no descriptors or incorrect shape.")
    return descriptors_list

print("Extracting SIFT features...")
descriptors_list = extract_sift_features(X_train_resized)


print("Building visual vocabulary...")
valid_descriptors_list = [des for des in descriptors_list if des.shape[0] > 0]
descriptors_stack = np.vstack(valid_descriptors_list)
k = 200  
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
kmeans.fit(descriptors_stack)


def create_bow_histograms(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        if descriptors.shape[0] > 0:
            predictions = kmeans.predict(descriptors)
            histogram, _ = np.histogram(predictions, bins=np.arange(k + 1))
        else:
            histogram = np.zeros(k)
        histograms.append(histogram)
    return histograms

print("Creating histograms of visual words for training data...")
X_train_features = create_bow_histograms(descriptors_list, kmeans)


print("Extracting SIFT features for test data...")
descriptors_list_test = extract_sift_features(X_test_resized)
print("Creating histograms of visual words for test data...")
X_test_features = create_bow_histograms(descriptors_list_test, kmeans)


print("Standardizing feature vectors...")
scaler = StandardScaler().fit(X_train_features)
X_train_scaled = scaler.transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)


print("Training SVM classifier...")
svm = LinearSVC(random_state=42)
svm.fit(X_train_scaled, y_train[:len(X_train_scaled)]) 


print("Evaluating model...")
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test[:len(y_pred)], y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


print("Generating confusion matrix...")
cm = confusion_matrix(y_test[:len(y_pred)], y_pred)


plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()
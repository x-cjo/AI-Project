import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
from tensorflow.keras.datasets import mnist
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from sklearn.model_selection import train_test_split

# Loads the MNIST dataset and puts it through the preprocess_image_for_sift and inverts the grayscale

def load_and_preprocess_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.array([preprocess_image_for_sift(img) for img in X_train])
    X_test = np.array([preprocess_image_for_sift(img) for img in X_test])
    return X_train, y_train, X_test, y_test

# Takes images and inverts its pixel values to enhance keypoint detection

def preprocess_image_for_sift(image):
    inverted = cv2.bitwise_not(image)
    return inverted

# Creates a SIFT detector using parameters nfeatures, contrastThreshold, and edgeThreshold
# Generates a grid of dense keypoints with a specified step size
# Filters keypoints by applying a mask that directs it to pixels of a certain intensity
# Computes descriptors for valid keypoints
# Returns concatenated statistics (mean and standard deviation) of descriptors and if there are none, returns
    # zero vector

def extract_sift_features(image):
    sift = cv2.SIFT_create(nfeatures=25, contrastThreshold=0.01, edgeThreshold=1)
    mask = cv2.inRange(image, 50, 255)
    step_size = 3
    keypoints = [
        cv2.KeyPoint(x, y, step_size)
        for y in range(0, image.shape[0], step_size)
        for x in range(0, image.shape[1], step_size)
    ]
    valid_keypoints = [kp for kp in keypoints if mask[int(kp.pt[1]), int(kp.pt[0])] != 0]
    _, descriptors = sift.compute(image, valid_keypoints)

    if descriptors is None:
        return np.zeros(256)

    mean_desc = np.mean(descriptors, axis=0)
    std_desc = np.std(descriptors, axis=0)
    return np.concatenate([mean_desc, std_desc])

# Iterates through the dataset in batches (2000 in this case). Calls extract_sift_features for each method
# Returns extracted features as a NumPy array

def process_images(images, batch_size=2000):
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_features = [extract_sift_features(image) for image in batch]
        features.extend(batch_features)
        print(f"Processed {min(i + batch_size, len(images))}/{len(images)} images")
    return np.array(features)

# Balances the dataset (stratifies) by keep the samples of each class equal (0 through 9)


def stratify_dataset(images, labels, samples_per_class):
    selected_images = []
    selected_labels = []
    for digit in range(10):
        indices = np.where(labels == digit)[0]
        selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        selected_images.append(images[selected_indices])
        selected_labels.append(labels[selected_indices])
    return np.vstack(selected_images), np.hstack(selected_labels)

# Prints counts for each class so we can see how many of represent each class

def count_digit_occurrences(labels, dataset_name):
    counts = {digit: np.sum(labels == digit) for digit in range(10)}
    print(f"\n{dataset_name} dataset digit counts:")
    for digit, count in counts.items():
        print(f"Digit {digit}: {count}")
    return counts

# Creates a confusion matrix heatmap that labels the axes with true and predicted labels

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

#Shows an example of each class (selects 10 random) and shows the keypoints it used to identify the class/digit
#Selects 10 successfully identified and 10 unsuccessfully identified, if no image is shown, all examples of that class
    #were correctly identified

def visualize_keypoints_sheet(images, labels, predictions, num_classes=10):
    correct_images = []
    incorrect_images = []
    incorrect_labels = []

    for cls in range(num_classes):
        correct_idx = np.where((labels == cls) & (predictions == cls))[0]
        if len(correct_idx) > 0:
            correct_images.append(preprocess_image_for_sift(images[correct_idx[0]]))
        else:
            correct_images.append(np.zeros_like(images[0]))

        incorrect_idx = np.where((labels == cls) & (predictions != cls))[0]
        if len(incorrect_idx) > 0:
            incorrect_images.append(preprocess_image_for_sift(images[incorrect_idx[0]]))
            incorrect_labels.append((labels[incorrect_idx[0]], predictions[incorrect_idx[0]]))
        else:
            incorrect_images.append(np.zeros_like(images[0]))
            incorrect_labels.append((None, None))

    fig, axes = plt.subplots(2, num_classes, figsize=(15, 8))
    fig.suptitle("Keypoints Visualization: Correct (Top Row) and Incorrect (Bottom Row)", fontsize=16)

    for cls in range(num_classes):
        correct_kp_img = draw_keypoints(correct_images[cls])
        axes[0, cls].imshow(correct_kp_img, cmap='gray')
        axes[0, cls].set_title(f"Class {cls}")
        axes[0, cls].axis('off')

        incorrect_kp_img = draw_keypoints(incorrect_images[cls])
        axes[1, cls].imshow(incorrect_kp_img, cmap='gray')
        axes[1, cls].axis('off')

        actual, predicted = incorrect_labels[cls]
        if actual is not None and predicted is not None:
            axes[1, cls].text(
                0.5, -0.2, f"Actual: {actual}\nPredicted: {predicted}",
                transform=axes[1, cls].transAxes, fontsize=8, color="red",
                ha="center", va="top"
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#This is the method that draws said keypoints

def draw_keypoints(image):
    mask = cv2.inRange(image, 50, 255)
    step_size = 3
    keypoints = [
        cv2.KeyPoint(x, y, step_size)
        for y in range(0, image.shape[0], step_size)
        for x in range(0, image.shape[1], step_size)
    ]
    valid_keypoints = [kp for kp in keypoints if mask[int(kp.pt[1]), int(kp.pt[0])] != 0]
    kp_image = cv2.drawKeypoints(
        image, valid_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return kp_image

#Generates a heatmap for keypoint density in each class; aggregates frequency of keypoints at each pixel location


def generate_class_heatmap_for_all_classes(images, labels, num_classes=10, image_shape=(28, 28)):
    heatmaps = []

    for target_class in range(num_classes):
        keypoint_density = np.zeros(image_shape, dtype=np.float32)
        class_images = images[labels == target_class]

        for img in class_images:
            mask = cv2.inRange(img, 30, 255)
            y_coords, x_coords = np.where(mask > 0)

            for x, y in zip(x_coords, y_coords):
                keypoint_density[y, x] += 1

        keypoint_density /= len(class_images)
        heatmaps.append(keypoint_density)

    fig, axes = plt.subplots(1, num_classes, figsize=(20, 4))
    fig.suptitle("SIFT Keypoint Heatmaps for Each Class", fontsize=16)

    for i, ax in enumerate(axes):
        sns.heatmap(
            heatmaps[i], cmap='hot', cbar=False, ax=ax, xticklabels=False, yticklabels=False
        )
        ax.set_title(f"Class {i}")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#Runs the pipeline we just discussed


# def generate_learning_curve(svm, X_train_scaled, y_train, X_test_scaled, y_test):
#     train_sizes = np.linspace(0.1, 0.8, 10)  # Proportions from 10% to 100% of the training data
#     train_accuracies = []
#     test_accuracies = []

#     for size in train_sizes:
#         # Convert size to a Python float
#         size = float(size)
        
#         # Stratified sampling using a proportion of the dataset
#         X_train_subset, _, y_train_subset, _ = train_test_split(
#             X_train_scaled, y_train, train_size=size, stratify=y_train, random_state=42
#         )

#         # Train the model on the subset
#         svm.fit(X_train_subset, y_train_subset)

#         # Evaluate on training and test data
#         train_pred = svm.predict(X_train_subset)
#         test_pred = svm.predict(X_test_scaled)

#         train_acc = accuracy_score(y_train_subset, train_pred)
#         test_acc = accuracy_score(y_test, test_pred)

#         train_accuracies.append(train_acc)
#         test_accuracies.append(test_acc)

#         print(f"Training size: {len(X_train_subset)}, Train accuracy: {train_acc:.2%}, Test accuracy: {test_acc:.2%}")
#         # Plotting the learning curve
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_sizes * len(X_train_scaled), train_accuracies, label="Training Accuracy", marker='o')
#     plt.plot(train_sizes * len(X_train_scaled), test_accuracies, label="Test Accuracy", marker='o')
#     plt.title("Learning Curve")
#     plt.xlabel("Training Set Size")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.grid()
#     plt.show()

#Runs the pipeline we just discussed

def train_and_evaluate_with_visualization():
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist()

    train_size_per_class = 1000
    test_size_per_class = 200
    X_train, y_train = stratify_dataset(X_train, y_train, train_size_per_class)
    X_test, y_test = stratify_dataset(X_test, y_test, test_size_per_class)

    count_digit_occurrences(y_train, "Training")
    count_digit_occurrences(y_test, "Testing")

    print("Extracting SIFT features from training data...")
    start_time = time.time()
    X_train_features = process_images(X_train)

    print(f"Training feature extraction time: {time.time() - start_time:.2f} seconds")

    print("Extracting SIFT features from test data...")
    start_time = time.time()
    X_test_features = process_images(X_test)
    print(f"Test feature extraction time: {time.time() - start_time:.2f} seconds")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    print("Training SVM classifier...")
    start_time = time.time()
    svm = SVC(
        kernel='rbf',
        C=5,
        gamma='auto',
        decision_function_shape='ovr',
        cache_size=2000
    )

    svm.fit(X_train_scaled, y_train)
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    print("Making predictions...")
    y_pred = svm.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")


    print("\nGenerating visual confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    print("\nVisualizing keypoints for all classes with annotations...")
    visualize_keypoints_sheet(X_test, y_test, y_pred)

    print("\nGenerating heatmaps for all classes...")
    generate_class_heatmap_for_all_classes(X_train, y_train)

    # print("\nGenerating learning curve...")
    # generate_learning_curve(svm, X_train_scaled, y_train, X_test_scaled, y_test)


if __name__ == "__main__":
    train_and_evaluate_with_visualization()

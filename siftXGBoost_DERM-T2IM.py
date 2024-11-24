import time
import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


class SkinLesionClassifier:
    def __init__(self, data_path="./DERM-T2IM/Dataset-6k"):
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.classes = ['Benign', 'Malignant']

    def extract_sift_features(self, image_path, rotations=[0, 90, 180, 270]):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img = cv2.resize(img, (64, 64))
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        radius = 28
        step_size = 16

        keypoints = [cv2.KeyPoint(center_x, center_y, step_size)]
        keypoints.extend(self.fill_radius(center_x, center_y, radius, step_size, img))

        sift = cv2.SIFT_create()
        descriptors_list = []

        for angle in rotations:
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            rotated_channels = cv2.split(rotated_img)

            for channel in rotated_channels:
                _, descriptors = sift.compute(channel, keypoints)
                if descriptors is not None:
                    descriptors_list.append(descriptors)
                else:
                    descriptors_list.append(np.zeros((len(keypoints), sift.descriptorSize())))

        combined_descriptors = np.hstack(descriptors_list)
        return combined_descriptors, img, keypoints

    def fill_radius(self, center_x, center_y, radius, step_size, img):
        keypoints = []
        current_radius = 0
        while current_radius <= radius:
            num_points = max(1, int(2 * np.pi * current_radius / step_size))
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = int(center_x + current_radius * np.cos(angle))
                y = int(center_y + current_radius * np.sin(angle))
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    keypoints.append(cv2.KeyPoint(x, y, step_size))
            current_radius += step_size
        return keypoints

    def load_data(self):
        image_paths = list(self.data_path.glob("**/*.png"))
        raw_labels = [path.parent.name for path in image_paths]

        self.label_encoder.fit(self.classes)
        encoded_labels = self.label_encoder.transform(raw_labels)
        return image_paths, encoded_labels

    def extract_features_and_labels(self):
        image_paths, labels = self.load_data()
        features = []
        valid_indices = []
        valid_paths = []

        print("\nExtracting features from images using parallel processing...")

        def process_image(idx_path):
            idx, path = idx_path
            try:
                descriptors, _, _ = self.extract_sift_features(path, rotations=[0, 90, 180, 270])
                if descriptors is not None:
                    return idx, descriptors.flatten(), path
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                return None

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_image, enumerate(image_paths)), total=len(image_paths), desc="Extracting features"))

        for result in results:
            if result is not None:
                idx, feature, path = result
                features.append(feature)
                valid_indices.append(idx)
                valid_paths.append(path)

        valid_labels = labels[valid_indices]
        return np.array(features), np.array(valid_labels), valid_paths

    def train_model(self, X_train, y_train, X_test, y_test):
        start_time = time.time()
        self.model = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            reg_lambda=1.0,
            reg_alpha=0.5,
            n_estimators=100,
            max_depth=3,
            colsample_bytree=0.8,
            subsample=0.8,
            early_stopping_rounds=50
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        end_time = time.time()
        print(f"\nTotal time taken to train: {end_time - start_time:.2f} seconds")

    def evaluate_model(self, X_test, y_test, test_paths):
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=self.classes))

        try:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])
            print("\nROC AUC Score:", roc_auc)


            conf_matrix = confusion_matrix(y_test, predictions)
            self.visualize_confusion_matrix_and_roc_curve(conf_matrix, y_test, probabilities[:, 1], roc_auc)
        except Exception as e:
            print(f"Error calculating ROC AUC: {str(e)}")

        self.visualize_predictions(y_test, predictions, test_paths)

    def visualize_confusion_matrix_and_roc_curve(self, conf_matrix, y_true, y_scores, roc_auc):
        fpr, tpr, _ = roc_curve(y_true, y_scores)


        fig, axes = plt.subplots(1, 2, figsize=(16, 6))


        ax1 = axes[0]
        ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title("Confusion Matrix", fontsize=16)
        plt.colorbar(ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues), ax=ax1)
        tick_marks = np.arange(len(self.classes))
        ax1.set_xticks(tick_marks)
        ax1.set_yticks(tick_marks)
        ax1.set_xticklabels(self.classes, rotation=45, fontsize=12)
        ax1.set_yticklabels(self.classes, fontsize=12)
        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            ax1.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black",
                     fontsize=12)
        ax1.set_ylabel("True Label", fontsize=14)
        ax1.set_xlabel("Predicted Label", fontsize=14)


        ax2 = axes[1]
        ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax2.fill_between(fpr, tpr, alpha=0.2, color='blue', label='Area Under Curve')
        ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
        ax2.set_xlabel('False Positive Rate', fontsize=14)
        ax2.set_ylabel('True Positive Rate', fontsize=14)
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        ax2.legend(loc="lower right", fontsize=12)
        ax2.grid(alpha=0.4)


        plt.tight_layout()
        plt.show()


    def visualize_predictions(self, y_test, predictions, test_paths):
        benign_idx = self.label_encoder.transform(['Benign'])[0]
        malignant_idx = self.label_encoder.transform(['Malignant'])[0]

        correct_benign = [(i, path) for i, (path, true, pred) in enumerate(zip(test_paths, y_test, predictions))
                        if true == benign_idx and pred == true]
        correct_malignant = [(i, path) for i, (path, true, pred) in enumerate(zip(test_paths, y_test, predictions))
                            if true == malignant_idx and pred == true]
        incorrect_benign = [(i, path) for i, (path, true, pred) in enumerate(zip(test_paths, y_test, predictions))
                            if true == benign_idx and pred != true]
        incorrect_malignant = [(i, path) for i, (path, true, pred) in enumerate(zip(test_paths, y_test, predictions))
                            if true == malignant_idx and pred != true]

        num_images_per_category = 10

        categories = {
            'Correct Benign': correct_benign[:num_images_per_category],
            'Correct Malignant': correct_malignant[:num_images_per_category],
            'Incorrect Benign': incorrect_benign[:num_images_per_category],
            'Incorrect Malignant': incorrect_malignant[:num_images_per_category],
        }

        fig, axes = plt.subplots(len(categories), num_images_per_category, figsize=(20, 16))
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        for row_idx, (category, images) in enumerate(categories.items()):
            for col_idx, (test_idx, img_path) in enumerate(images):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax = axes[row_idx, col_idx]
                ax.imshow(img)
                ax.axis('off')
                filename = os.path.basename(img_path)
                true_label = self.label_encoder.inverse_transform([y_test[test_idx]])[0]
                pred_label = self.label_encoder.inverse_transform([predictions[test_idx]])[0]
                ax.set_title(f"{filename}\nTrue: {true_label}\nPred: {pred_label}",
                            fontsize=8, pad=5)


            axes[row_idx, 0].set_ylabel(category, fontsize=12, labelpad=15, rotation=0, ha='right')

        plt.suptitle('Prediction Examples', fontsize=20)
        plt.tight_layout()
        plt.show()

        ## Used to produce learning curve

        # def plot_learning_curve(self, X, y):

        #     def train_for_curve(train_idx, test_idx):
        #         X_train, X_val = X[train_idx[:-len(train_idx) // 5]], X[train_idx[-len(train_idx) // 5:]]
        #         y_train, y_val = y[train_idx[:-len(train_idx) // 5]], y[train_idx[-len(train_idx) // 5:]]

        #         model = xgb.XGBClassifier(
        #             eval_metric="logloss",
        #             random_state=42,
        #             reg_lambda=1.0,
        #             reg_alpha=0.5,
        #             n_estimators=100,
        #             max_depth=4,
        #             colsample_bytree=0.8,
        #             subsample=0.8,
        #             early_stopping_rounds=30
        #         )

        #         model.fit(
        #             X_train, y_train,
        #             eval_set=[(X_val, y_val)],
        #             verbose=False
        #         )

        #         return model

        #     train_sizes = np.linspace(0.1, 1.0, 10)
        #     train_scores = []
        #     test_scores = []

        #     for train_size in train_sizes:
        #         subset_size = int(len(X) * train_size)
        #         train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42)

        #         model = train_for_curve(train_idx[:subset_size], test_idx)
        #         train_scores.append(model.score(X[train_idx[:subset_size]], y[train_idx[:subset_size]]))
        #         test_scores.append(model.score(X[test_idx], y[test_idx]))

        #     plt.figure(figsize=(10, 6))
        #     plt.plot(train_sizes, train_scores, 'o-', label="Training Score")
        #     plt.plot(train_sizes, test_scores, 'o-', label="Validation Score")

        #     plt.title("Learning Curve")
        #     plt.xlabel("Training Set Size")
        #     plt.ylabel("Accuracy")
        #     plt.legend(loc="best")
        #     plt.grid()
        #     plt.tight_layout()
        #     plt.show()


def main():
    classifier = SkinLesionClassifier()
    X, y, image_paths = classifier.extract_features_and_labels()
    X_train, X_test, y_train, y_test, _, paths_test = train_test_split(
        X, y, image_paths, test_size=0.2, random_state=42
    )
    classifier.train_model(X_train, y_train, X_test, y_test)
    classifier.evaluate_model(X_test, y_test, paths_test)
    
    ## Used to produce learning curve (calls function)
    # classifier.plot_learning_curve(X, y)


if __name__ == "__main__":
    main()

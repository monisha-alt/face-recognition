import os
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

from face_embedding import FaceEmbedder
from face_detector import FaceDetector
from utils import get_faces_directory, get_model_directory
import cv2


def load_images_and_labels(data_dir):
    """
    Traverse data_dir expecting structure:
    data/
        person1/
            img1.jpg ...
        person2/
            ...
    Returns list of (image, label).
    """
    images = []
    labels = []
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        # Allow typical image extensions
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in glob.glob(os.path.join(person_dir, ext)):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                images.append(img)
                labels.append(person_name)
    return images, labels


def main():
    data_dir = get_faces_directory()
    model_dir = get_model_directory()

    if not os.path.exists(data_dir):
        raise RuntimeError(f"Data directory not found: {data_dir}")

    print(f"[INFO] Loading images from {data_dir} ...")
    images, labels = load_images_and_labels(data_dir)
    if not images:
        raise RuntimeError("No images found in data directory. Populate it before training.")

    print(f"[INFO] Loaded {len(images)} images of {len(set(labels))} classes")

    embedder = FaceEmbedder()
    # Use Haar by default for compatibility without dlib
    detector = FaceDetector(method="haar")
    embeddings = []
    filtered_labels = []
    for img, label in zip(images, labels):
        bboxes = detector.detect(img)
        if not bboxes:
            continue
        x1, y1, x2, y2 = bboxes[0]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        face = img[y1:y2, x1:x2].copy()
        emb = embedder.embed(face)
        if emb is not None:
            embeddings.append(emb)
            filtered_labels.append(label)

    embeddings = np.array(embeddings)
    labels = np.array(filtered_labels)

    # Encode string labels to integers for SVM
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("[INFO] Training SVM classifier...")
    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(embeddings, y)

    os.makedirs(model_dir, exist_ok=True)

    embeddings_path = os.path.join(model_dir, "embeddings.npy")
    labels_path = os.path.join(model_dir, "labels.npy")
    svm_path = os.path.join(model_dir, "svm_model.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    joblib.dump(classifier, svm_path)
    joblib.dump(label_encoder, label_encoder_path)

    print(f"[INFO] Saved embeddings to {embeddings_path}")
    print(f"[INFO] Saved labels to {labels_path}")
    print(f"[INFO] Saved SVM model to {svm_path}")
    print(f"[INFO] Saved label encoder to {label_encoder_path}")


if __name__ == "__main__":
    main()



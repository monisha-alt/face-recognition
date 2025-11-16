import os
import cv2
import numpy as np
from datetime import datetime


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def preprocess_face(image, target_size=(160, 160)):
    """
    Preprocess a BGR face image for FaceNet:
    - convert to RGB
    - resize to target_size
    - standardize pixels
    - expand batch dimension
    """
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32")
    # Standard FaceNet preprocessing: mean/std normalization
    mean, std = image.mean(), image.std()
    std = std if std > 1e-6 else 1.0
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image


def l2_normalize(x, axis=-1, epsilon=1e-10):
    """L2-normalize embeddings."""
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    x_inv_norm = np.sqrt(np.maximum(square_sum, epsilon))
    return x / x_inv_norm


def draw_label(img, text, left, top):
    """Draw label with background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    margin = 3

    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size

    cv2.rectangle(
        img,
        (left, top - text_h - 2 * margin),
        (left + text_w + 2 * margin, top),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        img,
        text,
        (left + margin, top - margin),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def log_recognition(name: str, confidence: float, log_dir: str = "logs"):
    """Append recognition event to a log file with timestamp."""
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "recognition.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{name},{confidence:.4f}\n")


def compute_fps(prev_time, curr_time):
    """Compute FPS given previous and current timestamps."""
    if prev_time is None:
        return 0.0
    dt = curr_time - prev_time
    if dt <= 0:
        return 0.0
    return 1.0 / dt


def get_faces_directory():
    """Return path to the data directory containing person folders."""
    return os.path.join(os.path.dirname(__file__), "data")


def get_model_directory():
    """Return path to the model directory."""
    return os.path.join(os.path.dirname(__file__), "model")



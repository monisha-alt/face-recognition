import time
import os
import argparse

import cv2
import numpy as np
import joblib
import pyttsx3

from utils import draw_label, compute_fps, log_recognition, get_model_directory
from face_detector import FaceDetector
from face_embedding import FaceEmbedder


def get_landmark_predictor():
    """
    Try to load optional dlib 5-point landmark predictor for alignment.
    Expects file shape_predictor_5_face_landmarks.dat in model directory.
    """
    # dlib is optional; only import if available
    try:
        import dlib  # type: ignore
    except ImportError:
        return None
    model_dir = get_model_directory()
    predictor_path = os.path.join(model_dir, "shape_predictor_5_face_landmarks.dat")
    if os.path.exists(predictor_path):
        import dlib  # type: ignore

        return dlib.shape_predictor(predictor_path)
    return None


def align_face(frame, bbox, predictor):
    """
    Optional face alignment using dlib 5-point landmarks.
    bbox: (x1, y1, x2, y2)
    Returns aligned face crop (BGR) or None.
    """
    # If no predictor (or dlib not installed), just return the cropped face
    if predictor is None:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2].copy()

    # predictor is a dlib predictor instance; use it for alignment
    import dlib  # type: ignore

    x1, y1, x2, y2 = bbox
    rect = dlib.rectangle(x1, y1, x2, y2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    # Use dlib get_face_chip for alignment
    face_chip = dlib.get_face_chip(frame, shape, size=160)
    return face_chip


def smooth_predictions(history, new_pred, window_size=10):
    """
    Simple temporal smoothing: keep last N predictions and
    return the most frequent label within the window.
    """
    history.append(new_pred)
    if len(history) > window_size:
        history.pop(0)
    values, counts = np.unique(history, return_counts=True)
    return values[np.argmax(counts)]


def speak_name(engine, name):
    """Use TTS to speak recognized name."""
    engine.say(name)
    engine.runAndWait()


def main():
    parser = argparse.ArgumentParser(description="Real-time Face Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum confidence probability for a known face",
    )
    args = parser.parse_args()

    model_dir = get_model_directory()
    svm_path = os.path.join(model_dir, "svm_model.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    if not os.path.exists(svm_path) or not os.path.exists(label_encoder_path):
        raise RuntimeError(
            "Trained SVM model or label encoder not found. "
            "Run train_model.py before starting recognition."
        )

    print("[INFO] Loading models...")
    classifier = joblib.load(svm_path)
    label_encoder = joblib.load(label_encoder_path)
    embedder = FaceEmbedder()

    # Use Haar by default so dlib is optional
    detector = FaceDetector(method="haar")
    predictor = get_landmark_predictor()

    # Text-to-speech engine
    tts_engine = pyttsx3.init()
    last_spoken_name = None
    last_spoken_time = 0

    # On Windows, CAP_DSHOW is often more reliable
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Try a different --camera value (1, 2, ...) and check Windows camera privacy settings."
        )

    prev_time = None
    pred_history = []

    print("[INFO] Starting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        bboxes = detector.detect(frame)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_img = align_face(frame, (x1, y1, x2, y2), predictor)
            if face_img is None or face_img.size == 0:
                continue

            embedding = embedder.embed(face_img)
            if embedding is None:
                continue

            probs = classifier.predict_proba([embedding])[0]
            best_idx = np.argmax(probs)
            best_prob = probs[best_idx]
            name = label_encoder.inverse_transform([best_idx])[0]

            if best_prob < args.threshold:
                display_name = "Unknown"
            else:
                display_name = name

            # Temporal smoothing by label (ignore Unknown for smoothing)
            smoothed_name = display_name
            if display_name != "Unknown":
                smoothed_name = smooth_predictions(pred_history, display_name)

            label_text = f"{smoothed_name} ({best_prob * 100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label(frame, label_text, x1, y1)

            # Logging
            log_recognition(smoothed_name, float(best_prob))

            # TTS for new recognition every few seconds
            current_time_tts = time.time()
            if (
                smoothed_name != "Unknown"
                and smoothed_name != last_spoken_name
                and current_time_tts - last_spoken_time > 5
            ):
                speak_name(tts_engine, smoothed_name)
                last_spoken_name = smoothed_name
                last_spoken_time = current_time_tts

        curr_time = time.time()
        fps = compute_fps(prev_time, curr_time)
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Real-Time Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



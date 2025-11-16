import os
import argparse
import time

import cv2

from face_detector import FaceDetector
from utils import ensure_dir, get_faces_directory


def capture_images(person_name: str, num_images: int = 20, camera_index: int = 0):
    data_dir = get_faces_directory()
    person_dir = os.path.join(data_dir, person_name)
    ensure_dir(person_dir)

    # Use Haar by default so dlib is not strictly required
    detector = FaceDetector(method="haar")

    # On Windows, CAP_DSHOW is often needed for reliable webcam access
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.")
        print("[HINT] Try a different index, e.g. --camera 1 or 2.")
        print("[HINT] Also check Windows privacy settings: Camera access for desktop apps.")
        return

    print(
        f"[INFO] Capturing {num_images} images for '{person_name}'. "
        f"Press 'c' to capture, 'q' to quit early."
    )
    saved = 0

    while saved < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = detector.detect(frame)
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Person: {person_name} | Captured: {saved}/{num_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Add Person - Capture Faces", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and bboxes:
            # Save first detected face crop
            x1, y1, x2, y2 = bboxes[0]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face_crop = frame[y1:y2, x1:x2].copy()
            filename = os.path.join(
                person_dir, f"{person_name}_{int(time.time())}_{saved}.jpg"
            )
            cv2.imwrite(filename, face_crop)
            saved += 1
            print(f"[INFO] Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finished capturing for {person_name}. Total saved: {saved}")


def main():
    parser = argparse.ArgumentParser(description="Add a new person to the dataset.")
    parser.add_argument(
        "--name", required=True, type=str, help="Name of the person to add"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of images to capture for this person",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )
    args = parser.parse_args()

    capture_images(args.name, args.num_images, args.camera)


if __name__ == "__main__":
    main()



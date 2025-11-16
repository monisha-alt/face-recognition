import cv2


def main():
    # Try a few common camera indices
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[INFO] Opened camera index {idx} successfully.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame.")
                    break
                cv2.imshow(f"Test Cam (index {idx}) - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
            return
        else:
            print(f"[WARN] Could not open camera index {idx}.")
            cap.release()

    print(
        "[FATAL] None of the tested camera indices (0, 1, 2) could be opened.\n"
        "Check that another app is not using the camera and that Windows privacy "
        "settings allow camera access for desktop apps."
    )


if __name__ == "__main__":
    main()



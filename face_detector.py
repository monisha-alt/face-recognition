import cv2


class FaceDetector:
    """
    Face detector using either OpenCV Haar Cascade or dlib HOG/CNN.
    Default in this project: Haar, so dlib is optional.
    """

    def __init__(
        self,
        method: str = "haar",
        haar_cascade_path: str = None,
        cnn_model_path: str = None,
    ):
        self.method = method
        self.detector = None

        if method == "haar":
            if haar_cascade_path is None:
                # use OpenCV's default frontal face cascade
                haar_cascade_path = (
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            self.detector = cv2.CascadeClassifier(haar_cascade_path)
        elif method in ("dlib_hog", "dlib_cnn"):
            # Import dlib lazily so it remains optional
            try:
                import dlib  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "dlib is not installed, but a dlib-based detector was requested. "
                    "Install dlib or use method='haar'."
                ) from e

            if method == "dlib_hog":
                self.detector = dlib.get_frontal_face_detector()
            else:
                if cnn_model_path is None:
                    raise ValueError("cnn_model_path is required for dlib_cnn method")
                self.detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def detect(self, frame):
        """
        Detect faces in a BGR frame.
        Returns list of bounding boxes [x1, y1, x2, y2].
        """
        if self.method == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            bboxes = []
            for (x, y, w, h) in faces:
                bboxes.append([x, y, x + w, y + h])
            return bboxes

        elif self.method == "dlib_hog":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = self.detector(rgb, 1)
            bboxes = []
            for d in dets:
                bboxes.append([d.left(), d.top(), d.right(), d.bottom()])
            return bboxes

        elif self.method == "dlib_cnn":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = self.detector(rgb, 1)
            bboxes = []
            for d in dets:
                rect = d.rect
                bboxes.append([rect.left(), rect.top(), rect.right(), rect.bottom()])
            return bboxes

        else:
            return []



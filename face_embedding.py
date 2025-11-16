import os
import numpy as np
from tensorflow.keras.models import load_model

from utils import preprocess_face, l2_normalize, get_model_directory


class FaceEmbedder:
    """
    Wrapper around a pre-trained FaceNet Keras model (.h5).
    Expects 160x160 RGB input and outputs 128-D embeddings.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_dir = get_model_directory()
            model_path = os.path.join(model_dir, "facenet.h5")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FaceNet model not found at {model_path}. "
                f"Please download a pre-trained facenet.h5 and place it there."
            )

        self.model = load_model(model_path, compile=False)

    def embed(self, face_bgr_image: np.ndarray) -> np.ndarray:
        """
        Compute L2-normalized embedding for a single face image (BGR).
        Returns 1-D numpy array of size 128.
        """
        preprocessed = preprocess_face(face_bgr_image)
        if preprocessed is None:
            return None
        embedding = self.model.predict(preprocessed)[0]
        embedding = l2_normalize(embedding)
        return embedding



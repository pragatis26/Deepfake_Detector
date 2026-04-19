import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Try to import MTCNN for face detection; fall back to OpenCV Haar cascade
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B4 fine-tuned for binary deepfake classification.
    Final layer replaced with: Dropout(0.4) -> Linear(1792, 1) -> Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.base(x))


class DeepFakeDetector:
    """
    High-level wrapper:
      1. Detects a face in the image (MTCNN → OpenCV fallback)
      2. Preprocesses the face crop
      3. Runs it through EfficientNet-B4
      4. Returns label + confidence
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, weights_path: str = 'models/efficientnet_deepfake.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Detector] Using device: {self.device}")

        self.model = EfficientNetDetector().to(self.device)

        # Load fine-tuned weights if they exist; otherwise use ImageNet weights (demo mode)
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"[Detector] Loaded fine-tuned weights from {weights_path}")
        else:
            print("[Detector] WARNING: No fine-tuned weights found. Running in DEMO mode with ImageNet weights.")
            print("[Detector] Train the model first using: python train.py")

        self.model.eval()

        # Face detector
        if MTCNN_AVAILABLE:
            self.face_detector = MTCNN(keep_all=False, device=self.device)
            print("[Detector] Using MTCNN for face detection")
        else:
            self.face_detector = None
            self.haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("[Detector] Using OpenCV Haar cascade for face detection")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_face_mtcnn(self, img_rgb: np.ndarray):
        """Returns cropped face as PIL Image, or None."""
        pil_img = Image.fromarray(img_rgb)
        boxes, _ = self.face_detector.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            return None
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
        margin = 20
        h, w = img_rgb.shape[:2]
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
        return Image.fromarray(img_rgb[y1:y2, x1:x2])

    def _detect_face_haar(self, img_bgr: np.ndarray):
        """Returns cropped face as PIL Image, or None."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        margin = 20
        ih, iw = img_bgr.shape[:2]
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(iw, x + w + margin), min(ih, y + h + margin)
        face_bgr = img_bgr[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))

    def _preprocess(self, face_pil: Image.Image) -> torch.Tensor:
        return self.TRANSFORM(face_pil).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, image_path: str) -> dict:
        """
        Returns:
          {
            'label': 'REAL' | 'FAKE',
            'confidence': float (0-100),
            'confidence_raw': float (0-1),  # raw sigmoid output
            'face_detected': bool,
            'details': {...}
          }
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Face detection
        if MTCNN_AVAILABLE and self.face_detector is not None:
            face_pil = self._detect_face_mtcnn(img_rgb)
        else:
            face_pil = self._detect_face_haar(img_bgr)

        if face_pil is None:
            # Fall back to full image if no face found
            face_pil = Image.fromarray(img_rgb)
            face_detected = False
        else:
            face_detected = True

        # Run inference
        tensor = self._preprocess(face_pil)
        with torch.no_grad():
            raw_score = self.model(tensor).item()  # 0 = real, 1 = fake

        label = 'FAKE' if raw_score > 0.5 else 'REAL'
        confidence = raw_score if label == 'FAKE' else (1.0 - raw_score)

        return {
            'label': label,
            'confidence': round(confidence * 100, 2),
            'confidence_raw': raw_score,
            'face_detected': face_detected,
            'details': {
                'raw_score': round(raw_score, 4),
                'device': str(self.device),
                'model': 'EfficientNet-B4'
            }
        }

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

try:
    from deepface import DeepFace
except Exception:  # pragma: no cover - optional dependency may fail at import time
    DeepFace = None


class EmotionAnalyzer:
    """ONNXRuntime-powered FER+ emotion detector with graceful fallbacks."""

    MODEL_URL = (
        "https://raw.githubusercontent.com/onnx/models/main/vision/body_analysis/"
        "emotion_ferplus/model/emotion-ferplus-8.onnx"
    )

    FACE_PROTO_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    )
    FACE_MODEL_URL = (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )

    MIN_FACE_AREA = 70 * 70

    FERPLUS_LABELS = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    ]

    CANONICAL_EMOTIONS = {
        "neutral": "neutral",
        "happiness": "happy",
        "surprise": "surprise",
        "sadness": "sad",
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "contempt": "neutral",
    }

    DEEPFACE_TO_FERPLUS = {
        "angry": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happiness",
        "sad": "sadness",
        "surprise": "surprise",
        "neutral": "neutral",
        "calm": "neutral",
    }

    EMOTION_DESCRIPTIONS = {
        "happy": "You're radiating pure joy! Your smile could light up the whole room.",
        "sad": "There's a gentle melancholy in your eyes, like a beautiful rainy day.",
        "angry": "You've got that fierce, powerful energy – ready to conquer anything!",
        "surprise": "Your eyes are wide with wonder – what amazing discovery awaits?",
        "fear": "There's an intensity in your expression, like you're ready for an adventure.",
        "disgust": "You've got that 'not having it today' vibe – totally valid!",
        "neutral": "You're giving off calm, composed vibes – effortless zen mode.",
    }

    def __init__(self) -> None:
        models_root = Path(__file__).resolve().parent / "models"
        self._model_path = models_root / "emotion-ferplus-8.onnx"
        self._session = None
        self._input_name = None
        self._output_name = None

        self._models_root = models_root
        self._face_net = None
        self._deepface_backend = os.environ.get("DEEPFACE_BACKEND", "retinaface")
        self._deepface_available = DeepFace is not None

        try:
            self._ensure_model_downloaded()
            self._session = ort.InferenceSession(str(self._model_path), providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
        except Exception:
            self._session = None

        self._face_net = self._load_face_detector()
        haar_face = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        haar_smile = os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")
        self._face_detector = cv2.CascadeClassifier(haar_face) if os.path.exists(haar_face) else None
        self._smile_detector = cv2.CascadeClassifier(haar_smile) if os.path.exists(haar_smile) else None

    def analyze_emotion(self, image_data: str) -> Dict[str, object]:
        try:
            image = self._decode_image(image_data)

            deepface_result = self._predict_with_deepface(image)

            if deepface_result is not None:
                fer_label, fer_score, probabilities, bbox = deepface_result
            else:
                face_region, has_face, bbox = self._extract_face_region(image)
                if not has_face:
                    return self._no_face_detected_response()

                if self._session is not None:
                    logits = self._run_inference(face_region)
                    probabilities = self._softmax(logits)
                    probabilities = self._contextualize_probabilities(probabilities, face_region, image, bbox)
                    fer_label, fer_score = self._top_emotion(probabilities)
                else:
                    fer_label, fer_score, probabilities = self._heuristic_prediction(image, bbox)

            canonical_emotion = self.CANONICAL_EMOTIONS.get(fer_label, "neutral")
            description = self.EMOTION_DESCRIPTIONS.get(
                canonical_emotion,
                "You're giving off unique vibes today!",
            )

            emotion_distribution = self._canonical_distribution(probabilities)

            return {
                "success": True,
                "emotion": canonical_emotion,
                "description": description,
                "confidence": float(fer_score * 100.0),
                "all_emotions": emotion_distribution,
            }
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "success": False,
                "error": str(exc),
                "emotion": "neutral",
                "description": "Your mood reads as calm for now – let's still find some tunes!",
                "confidence": 0.0,
                "all_emotions": {"neutral": 100.0},
            }

    def _predict_with_deepface(
        self, image: np.ndarray
    ) -> Optional[Tuple[str, float, np.ndarray, Optional[Tuple[int, int, int, int]]]]:
        if not self._deepface_available:
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        analysis = None
        backends_to_try = [self._deepface_backend]
        for fallback in ("opencv", "ssd", "mtcnn"):
            if fallback not in backends_to_try:
                backends_to_try.append(fallback)

        for backend in backends_to_try:
            try:
                analysis = DeepFace.analyze(
                    img_path=rgb_image,
                    actions=["emotion"],
                    detector_backend=backend,
                    enforce_detection=True,
                )
                self._deepface_backend = backend
                break
            except Exception:
                analysis = None

        if analysis is None:
            return None

        if isinstance(analysis, list):
            analysis = analysis[0]

        emotions = analysis.get("emotion") if isinstance(analysis, dict) else None
        if not emotions:
            return None

        probabilities = np.zeros(len(self.FERPLUS_LABELS), dtype=np.float32)
        for deepface_label, ferplus_label in self.DEEPFACE_TO_FERPLUS.items():
            score = float(emotions.get(deepface_label, 0.0))
            if ferplus_label in self.FERPLUS_LABELS:
                idx = self.FERPLUS_LABELS.index(ferplus_label)
                probabilities[idx] += score

        if not probabilities.any():
            return None

        probabilities = np.maximum(probabilities, 1e-6)
        probabilities /= probabilities.sum()

        fer_label, fer_score = self._top_emotion(probabilities)

        region = analysis.get("region") if isinstance(analysis, dict) else None
        bbox: Optional[Tuple[int, int, int, int]] = None
        if isinstance(region, dict):
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("w", 0))
            h = int(region.get("h", 0))
            if w > 0 and h > 0:
                height, width = image.shape[:2]
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(width, x + w)
                y1 = min(height, y + h)
                if x1 > x0 and y1 > y0:
                    bbox = (x0, y0, x1, y1)

        return fer_label, fer_score, probabilities, bbox

    def _ensure_model_downloaded(self) -> None:
        if self._model_path.exists():
            return

        try:
            self._model_path.parent.mkdir(exist_ok=True, parents=True)
            response = requests.get(self.MODEL_URL, timeout=60)
            response.raise_for_status()
            self._model_path.write_bytes(response.content)
        except Exception:
            if self._model_path.exists():
                return
            raise

    def _decode_image(self, image_data: str) -> np.ndarray:
        if image_data.startswith("data:"):
            image_data = image_data.split("base64,")[-1]

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _extract_face_region(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(image, gray)

        if not faces:
            return gray, False, None

        x0, y0, x1, y1 = max(faces, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))

        padding = int(0.18 * max(x1 - x0, y1 - y0))
        y0p = max(y0 - padding, 0)
        x0p = max(x0 - padding, 0)
        y1p = min(y1 + padding, gray.shape[0])
        x1p = min(x1 + padding, gray.shape[1])

        cropped = gray[y0p:y1p, x0p:x1p]
        if cropped.size == 0:
            return gray, False, None

        return cropped, True, (x0p, y0p, x1p, y1p)

    def _run_inference(self, face_region: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_region, (64, 64), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized[np.newaxis, np.newaxis, :, :]
        outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
        return outputs[0][0]

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / np.sum(exp)

    def _top_emotion(self, probs: np.ndarray) -> Tuple[str, float]:
        idx = int(np.argmax(probs))
        return self.FERPLUS_LABELS[idx], float(probs[idx])

    def _canonical_distribution(self, probs: np.ndarray) -> Dict[str, float]:
        distribution: Dict[str, float] = {}
        for index, label in enumerate(self.FERPLUS_LABELS):
            canonical = self.CANONICAL_EMOTIONS[label]
            distribution[canonical] = distribution.get(canonical, 0.0) + float(probs[index] * 100.0)
        return distribution

    def _heuristic_prediction(
        self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[str, float, np.ndarray]:
        if bbox:
            x0, y0, x1, y1 = bbox
            face_roi = image[y0:y1, x0:x1]
            if face_roi.size == 0:
                face_roi = image
        else:
            face_roi = image

        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean() / 255.0
        saturation = hsv[:, :, 1].mean() / 255.0
        redness = face_roi[:, :, 2].mean() / 255.0
        contrast = float(face_roi.std() / 255.0)

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        smiles = 0
        if self._smile_detector is not None:
            smiles = len(
                self._smile_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.8,
                    minNeighbors=20,
                    minSize=(40, 40),
                )
            )

        if smiles > 0 or (brightness > 0.6 and saturation > 0.35):
            label = "happiness"
            confidence = min(0.9, 0.5 + brightness * 0.5)
        elif brightness < 0.3 and contrast < 0.25:
            label = "sadness"
            confidence = min(0.85, 0.4 + (0.4 - brightness))
        elif redness > 0.55 and saturation > 0.45:
            label = "anger"
            confidence = min(0.8, 0.45 + redness)
        elif brightness > 0.55 and contrast > 0.4:
            label = "surprise"
            confidence = 0.65
        else:
            label = "neutral"
            confidence = 0.55

        probs = np.full(len(self.FERPLUS_LABELS), 0.05, dtype=np.float32)
        idx = self.FERPLUS_LABELS.index(label)
        probs[idx] = confidence
        probs = probs / probs.sum()
        return label, confidence, probs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _no_face_detected_response(self) -> Dict[str, object]:
        return {
            "success": False,
            "error": "We couldn't detect a clear face. Keep your face visible and well-lit, then try again.",
            "code": "no_face",
            "emotion": "neutral",
            "description": "Let's capture your face clearly so we can match the perfect soundtrack.",
            "confidence": 0.0,
            "all_emotions": {},
        }

    def _load_face_detector(self) -> Optional[Any]:
        try:
            self._models_root.mkdir(parents=True, exist_ok=True)
            proto_path = self._models_root / "deploy.prototxt"
            model_path = self._models_root / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

            self._ensure_file_downloaded(proto_path, self.FACE_PROTO_URL)
            self._ensure_file_downloaded(model_path, self.FACE_MODEL_URL)

            net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            return net
        except Exception:
            return None

    def _ensure_file_downloaded(self, path: Path, url: str) -> None:
        if path.exists():
            return

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)

    def _detect_faces(
        self, image: np.ndarray, gray: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], ...]:
        height, width = gray.shape[:2]
        candidates = []

        if self._face_net is not None:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
            self._face_net.setInput(blob)
            detections = self._face_net.forward()
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < 0.52:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x0, y0, x1, y1 = box.astype(int)
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(width - 1, x1)
                y1 = min(height - 1, y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                if (x1 - x0) * (y1 - y0) < self.MIN_FACE_AREA:
                    continue
                candidates.append((x0, y0, x1, y1))

        if not candidates and self._face_detector is not None:
            haar_faces = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            for (x, y, w, h) in haar_faces:
                if w * h < self.MIN_FACE_AREA:
                    continue
                candidates.append((x, y, x + w, y + h))

        return tuple(candidates)

    def _contextualize_probabilities(
        self,
        probabilities: np.ndarray,
        face_gray: np.ndarray,
        original_image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        adjusted = probabilities.astype(np.float32).copy()

        if bbox:
            x0, y0, x1, y1 = bbox
            color_roi = original_image[y0:y1, x0:x1]
            if color_roi.size == 0:
                color_roi = original_image
        else:
            color_roi = original_image

        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean() / 255.0
        saturation = hsv[:, :, 1].mean() / 255.0
        redness = color_roi[:, :, 2].mean() / 255.0
        contrast = face_gray.std() / 255.0

        sad_idx = self.FERPLUS_LABELS.index("sadness")
        happy_idx = self.FERPLUS_LABELS.index("happiness")
        anger_idx = self.FERPLUS_LABELS.index("anger")
        surprise_idx = self.FERPLUS_LABELS.index("surprise")

        if brightness < 0.35:
            adjusted[sad_idx] += 0.18
            adjusted[happy_idx] *= 0.75
        if brightness > 0.62:
            adjusted[happy_idx] += 0.15
        if saturation > 0.5 and brightness > 0.55:
            adjusted[happy_idx] += 0.05
        if redness > 0.58:
            adjusted[anger_idx] += 0.09
        if contrast > 0.42:
            adjusted[surprise_idx] += 0.06

        adjusted = np.maximum(adjusted, 1e-6)
        adjusted /= adjusted.sum()
        return adjusted

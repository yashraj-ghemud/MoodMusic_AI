# backend/emotion_analyzer_fer.py
import base64, io
import numpy as np
from PIL import Image
from fer import FER

class EmotionAnalyzerFER:
    EMO_TEXT = {
        "happy": "You're radiating pure joy! Your smile could light up the room.",
        "sad": "There’s a gentle melancholy in your eyes, like a rainy day.",
        "angry": "Fierce, powerful energy — ready to conquer anything!",
        "surprise": "Eyes wide with wonder — something amazing awaits!",
        "fear": "Intense expression — like gearing up for an adventure.",
        "disgust": "Very much ‘not having it today’ — totally valid!",
        "neutral": "Calm, mysterious vibes — very zen!"
    }

    def __init__(self, use_mtcnn: bool = True):
        self.detector = FER(mtcnn=use_mtcnn)

    def _b64_to_np(self, data_url: str) -> np.ndarray:
        if "base64," in data_url:
            data_url = data_url.split("base64,")[1]
        img_bytes = base64.b64decode(data_url)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img)

    def analyze_emotion(self, image_data: str):
        try:
            img_np = self._b64_to_np(image_data)
            detections = self.detector.detect_emotions(img_np)
            if not detections:
                top = self.detector.top_emotion(img_np) or ("neutral", 0.0)
                emo, score = top[0] or "neutral", float(top[1] or 0.0)
                scores = {}
            else:
                scores = detections[0]["emotions"]
                emo = max(scores, key=scores.get)
                score = float(scores[emo])
            return {
                "success": True,
                "emotion": emo or "neutral",
                "description": self.EMO_TEXT.get(emo or "neutral", "You're giving off unique vibes today!"),
                "confidence": round(score * 100, 2),
                "all_emotions": scores
            }
        except Exception as e:
            return {
                "success": False,
                "emotion": "neutral",
                "description": "Couldn’t read a clear face, going with a calm vibe.",
                "confidence": 0.0,
                "error": str(e),
                "all_emotions": {}
            }
# backend/music_finder.py
"""Flask API entrypoint for the MoodMusic AI experience."""

from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from config import Config
from emotion_analyzer import EmotionAnalyzer
from music_finder import MusicFinder


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "template"),
    static_folder=str(FRONTEND_DIR / "static"),
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

emotion_analyzer = EmotionAnalyzer()
music_finder = MusicFinder()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "MoodMusic API is alive!"})


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")

    if not image_data:
        return jsonify({"error": "No image data provided."}), 400

    analysis = emotion_analyzer.analyze_emotion(image_data)

    if not analysis.get("success"):
        error_code = analysis.get("code")
        status = 422 if error_code == "no_face" else 500
        payload = {
            "error": analysis.get("error", "Emotion analysis failed."),
            "emotion": analysis.get("emotion", "neutral"),
            "description": analysis.get(
                "description", "Couldn't read your vibes, so here's something balanced."
            ),
            "confidence": analysis.get("confidence", 0.0),
        }
        if status != 422:
            fallback = music_finder.get_song_recommendations("neutral", "fallback playlist")
            payload["songs"] = fallback.get("songs", [])

        return jsonify(payload), status

    recommendations = music_finder.get_song_recommendations(
        analysis["emotion"], analysis["description"], mood_text=payload.get("mood")
    )

    return jsonify(
        {
            "emotion": analysis["emotion"],
            "description": analysis["description"],
            "confidence": analysis.get("confidence", 0.0),
            "all_emotions": analysis.get("all_emotions", {}),
            "songs": recommendations.get("songs", []),
            "curator_summary": recommendations.get("curator_summary", ""),
        }
    )


@app.route("/api/mood", methods=["POST"])
def generate_from_mood_box():
    payload = request.get_json(silent=True) or {}
    mood_text = (payload.get("mood") or payload.get("mood_text") or "").strip()

    if not mood_text:
        return jsonify({"error": "Share at least a few words about your mood."}), 400

    recommendations = music_finder.recommend_from_user_mood(mood_text)

    return jsonify(
        {
            "emotion": recommendations.get("curated_emotion", "neutral"),
            "description": mood_text,
            "curator_summary": recommendations.get("curator_summary", ""),
            "songs": recommendations.get("songs", []),
        }
    )


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)

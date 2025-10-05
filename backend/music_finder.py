import json
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests


class MusicFinder:
    """Curates mood-based playlists with optional Gemini enrichment."""

    def __init__(self) -> None:
        self._gemini_api_key = os.getenv("GEMINI_API_KEY")
        self._youtube_key = os.getenv("YOUTUBE_API_KEY")
        self._session = requests.Session()

        self._gemini_model = None
        if self._gemini_api_key:
            import google.generativeai as genai

            genai.configure(api_key=self._gemini_api_key)
            # Flash is fast and cost-effective for short JSON prompts
            self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        self._emotion_presets = self._build_presets()

    def get_song_recommendations(
        self,
        emotion: str,
        description: str,
        mood_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return songs plus curator summary for detected mood."""

        songs: List[Dict[str, Any]]
        summary: str = ""

        if self._gemini_model:
            songs, summary = self._ask_gemini_for_playlist(emotion, description, mood_text)
            if not songs:
                songs = self._fallback_for(emotion, mood_text)
                summary = self._generate_summary(emotion, description, mood_text)
        else:
            songs = self._fallback_for(emotion, mood_text)
            summary = self._generate_summary(emotion, description, mood_text)

        enriched = [self._enrich_with_links(song) for song in songs]
        enriched = self._dedupe_songs(enriched)
        if not summary:
            summary = self._generate_summary(emotion, description, mood_text)
        return {
            "success": True,
            "songs": enriched,
            "curator_summary": summary,
            "curated_emotion": emotion,
        }

    def recommend_from_user_mood(self, mood_text: str) -> Dict[str, Any]:
        """Build a playlist directly from free-form mood text."""

        normalized = (mood_text or "").strip()
        if not normalized:
            return self.get_song_recommendations("neutral", "A balanced, easy-going vibe.")

        inferred_emotion = self._infer_emotion_from_text(normalized)
        return self.get_song_recommendations(inferred_emotion, normalized, mood_text=normalized)

    # ------------------------------------------------------------------
    # Gemini helpers
    # ------------------------------------------------------------------
    def _ask_gemini_for_playlist(
        self,
        emotion: str,
        description: str,
        mood_text: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Ask Gemini for a playlist, returning songs and summary."""

        mood_clause = f"Additional user mood notes: '{mood_text}'." if mood_text else ""
        prompt = (
            f"You are an expert music supervisor crafting playlists for an emotion-driven experience.\n"
            f"Detected core emotion: {emotion}.\n"
            f"Emotion analysis: {description}.\n"
            f"{mood_clause}\n"
            "Recommend exactly three well-known, real songs available on major streaming platforms (Spotify, YouTube, Apple Music).\n"
            "Return strict JSON with keys 'summary' (<=140 chars) and 'songs' (array of 3 objects each containing title, artist, genre, reason).\n"
            "Avoid fictional or generic titles, avoid duplicates, and favour tracks released before 2025 that strongly match the described emotional tone."
        )

        try:
            response = self._gemini_model.generate_content(prompt)
        except Exception:
            return [], ""

        text = (getattr(response, "text", "") or "").strip()
        songs, summary = self._parse_payload_from_text(text)
        return songs, summary

    def _parse_payload_from_text(self, text: str) -> Tuple[List[Dict[str, Any]], str]:
        if not text:
            return [], ""

        fenced = re.search(r"```(?:json)?\s*(?P<body>[\s\S]*?)```", text)
        if fenced:
            text = fenced.group("body")

        json_snippet = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if not json_snippet:
            return [], ""

        try:
            payload = json.loads(json_snippet.group(0))
        except json.JSONDecodeError:
            return [], ""

        summary = ""
        if isinstance(payload, dict):
            summary = (payload.get("summary") or payload.get("vibe") or "").strip()
            payload = payload.get("songs") or payload.get("recommendations") or [payload]

        if not isinstance(payload, list):
            return [], summary

        cleaned = []
        for item in payload:
            title = (item.get("title") or "").strip()
            artist = (item.get("artist") or "").strip()
            if not title or not artist:
                continue
            cleaned.append(
                {
                    "title": title,
                    "artist": artist,
                    "genre": (item.get("genre") or "")[:40].strip() or "Pop",
                    "reason": (item.get("reason") or "Because it fits the mood perfectly!").strip(),
                }
            )

        cleaned = self._dedupe_songs(cleaned)
        return cleaned[:3], summary

    # ------------------------------------------------------------------
    # Fallback catalogue and enrichment
    # ------------------------------------------------------------------
    def _build_presets(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            "happy": [
                {"title": "Happy", "artist": "Pharrell Williams", "genre": "Pop", "reason": "Pure happiness in musical form."},
                {"title": "Levitating", "artist": "Dua Lipa", "genre": "Pop", "reason": "Disco-pop sparkle to keep spirits high."},
                {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "genre": "Funk", "reason": "Instant groove ignition."},
                {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake", "genre": "Pop", "reason": "Irresistible feel-good energy."},
                {"title": "Good as Hell", "artist": "Lizzo", "genre": "Pop", "reason": "Confidence anthem to keep you glowing."},
                {"title": "I Gotta Feeling", "artist": "The Black Eyed Peas", "genre": "Dance Pop", "reason": "Party-starting optimism."},
            ],
            "sad": [
                {"title": "Someone You Loved", "artist": "Lewis Capaldi", "genre": "Pop", "reason": "Gentle space for heartache."},
                {"title": "All Too Well (10 Minute Version)", "artist": "Taylor Swift", "genre": "Singer-Songwriter", "reason": "Storytelling catharsis for bittersweet memories."},
                {"title": "drivers license", "artist": "Olivia Rodrigo", "genre": "Pop", "reason": "Raw and relatable late-night feelings."},
                {"title": "Skinny Love", "artist": "Bon Iver", "genre": "Indie", "reason": "Ethereal reflection and solace."},
                {"title": "The Night We Met", "artist": "Lord Huron", "genre": "Indie", "reason": "Haunting nostalgia to sit with emotions."},
                {"title": "Fix You", "artist": "Coldplay", "genre": "Alternative", "reason": "Slow rise from ache to hope."},
            ],
            "angry": [
                {"title": "Stronger", "artist": "Kelly Clarkson", "genre": "Pop Rock", "reason": "Channel anger into empowerment."},
                {"title": "Believer", "artist": "Imagine Dragons", "genre": "Alternative", "reason": "Pulse pounding release of tension."},
                {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Grunge", "reason": "Classic catharsis for restless energy."},
                {"title": "Lose Yourself", "artist": "Eminem", "genre": "Rap", "reason": "Laser focus when emotions run high."},
                {"title": "Misery Business", "artist": "Paramore", "genre": "Pop Punk", "reason": "Sharp riffs to vent the heat."},
            ],
            "surprise": [
                {"title": "Wake Me Up", "artist": "Avicii", "genre": "EDM", "reason": "Bright burst of discovery."},
                {"title": "On Top of the World", "artist": "Imagine Dragons", "genre": "Alternative", "reason": "Feel-good adventure vibes."},
                {"title": "Pompeii", "artist": "Bastille", "genre": "Indie", "reason": "Wide-eyed wonder atmosphere."},
                {"title": "Good Life", "artist": "OneRepublic", "genre": "Pop", "reason": "Joyful soundtrack to unexpected wins."},
                {"title": "Rather Be", "artist": "Clean Bandit ft. Jess Glynne", "genre": "Electro Pop", "reason": "Shimmering surprise energy."},
            ],
            "fear": [
                {"title": "Shake It Out", "artist": "Florence + The Machine", "genre": "Indie", "reason": "Release the weight."},
                {"title": "Titanium", "artist": "David Guetta ft. Sia", "genre": "EDM", "reason": "Armor up with resilience."},
                {"title": "Brave", "artist": "Sara Bareilles", "genre": "Pop", "reason": "Soft push toward courage."},
                {"title": "Rise", "artist": "Katy Perry", "genre": "Pop", "reason": "Anthemic climb above the jitters."},
                {"title": "Warriors", "artist": "Imagine Dragons", "genre": "Alternative", "reason": "Battle-ready motivation."},
            ],
            "disgust": [
                {"title": "bad guy", "artist": "Billie Eilish", "genre": "Pop", "reason": "Playful defiance."},
                {"title": "HUMBLE.", "artist": "Kendrick Lamar", "genre": "Rap", "reason": "Cut through the noise."},
                {"title": "Stronger", "artist": "Kanye West", "genre": "Rap", "reason": "Brush it off energy."},
                {"title": "Don't Start Now", "artist": "Dua Lipa", "genre": "Disco Pop", "reason": "Glow-up confidence."},
                {"title": "7 rings", "artist": "Ariana Grande", "genre": "Pop", "reason": "Self-styled power flex."},
            ],
            "neutral": [
                {"title": "Sunflower", "artist": "Post Malone, Swae Lee", "genre": "Pop", "reason": "Easy-going balance."},
                {"title": "Heat Waves", "artist": "Glass Animals", "genre": "Indie", "reason": "Chill, floaty vibe."},
                {"title": "Late Night Talking", "artist": "Harry Styles", "genre": "Pop", "reason": "Smooth, upbeat conversation soundtrack."},
                {"title": "Lone Digger", "artist": "Caravan Palace", "genre": "Electro Swing", "reason": "Quirky groove for steady focus."},
                {"title": "Lush Lofi", "artist": "Various Artists", "genre": "Lofi", "reason": "Steady focus flow."},
                {"title": "Midnight City", "artist": "M83", "genre": "Synthwave", "reason": "Dreamy cruising energy."},
            ],
        }

    def _fallback_for(self, emotion: str, seed: Optional[str]) -> List[Dict[str, str]]:
        catalogue = self._emotion_presets.get(emotion, self._emotion_presets["neutral"])
        if not catalogue:
            return []

        if seed:
            offset = abs(hash(seed)) % len(catalogue)
            rotated = catalogue[offset:] + catalogue[:offset]
        else:
            rotated = catalogue

        return rotated[:3]

    def _generate_summary(self, emotion: str, description: str, mood_text: Optional[str]) -> str:
        fragments = [description.strip()] if description else []
        if mood_text:
            fragments.append(f"Listener mood: {mood_text.strip()[:80]}")

        extras = {
            "happy": "Expect upbeat pop bops to keep energy high.",
            "sad": "Gentle, emotive tracks to hold space for the feels.",
            "angry": "High-octane anthems to channel every ounce of fire.",
            "surprise": "Adventurous sounds packed with bright twists.",
            "fear": "Resilient, cinematic climbs to steady the nerves.",
            "disgust": "Bold, confident cuts to reset the vibe.",
            "neutral": "Balanced rhythms perfect for any moment.",
        }
        fragments.append(extras.get(emotion, "Curated staples to match your aura."))
        return " | ".join(fragments)

    def _infer_emotion_from_text(self, mood_text: str) -> str:
        lowered = mood_text.lower()
        lexicon = {
            "happy": ["happy", "joy", "excited", "sunny", "ecstatic", "grateful", "smile"],
            "sad": ["sad", "down", "blue", "melancholy", "depressed", "lonely", "cry"],
            "angry": ["angry", "mad", "furious", "rage", "irritated", "frustrated"],
            "surprise": ["surprised", "wow", "shocked", "unexpected", "thrilled"],
            "fear": ["anxious", "nervous", "scared", "fear", "worried", "tense"],
            "disgust": ["disgust", "gross", "meh", "ew", "annoyed", "bored"],
            "neutral": ["calm", "okay", "fine", "neutral", " chill", "focused"],
        }

        for emotion, keywords in lexicon.items():
            if any(keyword in lowered for keyword in keywords):
                return emotion

        return "neutral"

    def _enrich_with_links(self, song: Dict[str, str]) -> Dict[str, str]:
        title = song.get("title", "").strip()
        artist = song.get("artist", "").strip()
        query = f"{title} {artist}".strip()

        youtube_link = self._yt_api_first_video(query)
        if youtube_link is None:
            youtube_link = self._yt_search_url(title, artist)

        return {
            **song,
            "youtube_link": youtube_link,
            "spotify_search": f"https://open.spotify.com/search/{urllib.parse.quote_plus(query)}",
        }

    def _dedupe_songs(self, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for song in songs:
            title = (song.get("title") or "").strip()
            artist = (song.get("artist") or "").strip()
            if not title or not artist:
                continue
            key = (title.lower(), artist.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(song)
            if len(deduped) == 3:
                break
        return deduped

    # ------------------------------------------------------------------
    # YouTube utilities
    # ------------------------------------------------------------------
    def _yt_search_url(self, title: str, artist: str) -> str:
        query = urllib.parse.quote_plus(f"{title} {artist}")
        return f"https://www.youtube.com/results?search_query={query}"

    def _yt_api_first_video(self, query: str) -> Optional[str]:
        if not self._youtube_key:
            return None

        try:
            response = self._session.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "videoCategoryId": "10",
                    "maxResults": 1,
                    "key": self._youtube_key,
                },
                timeout=8,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items") or []
            video_id = (
                items[0]
                .get("id", {})
                .get("videoId")
                if items and isinstance(items[0], dict)
                else None
            )
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"
        except Exception:
            return None

        return None

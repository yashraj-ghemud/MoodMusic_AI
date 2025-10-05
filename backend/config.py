import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (You'll add these)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'your-youtube-api-key-here')
    SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'your-spotify-client-id-here')
    
    # Server config
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000

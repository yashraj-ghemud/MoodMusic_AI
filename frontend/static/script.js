'use strict';

const API_BASE = (() => {
    const { protocol, hostname, port } = window.location;
    if (!port || port === '5000') {
        return '';
    }
    const safeProtocol = protocol.startsWith('http') ? protocol : 'http:';
    const host = hostname || '127.0.0.1';
    return `${safeProtocol}//${host}:5000`;
})();

class MoodMusicApp {
    constructor() {
        this.currentImageData = null;
        this.stream = null;
        this.loadingTimers = [];
        this.cachedResults = null;

        this.initializeElements();
        this.attachEventListeners();
        this.setupDragAndDrop();
    }

    initializeElements() {
        this.uploadSection = document.getElementById('uploadSection');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.previewContainer = document.getElementById('previewContainer');

        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startCameraBtn = document.getElementById('startCamera');
        this.captureBtn = document.getElementById('capturePhoto');

        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.previewImage = document.getElementById('previewImage');

        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.retakeBtn = document.getElementById('retakeBtn');
        this.tryAgainBtn = document.getElementById('tryAgainBtn');
        this.shareBtn = document.getElementById('shareBtn');

        this.emotionIcon = document.getElementById('emotionIcon');
        this.emotionTitle = document.getElementById('emotionTitle');
        this.emotionDescription = document.getElementById('emotionDescription');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceText = document.getElementById('confidenceText');
        this.emotionBreakdown = document.getElementById('emotionBreakdown');

        this.songsGrid = document.getElementById('songsGrid');
        this.curatorSummary = document.getElementById('curatorSummary');

        this.moodInput = document.getElementById('moodInput');
        this.moodSubmit = document.getElementById('moodSubmit');
        this.moodChips = Array.from(document.querySelectorAll('[data-mood-preset]'));
    }

    attachEventListeners() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.captureBtn.addEventListener('click', () => this.capturePhoto());

        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (event) => this.handleFileSelect(event));

        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        this.retakeBtn.addEventListener('click', () => this.resetToUpload());
        this.tryAgainBtn.addEventListener('click', () => this.resetToUpload());
        this.shareBtn.addEventListener('click', () => this.shareResults());

        if (this.moodSubmit) {
            this.moodSubmit.addEventListener('click', () => this.generateFromMoodBox());
        }
        if (this.moodInput) {
            this.moodInput.addEventListener('keydown', (event) => {
                if (event.ctrlKey && event.key === 'Enter') {
                    event.preventDefault();
                    this.generateFromMoodBox();
                }
            });
        }
        this.moodChips.forEach((chip) => {
            chip.addEventListener('click', () => {
                const preset = chip.getAttribute('data-mood-preset');
                if (preset && this.moodInput) {
                    this.moodInput.value = preset;
                    this.moodInput.focus();
                }
            });
        });

        window.addEventListener('beforeunload', () => this.stopCamera());
    }

    setupDragAndDrop() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
            this.uploadArea.addEventListener(eventName, this.preventDefaults);
        });

        ['dragenter', 'dragover'].forEach((eventName) => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach((eventName) => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.remove('dragover');
            });
        });

        this.uploadArea.addEventListener('drop', (event) => this.handleDrop(event));
    }

    preventDefaults(event) {
        event.preventDefault();
        event.stopPropagation();
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 720 },
                    height: { ideal: 540 },
                    facingMode: 'user',
                },
            });

            this.video.srcObject = this.stream;
            this.video.classList.add('active');
            this.startCameraBtn.disabled = true;
            this.captureBtn.disabled = false;
            this.showNotification('Camera live! Center your face and hit capture.', 'success');
        } catch (error) {
            console.error('Camera error', error);
            this.showNotification('Unable to access camera. Try uploading a photo instead.', 'error');
        }
    }

    capturePhoto() {
        if (!this.video.videoWidth) {
            this.showNotification('Camera still warming up. Give it a second!', 'info');
            return;
        }

        const context = this.canvas.getContext('2d');
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        context.drawImage(this.video, 0, 0);

        this.currentImageData = this.canvas.toDataURL('image/jpeg', 0.9);
        this.previewImage.src = this.currentImageData;
        this.showPreview();
        this.stopCamera();
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach((track) => track.stop());
            this.stream = null;
        }
        this.video.classList.remove('active');
        this.startCameraBtn.disabled = false;
        this.captureBtn.disabled = true;
    }

    handleFileSelect(event) {
        const file = event.target.files?.[0];
        if (file) {
            this.processFile(file);
        }
    }

    handleDrop(event) {
        const file = event.dataTransfer?.files?.[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showNotification('Please choose an image file (jpg, png, heic).', 'error');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            this.showNotification('Image is too large. Keep it under 5MB.', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            this.currentImageData = event.target?.result;
            this.previewImage.src = this.currentImageData;
            this.showPreview();
        };
        reader.readAsDataURL(file);
    }

    showPreview() {
        this.previewContainer.hidden = false;
        this.previewContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        this.analyzeBtn.focus();
    }

    async analyzeImage() {
        if (!this.currentImageData) {
            this.showNotification('Choose or capture a photo first.', 'error');
            return;
        }

        this.showSection('loading');
        this.startLoadingAnimation();
        this.analyzeBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/api/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: this.currentImageData }),
            });

            const payload = await response.json();
            if (!response.ok || payload.error) {
                throw new Error(payload.error || `HTTP ${response.status}`);
            }

            this.cachedResults = payload;
            this.displayResults(payload);
            this.showSection('results');
        } catch (error) {
            console.error('Analyze error', error);
            this.showNotification(error.message || 'Mood analysis failed. Please try again.', 'error');
            this.resetLoadingAnimation();
            this.showSection('upload');
            this.updateCuratorSummary('', '');
        } finally {
            this.analyzeBtn.disabled = false;
        }
    }

    async generateFromMoodBox() {
        if (!this.moodInput) {
            return;
        }

        const moodText = this.moodInput.value.trim();
        if (!moodText) {
            this.showNotification('Type a few words about your mood first.', 'error');
            this.moodInput.focus();
            return;
        }

        this.showSection('loading');
        this.startLoadingAnimation();
        this.toggleMoodSubmit(true);

        try {
            const response = await fetch(`${API_BASE}/api/mood`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mood: moodText }),
            });

            const payload = await response.json();
            if (!response.ok || payload.error) {
                throw new Error(payload.error || `HTTP ${response.status}`);
            }

            const normalized = {
                emotion: payload.emotion || 'neutral',
                description: payload.description || moodText,
                confidence: payload.confidence ?? 0,
                songs: payload.songs || [],
                all_emotions: payload.all_emotions || {},
                curator_summary: payload.curator_summary || '',
            };

            this.cachedResults = normalized;
            this.displayResults(normalized);
            this.showSection('results');
        } catch (error) {
            console.error('Mood box error', error);
            this.showNotification(error.message || 'Could not build a playlist. Try again.', 'error');
            this.resetLoadingAnimation();
            this.showSection('upload');
            this.updateCuratorSummary('', '');
        } finally {
            this.toggleMoodSubmit(false);
        }
    }

    toggleMoodSubmit(disabled) {
        if (!this.moodSubmit) {
            return;
        }
        this.moodSubmit.disabled = disabled;
        this.moodSubmit.classList.toggle('loading', disabled);
        if (disabled) {
            this.moodSubmit.dataset.originalText = this.moodSubmit.dataset.originalText || this.moodSubmit.innerHTML;
            this.moodSubmit.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Curating...';
        } else if (this.moodSubmit.dataset.originalText) {
            this.moodSubmit.innerHTML = this.moodSubmit.dataset.originalText;
        }
    }

    startLoadingAnimation() {
        this.resetLoadingAnimation();
        ['step1', 'step2', 'step3'].forEach((id, index) => {
            const timer = setTimeout(() => {
                const el = document.getElementById(id);
                if (el) {
                    el.classList.add('active');
                }
            }, (index + 1) * 900);
            this.loadingTimers.push(timer);
        });
    }

    resetLoadingAnimation() {
        this.loadingTimers.forEach((timer) => clearTimeout(timer));
        this.loadingTimers = [];
        ['step1', 'step2', 'step3'].forEach((id) => {
            const el = document.getElementById(id);
            if (el) {
                el.classList.remove('active');
            }
        });
    }

    displayResults(result) {
        this.updateEmotionDisplay(result.emotion, result.description, result.confidence);
        this.renderEmotionBreakdown(result.all_emotions || {});
        this.renderSongs(result.songs || []);
        this.updateCuratorSummary(result.curator_summary, result.description);
    }

    updateEmotionDisplay(emotion, description, confidence = 0) {
        const emotionIcons = {
            happy: 'fa-face-grin-stars',
            sad: 'fa-face-sad-tear',
            angry: 'fa-face-angry',
            surprise: 'fa-face-surprise',
            fear: 'fa-face-frown-open',
            disgust: 'fa-face-dizzy',
            neutral: 'fa-face-meh',
        };

        const icon = emotionIcons[emotion] || 'fa-face-smile';
        this.emotionIcon.innerHTML = `<i class="fas ${icon}"></i>`;

        const title = emotion ? emotion.charAt(0).toUpperCase() + emotion.slice(1) : 'Unknown';
        this.emotionTitle.textContent = `Your Mood: ${title}`;
        this.emotionDescription.textContent = description || 'Curated tunes coming right up!';

        const clampedConfidence = Math.max(0, Math.min(100, Math.round(confidence)));
        this.confidenceFill.style.width = `${clampedConfidence}%`;
        this.confidenceText.textContent = `${clampedConfidence}% confident`;
    }

    renderEmotionBreakdown(probabilities) {
        if (!this.emotionBreakdown) {
            return;
        }

        this.emotionBreakdown.innerHTML = '';
        const entries = Object.entries(probabilities);
        if (!entries.length) {
            this.emotionBreakdown.hidden = true;
            return;
        }

        this.emotionBreakdown.hidden = false;
        entries
            .sort(([, a], [, b]) => b - a)
            .forEach(([label, value]) => {
                const chip = document.createElement('span');
                chip.className = 'chip';
                chip.textContent = `${label}: ${value.toFixed(1)}%`;
                this.emotionBreakdown.appendChild(chip);
            });
    }

    updateCuratorSummary(summary, fallbackDescription) {
        if (!this.curatorSummary) {
            return;
        }

        const text = summary?.trim() || fallbackDescription?.trim() || '';
        if (!text) {
            this.curatorSummary.hidden = true;
            this.curatorSummary.textContent = '';
            return;
        }

        this.curatorSummary.hidden = false;
        this.curatorSummary.textContent = text;
    }

    renderSongs(songs) {
        this.songsGrid.innerHTML = '';

        if (!songs.length) {
            const empty = document.createElement('p');
            empty.className = 'song-reason';
            empty.textContent = 'No tracks yet – try another capture or check your connection.';
            this.songsGrid.appendChild(empty);
            return;
        }

        songs.forEach((song, index) => {
            const card = this.createSongCard(song, index);
            this.songsGrid.appendChild(card);
        });
    }

    createSongCard(song, index) {
        const card = document.createElement('div');
        card.className = 'song-card';
        card.style.animationDelay = `${index * 120}ms`;

        const youtube = song.youtube_link || '#';
        const spotify = song.spotify_search || '#';

        card.innerHTML = `
      <div class="song-header">
        <div class="song-icon"><i class="fas fa-music"></i></div>
        <div class="song-info">
          <h4>${song.title || 'Untitled track'}</h4>
          <div class="artist">${song.artist || 'Unknown artist'}</div>
        </div>
      </div>
      <div class="song-reason">${song.reason || 'Handpicked for your vibe.'}</div>
      <div class="song-links">
        <a class="btn youtube-btn" href="${youtube}" target="_blank" rel="noopener">
          <i class="fab fa-youtube"></i> YouTube
        </a>
        <a class="btn spotify-btn" href="${spotify}" target="_blank" rel="noopener">
          <i class="fab fa-spotify"></i> Spotify
        </a>
      </div>
    `;

        return card;
    }

    showSection(section) {
        this.uploadSection.hidden = section !== 'upload';
        this.loadingSection.hidden = section !== 'loading';
        this.resultsSection.hidden = section !== 'results';
    }

    resetToUpload() {
        this.stopCamera();
        this.currentImageData = null;
        this.fileInput.value = '';
        this.previewContainer.hidden = true;
        this.cachedResults = null;
        this.resetLoadingAnimation();
        this.showSection('upload');
        this.updateCuratorSummary('', '');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    shareResults() {
        if (!this.cachedResults) {
            this.showNotification('Analyze a photo first.', 'info');
            return;
        }

        const { emotion, description, songs = [] } = this.cachedResults;
        const summary = songs
            .slice(0, 3)
            .map((track, idx) => `${idx + 1}. ${track.title} – ${track.artist}`)
            .join('\n');

        const shareText = `MoodMusic decoded my vibe as ${emotion}. ${description}\nTop picks:\n${summary}`;

        if (navigator.share) {
            navigator.share({
                title: 'My MoodMusic AI playlist',
                text: shareText,
                url: window.location.href,
            }).catch(() => {
                this.showNotification('Sharing cancelled.', 'info');
            });
        } else {
            navigator.clipboard
                .writeText(`${shareText}\n${window.location.href}`)
                .then(() => this.showNotification('Playlist copied to clipboard!', 'success'))
                .catch(() => this.showNotification('Could not copy link.', 'error'));
        }
    }

    showNotification(message, type = 'info') {
        const palette = {
            success: '#46d68c',
            error: '#ff6d6d',
            info: '#6a5bff',
        };

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
      position: fixed;
      top: 24px;
      right: 24px;
      z-index: 9999;
      padding: 14px 22px;
      border-radius: 18px;
      background: ${palette[type] || palette.info};
      color: #fff;
      box-shadow: 0 18px 40px rgba(0,0,0,0.25);
      font-weight: 600;
      letter-spacing: 0.01em;
      animation: fade-in-down 220ms ease;
    `;
        notification.textContent = message;

        document.body.appendChild(notification);
        setTimeout(() => {
            notification.style.animation = 'fade-out-up 250ms ease forwards';
            setTimeout(() => notification.remove(), 260);
        }, 4200);
    }
}

const extraKeyframes = document.createElement('style');
extraKeyframes.textContent = `
  @keyframes fade-out-up {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(-12px); }
  }
`;
document.head.appendChild(extraKeyframes);

document.addEventListener('DOMContentLoaded', () => {
    window.moodMusicApp = new MoodMusicApp();
});


// script.js (top)
const API_BASE = (() => {
    if (window.location.port === '5000') {
        return ''; // same-origin when served directly by Flask
    }

    const protocol = window.location.protocol.startsWith('http')
        ? window.location.protocol
        : 'http:';
    const host = window.location.hostname || '127.0.0.1';

    return `${protocol}//${host}:5000`; // fallback to local Flask server during standalone frontend dev
})();

class MoodMusicApp {

    constructor() {
        this.currentImageData = null;
        this.stream = null;
        this.initializeElements();
        this.attachEventListeners();
        this.setupDragAndDrop();
    }

    initializeElements() {
        // Sections
        this.uploadSection = document.getElementById('uploadSection');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.previewContainer = document.getElementById('previewContainer');

        // Camera elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startCameraBtn = document.getElementById('startCamera');
        this.captureBtn = document.getElementById('capturePhoto');

        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.previewImage = document.getElementById('previewImage');

        // Action buttons
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.retakeBtn = document.getElementById('retakeBtn');
        this.tryAgainBtn = document.getElementById('tryAgainBtn');
        this.shareBtn = document.getElementById('shareBtn');

        // Results elements
        this.emotionIcon = document.getElementById('emotionIcon');
        this.emotionTitle = document.getElementById('emotionTitle');
        this.emotionDescription = document.getElementById('emotionDescription');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceText = document.getElementById('confidenceText');
        this.songsGrid = document.getElementById('songsGrid');
    }

    attachEventListeners() {
        // Camera controls
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.captureBtn.addEventListener('click', () => this.capturePhoto());

        // File upload
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.uploadArea.addEventListener('click', () => this.fileInput.click());

        // Action buttons
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        this.retakeBtn.addEventListener('click', () => this.resetToUpload());
        this.tryAgainBtn.addEventListener('click', () => this.resetToUpload());
        this.shareBtn.addEventListener('click', () => this.shareResults());
    }

    setupDragAndDrop() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, this.preventDefaults);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.remove('dragover');
            });
        });

        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async startCamera() {
        this.stopCamera();
        this.startCameraBtn.disabled = true;
        this.captureBtn.disabled = true;

        const primaryConstraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: { ideal: 'user' }
            },
            audio: false
        };

        const fallbackConstraints = {
            video: true,
            audio: false
        };

        this.video.setAttribute('playsinline', 'true');
        this.video.muted = true;

        try {
            this.stream = await this.requestCameraStream(primaryConstraints);
        } catch (primaryError) {
            console.warn('Primary camera constraints failed:', primaryError);

            if (primaryError && (primaryError.name === 'AbortError' || primaryError.name === 'NotReadableError')) {
                await this.delay(400);
            }

            try {
                this.stream = await this.requestCameraStream(fallbackConstraints);
            } catch (fallbackError) {
                console.error('Error accessing camera:', fallbackError);
                const message = fallbackError && fallbackError.name === 'AbortError'
                    ? 'Camera timed out. Ensure no other app is using it, then try again or upload a photo.'
                    : 'Unable to access camera. Please try uploading a photo instead.';
                this.showNotification(message, 'error');
                this.startCameraBtn.disabled = false;
                return;
            }
        }

        this.video.srcObject = this.stream;
        this.video.classList.add('active');
        this.captureBtn.disabled = false;

        this.showNotification('Camera started! Position yourself and click capture.', 'success');
    }

    async requestCameraStream(constraints) {
        return navigator.mediaDevices.getUserMedia(constraints);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    capturePhoto() {
        const context = this.canvas.getContext('2d');
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;

        context.drawImage(this.video, 0, 0);

        // Convert to base64
        this.currentImageData = this.canvas.toDataURL('image/jpeg', 0.8);

        // Show preview
        this.previewImage.src = this.currentImageData;
        this.showPreview();

        // Stop camera
        this.stopCamera();
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }

        if (this.video) {
            this.video.srcObject = null;
            this.video.classList.remove('active');
        }

        this.stream = null;
        this.startCameraBtn.disabled = false;
        this.captureBtn.disabled = true;
    }

    handleFileSelect(event) {
        const [file] = (event.target.files || []);
        if (file) {
            this.processFile(file);
        }
    }

    handleDrop(event) {
        const files = Array.from(event.dataTransfer.files || []);
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showNotification('Please select a valid image file.', 'error');
            return;
        }

        if (file.size > 5 * 1024 * 1024) { // 5MB limit
            this.showNotification('Image size should be less than 5MB.', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImageData = e.target.result;
            this.previewImage.src = this.currentImageData;
            this.showPreview();
        };
        reader.readAsDataURL(file);
    }

    showPreview() {
        this.previewContainer.style.display = 'block';
        this.previewContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    async analyzeImage() {
        if (!this.currentImageData) {
            this.showNotification('Please select an image first.', 'error');
            return;
        }

        // Show loading
        this.showSection('loading');
        this.simulateLoadingSteps();

        try {
            const response = await fetch(`${API_BASE}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: this.currentImageData
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification('Failed to analyze image. Please try again.', 'error');
            this.showSection('upload');
        }
    }

    simulateLoadingSteps() {
        const steps = ['step1', 'step2', 'step3'];
        steps.forEach((stepId, index) => {
            setTimeout(() => {
                document.getElementById(stepId).classList.add('active');
            }, (index + 1) * 1000);
        });
    }

    displayResults(result) {
        // Update emotion display
        this.updateEmotionDisplay(result.emotion, result.description, result.confidence);

        // Display songs
        this.displaySongs(result.songs);

        // Show results section
        this.showSection('results');
    }

    updateEmotionDisplay(emotion, description, confidence) {
        // Update emotion icon
        const emotionIcons = {
            'happy': 'fa-smile',
            'sad': 'fa-frown',
            'angry': 'fa-angry',
            'surprise': 'fa-surprise',
            'fear': 'fa-meh',
            'disgust': 'fa-meh-rolling-eyes',
            'neutral': 'fa-meh'
        };

        const iconClass = emotionIcons[emotion] || 'fa-smile';
        this.emotionIcon.innerHTML = `<i class="fas ${iconClass}"></i>`;

        // Update text
        this.emotionTitle.textContent = `Your Mood: ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`;
        this.emotionDescription.textContent = description;

        // Update confidence bar
        const confidencePercent = Math.round(confidence);
        this.confidenceFill.style.width = `${confidencePercent}%`;
        this.confidenceText.textContent = `${confidencePercent}% confident`;
    }

    displaySongs(songs) {
        this.songsGrid.innerHTML = '';

        songs.forEach((song, index) => {
            const songCard = this.createSongCard(song, index);
            this.songsGrid.appendChild(songCard);
        });
    }

    createSongCard(song, index) {
        const card = document.createElement('div');
        card.className = 'song-card';
        card.style.animationDelay = `${index * 0.2}s`;

        card.innerHTML = `
            <div class="song-header">
                <div class="song-icon">
                    <i class="fas fa-music"></i>
                </div>
                <div class="song-info">
                    <h4>${song.title}</h4>
                    <div class="artist">by ${song.artist}</div>
                </div>
            </div>
            <div class="song-reason">"${song.reason}"</div>
            <div class="song-links">
                <a href="${song.youtube_link}" target="_blank" class="btn youtube-btn">
                    <i class="fab fa-youtube"></i> YouTube
                </a>
                <a href="${song.spotify_search}" target="_blank" class="btn spotify-btn">
                    <i class="fab fa-spotify"></i> Spotify
                </a>
            </div>
        `;

        return card;
    }

    showSection(sectionName) {
        // Hide all sections
        [this.uploadSection, this.loadingSection, this.resultsSection].forEach(section => {
            section.style.display = 'none';
        });

        // Show requested section
        switch (sectionName) {
            case 'upload':
                this.uploadSection.style.display = 'block';
                break;
            case 'loading':
                this.loadingSection.style.display = 'block';
                break;
            case 'results':
                this.resultsSection.style.display = 'block';
                break;
        }
    }

    resetToUpload() {
        this.stopCamera();
        this.currentImageData = null;
        this.previewContainer.style.display = 'none';
        this.fileInput.value = '';
        this.showSection('upload');

        // Reset loading steps
        ['step1', 'step2', 'step3'].forEach(stepId => {
            document.getElementById(stepId).classList.remove('active');
        });
    }

    shareResults() {
        if (navigator.share) {
            navigator.share({
                title: 'My Mood & Music Results',
                text: 'Check out what MoodMusic AI discovered about my emotions!',
                url: window.location.href
            });
        } else {
            // Fallback - copy to clipboard
            const text = `Check out my mood analysis on MoodMusic AI: ${window.location.href}`;
            navigator.clipboard.writeText(text).then(() => {
                this.showNotification('Results link copied to clipboard!', 'success');
            });
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#ff6b6b' : '#51cf66'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
            max-width: 300px;
        `;
        notification.textContent = message;

        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
}

// Add notification animations
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(notificationStyles);

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.moodMusicApp = new MoodMusicApp();
});

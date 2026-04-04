"""
config.py — Project-wide constants and configuration
"""
import os

# ── YouTube API ──────────────────────────────────────────────────────────────
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"
YOUTUBE_API_KEY      = os.environ.get("YOUTUBE_API_KEY", "")

# ── Supported Regions ────────────────────────────────────────────────────────
SUPPORTED_REGIONS = [
    "IN",  # India
    "US",  # United States
    "GB",  # United Kingdom
    "JP",  # Japan
    "BR",  # Brazil
    "KR",  # South Korea
    "DE",  # Germany
    "FR",  # France
    "MX",  # Mexico
    "AU",  # Australia
]

REGION_NAMES = {
    "IN": "India", "US": "United States", "GB": "United Kingdom",
    "JP": "Japan", "BR": "Brazil", "KR": "South Korea",
    "DE": "Germany", "FR": "France", "MX": "Mexico", "AU": "Australia",
}

# ── Image Analysis ───────────────────────────────────────────────────────────
THUMBNAIL_WIDTH  = 1280
THUMBNAIL_HEIGHT = 720
FACE_CASCADE_PATH = cv_default = None  # loaded dynamically in image_analyzer

# Haarcascade paths (bundled with OpenCV)
HAAR_FACE_PATH     = "haarcascade_frontalface_default.xml"
HAAR_PROFILE_PATH  = "haarcascade_profileface.xml"

# ── Feature Extraction ───────────────────────────────────────────────────────
COLOR_BINS     = 32    # histogram bins per channel
EDGE_THRESHOLD = (100, 200)  # Canny thresholds

# ── ML Model ────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/trend_predictor.pkl"
SCALER_PATH     = "models/scaler.pkl"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

# Feature list used by the ML model (must match ImageAnalyzer.extract_all_features)
FEATURE_COLUMNS = [
    "brightness", "contrast", "saturation_mean", "saturation_std",
    "hue_mean", "edge_density", "face_count", "has_face",
    "red_ratio", "green_ratio", "blue_ratio",
    "sharpness", "color_diversity", "text_region_density",
    "title_length", "title_has_numbers", "title_has_caps_words",
    "duration_minutes", "subscribers_log",
]

# ── Misleading Detection ─────────────────────────────────────────────────────
MISLEADING_THRESHOLD         = 0.65
MISMATCH_THRESHOLD           = 0.40
SENTIMENT_DISCREPANCY_THRESH = 0.30

# Keywords that are common clickbait / misleading signals
CLICKBAIT_KEYWORDS = [
    "you won't believe", "shocking", "exposed", "gone wrong",
    "gone sexual", "*must watch*", "not clickbait", "100%",
    "insane", "leaked", "secret", "hidden", "truth about",
]

# ── Visualization ────────────────────────────────────────────────────────────
PLOT_STYLE  = "dark_background"
ACCENT_COLOR = "#ff4444"
PALETTE     = ["#ff4444", "#4488ff", "#44ff88", "#ffaa44", "#aa44ff"]

# ── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR        = "data"
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")
COLLECTED_DIR   = os.path.join(DATA_DIR, "collected")
MODELS_DIR      = "models"

os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(COLLECTED_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR,      exist_ok=True)

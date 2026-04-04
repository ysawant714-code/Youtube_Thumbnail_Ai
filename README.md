# 🎬 Region-Based Analysis of YouTube Trending Videos using Thumbnails

> **CS Mini-Project** | Python · OpenCV · Streamlit · scikit-learn · YouTube Data API v3

A research-backed web application that extracts visual features from YouTube thumbnails, compares design patterns across global regions, predicts trending potential using machine learning, and detects misleading content through multi-modal analysis.

---

## 📁 Project Architecture

```
youtube_thumbnail_analyzer/
│
├── app.py                        ← Streamlit entry point (all UI pages)
├── config.py                     ← Global constants (regions, thresholds, paths)
├── requirements.txt
├── .env                          ← API key (gitignored)
│
├── modules/
│   ├── __init__.py
│   ├── youtube_collector.py      ← YouTube Data API v3 wrapper
│   ├── image_analyzer.py         ← OpenCV feature extraction
│   ├── ml_model.py               ← scikit-learn ML pipeline
│   ├── visualizer.py             ← Matplotlib/Seaborn charts
│   ├── misleading_detector.py    ← Multi-modal mismatch detection
│   └── region_analyzer.py        ← Regional statistical comparison
│
├── data/
│   ├── sample/                   ← Auto-generated synthetic datasets
│   └── collected/                ← Your collected YouTube data (JSON)
│
└── models/
    └── trend_predictor.pkl       ← Saved ML model (auto-trained on first run)
```

---

## 🏗️ System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Thumbnail │ │  Region  │ │  Trend   │ │Misleading│  │
│  │ Analyzer │ │ Compare  │ │Predictor │ │Detector  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
└───────┼─────────────┼────────────┼─────────────┼────────┘
        │             │            │             │
   ┌────▼─────┐  ┌────▼──────┐ ┌──▼──────┐ ┌───▼───────┐
   │  Image   │  │  Region   │ │   ML    │ │Misleading │
   │ Analyzer │  │ Analyzer  │ │  Model  │ │ Detector  │
   │ (OpenCV) │  │(Stats+Viz)│ │(sklearn)│ │(NLP+CV)   │
   └────┬─────┘  └─────┬─────┘ └──┬──────┘ └───┬───────┘
        │              │           │             │
   ┌────▼──────────────▼───────────▼─────────────▼───────┐
   │              youtube_collector.py                    │
   │           (YouTube Data API v3)                      │
   └──────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download the project
```bash
git clone <your-repo>
cd youtube_thumbnail_analyzer
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up YouTube API Key (optional — sample data works without it)

Create a `.env` file:
```
YOUTUBE_API_KEY=your_key_here
```

Or set the environment variable:
```bash
export YOUTUBE_API_KEY="your_key_here"
```

**Get a key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **YouTube Data API v3**
3. Create an **API Key** credential

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🔬 Feature Extraction Pipeline

| Feature | Method | Insight |
|---------|--------|---------|
| **Brightness** | HSV V-channel mean | Low brightness → lower CTR |
| **Contrast** | Grayscale std dev | Higher contrast → more attention |
| **Saturation** | HSV S-channel stats | High sat → emotional engagement |
| **Hue Distribution** | HSV H-channel | Cultural color preferences |
| **Edge Density** | Canny edge detector | Visual complexity |
| **Face Count** | Haar cascade | Faces → +30–40% CTR boost |
| **Sharpness** | Laplacian variance | Blur = low quality signal |
| **Color Diversity** | K-means clustering | Palette complexity |
| **Text Region Density** | MSER detector | Text-heavy overlays |

---

## 🤖 ML Model Details

**Algorithm:** Random Forest Classifier (200 trees, max_depth=8)

**Feature vector (19 features):**
- 14 image features from OpenCV
- 3 metadata features (title length, has numbers, has caps)
- 2 channel features (subscribers_log, duration_minutes)

**Label:** `trending=1` if views ≥ median views of collected dataset

**Training:** Auto-trains on synthetic data with realistic regional priors.
Retrain on real data: `predictor.train_on_real_data(videos, features)`

---

## 🚨 Misleading Detection Approach

Based on **multi-modal mismatch** detection (inspired by recent research):

| Check | Method | Weight |
|-------|--------|--------|
| Clickbait Title | Phrase + keyword matching | 30% |
| Sentiment Mismatch | Negative title + bright image | 25% |
| Visual Anomaly | Extreme HSV values | 20% |
| Extreme Editing | Laplacian + unsharp residual | 15% |
| Face Extremity | Reaction-face size analysis | 10% |

---

## 📊 Dataset Sources

### Option A: Live Collection (YouTube API)
- Use the built-in **Data Collector** page
- Requires YouTube Data API v3 key
- Quota: 10,000 units/day (free tier)
- ~100 units per 50 videos

### Option B: Kaggle Datasets (no API key needed)
| Dataset | Link | Contents |
|---------|------|----------|
| YouTube Trending Videos | [kaggle.com/datasnaek/youtube-new](https://www.kaggle.com/datasnaek/youtube-new) | US/GB/IN/CA/DE/FR/JP/KR/MX/RU trending, 2017–2018 |
| YT Trending 2020–2021 | [kaggle.com/rsrishav/youtube-trending-video-dataset](https://www.kaggle.com/rsrishav/youtube-trending-video-dataset) | 10 regions, daily snapshots |
| Thumbnail Images | [kaggle.com/praneshmukhopadhyay/youtube-thumbnail-dataset](https://www.kaggle.com/praneshmukhopadhyay/youtube-thumbnail-dataset) | Pre-downloaded thumbnails |

### Option C: Use Synthetic Sample Data
The app auto-generates realistic synthetic data using regional priors from research literature. No API key or download needed — just check **"Use sample dataset"** in the Region Comparison page.

---

## 📚 Research Papers & References

1. **"You Shouldn't Trust Me": Learning the Danger of Clickbait** — Zhou et al., 2017
2. **Predicting the Popularity of Online News Using Multimodal Features** — Bandari et al.
3. **FakeThumbnail: Fake Thumbnail Detection on YouTube** — Lee et al., 2022
4. **Regional Differences in YouTube Thumbnail Design** — Analyzing cultural clickbait patterns
5. **Visual Sentiment Analysis Using Deep Convolutional Networks** (for emotion proxy logic)

---

## 🗺️ Supported Regions

| Code | Country | Code | Country |
|------|---------|------|---------|
| IN | India | JP | Japan |
| US | United States | KR | South Korea |
| GB | United Kingdom | DE | Germany |
| BR | Brazil | FR | France |
| MX | Mexico | AU | Australia |

---

## 📈 Extending the Project

- **Upgrade ML model**: Replace RandomForest with a CNN (ResNet/EfficientNet) for direct pixel-level feature learning
- **NLP improvement**: Use sentence-transformers for semantic title-image matching  
- **Real emotion detection**: Integrate DeepFace or FER+ for actual emotion classification
- **Time-series analysis**: Track how trending thumbnails evolve over weeks
- **Dashboard export**: Add PDF report generation via ReportLab

---

## 📝 Submission Checklist

- [x] YouTube API integration
- [x] OpenCV feature extraction (brightness, contrast, edge, face)
- [x] Region-based comparative analysis (10 regions)
- [x] ML trend prediction (Random Forest + feature importance)
- [x] Misleading thumbnail detection (multi-modal)
- [x] Matplotlib visualizations (histograms, radar, heatmap, bar charts)
- [x] Interactive Streamlit interface
- [x] Sample data (no API key required for demo)
- [x] Research paper insights incorporated
- [x] Modular, documented Python code

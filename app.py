"""
Region-Based Analysis of YouTube Trending Videos using Thumbnails
Main Streamlit Application Entry Point
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.image_analyzer import ImageAnalyzer
from modules.ml_model import TrendPredictor
from modules.visualizer import Visualizer
from modules.misleading_detector import MisleadingDetector
from modules.region_analyzer import RegionAnalyzer
from modules.youtube_collector import YouTubeCollector
import config

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="YT Thumbnail Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #ff0000;
        box-shadow: 0 0 30px rgba(255,0,0,0.15);
    }
    .main-header h1 { font-family: 'Space Mono', monospace; color: #ff4444; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #aaa; margin: 0.5rem 0 0; font-size: 0.9rem; }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: border-color .2s;
    }
    .metric-card:hover { border-color: #ff4444; }
    .metric-card .label { color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: white; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }

    .verdict-safe    { background:#0d2b1f; border:1px solid #00cc66; border-radius:10px; padding:1rem; color:#00cc66; }
    .verdict-warning { background:#2b1a0d; border:1px solid #ff9900; border-radius:10px; padding:1rem; color:#ff9900; }
    .verdict-danger  { background:#2b0d0d; border:1px solid #ff4444; border-radius:10px; padding:1rem; color:#ff4444; }

    .section-header { font-family: 'Space Mono', monospace; color: #ff4444;
                      border-bottom: 1px solid #333; padding-bottom: .5rem; margin: 1.5rem 0 1rem; }

    .stButton > button {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white; border: none; border-radius: 8px;
        font-family: 'Space Mono', monospace; font-weight: 700;
        padding: .6rem 2rem; transition: all .2s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(255,0,0,.4); }

    .stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; color: #888; }
    .stTabs [aria-selected="true"] { color: #ff4444 !important; }

    div[data-testid="stSidebar"] { background: #0f0f0f; border-right: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 YT Thumbnail Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🔍 Thumbnail Analyzer",
        "📊 Region Comparison",
        "🤖 Trend Predictor",
        "🚨 Misleading Detector",
        "📥 Data Collector",
        "ℹ️ About",
    ])
    st.markdown("---")
    st.markdown("**API Configuration**")
    api_key = st.text_input("YouTube API Key", type="password",
                            value=os.environ.get("YOUTUBE_API_KEY", ""),
                            help="Required for live data collection")
    if api_key:
        os.environ["YOUTUBE_API_KEY"] = api_key

    st.markdown("---")
    st.caption("Built with OpenCV · Streamlit · scikit-learn")

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎬 YT Thumbnail Intelligence</h1>
  <p>Region-Based Analysis of YouTube Trending Videos using Computer Vision & ML</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Initialize modules
# ──────────────────────────────────────────────
@st.cache_resource
def load_modules():
    return {
        "analyzer":   ImageAnalyzer(),
        "predictor":  TrendPredictor(),
        "visualizer": Visualizer(),
        "misleading": MisleadingDetector(),
        "region":     RegionAnalyzer(),
    }

modules = load_modules()

# ══════════════════════════════════════════════
# PAGE: Thumbnail Analyzer
# ══════════════════════════════════════════════
if page == "🔍 Thumbnail Analyzer":
    st.markdown('<h2 class="section-header">Thumbnail Feature Extraction</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("Upload Thumbnail", type=["jpg", "jpeg", "png", "webp"])
        title_input = st.text_input("Video Title", placeholder="Enter the video title…")
        description_input = st.text_area("Video Description (optional)", height=80,
                                         placeholder="Paste description for misleading detection…")
        region_sel = st.selectbox("Target Region", config.SUPPORTED_REGIONS)
        analyze_btn = st.button("⚡ Analyze Thumbnail")

    with col2:
        if uploaded and analyze_btn:
            # Load image
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Uploaded Thumbnail", use_container_width=True)

            with st.spinner("Extracting features…"):
                features = modules["analyzer"].extract_all_features(img_bgr)

            # ── Metric Cards ──────────────────────────────────
            st.markdown('<h3 class="section-header">Feature Metrics</h3>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            mc = '<div class="metric-card"><div class="label">{l}</div><div class="value">{v}</div></div>'

            with c1: st.markdown(mc.format(l="Brightness", v=f"{features['brightness']:.1f}"), unsafe_allow_html=True)
            with c2: st.markdown(mc.format(l="Contrast",   v=f"{features['contrast']:.1f}"),   unsafe_allow_html=True)
            with c3: st.markdown(mc.format(l="Edge Density",v=f"{features['edge_density']:.3f}"), unsafe_allow_html=True)
            with c4: st.markdown(mc.format(l="Faces",       v=str(features['face_count'])),      unsafe_allow_html=True)

            st.markdown("")

            # ── Charts ───────────────────────────────────────
            tabs = st.tabs(["🎨 Color Distribution", "📐 Edge Map", "😊 Face Analysis", "📊 All Features"])

            with tabs[0]:
                fig_color = modules["visualizer"].plot_color_histogram(img_bgr)
                st.pyplot(fig_color)

            with tabs[1]:
                edges     = modules["analyzer"].detect_edges(img_bgr)
                fig_edge  = modules["visualizer"].plot_edge_map(img_rgb, edges)
                st.pyplot(fig_edge)

            with tabs[2]:
                face_img, face_data = modules["analyzer"].detect_faces(img_bgr)
                fig_face = modules["visualizer"].plot_face_detection(
                    cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), face_data)
                st.pyplot(fig_face)
                if face_data:
                    st.info(f"**{len(face_data)} face(s)** detected — face presence is a strong clickbait indicator.")
                else:
                    st.info("No faces detected.")

            with tabs[3]:
                fig_radar = modules["visualizer"].plot_feature_radar(features)
                st.pyplot(fig_radar)

            # ── Full feature table ────────────────────────────
            with st.expander("📋 Raw Feature Values"):
                import pandas as pd
                df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
                st.dataframe(df, use_container_width=True)

        elif not uploaded:
            st.info("👈 Upload a thumbnail on the left to start analysis.")

# ══════════════════════════════════════════════
# PAGE: Region Comparison
# ══════════════════════════════════════════════
elif page == "📊 Region Comparison":
    st.markdown('<h2 class="section-header">Region-Based Comparative Analysis</h2>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        region_a = st.selectbox("Region A", config.SUPPORTED_REGIONS, index=0)
    with col_r:
        region_b = st.selectbox("Region B", config.SUPPORTED_REGIONS, index=1)

    use_sample = st.checkbox("Use sample dataset (no API key needed)", value=True)

    if st.button("🔄 Run Comparison"):
        with st.spinner("Analyzing regional patterns…"):
            if use_sample:
                data_a = modules["region"].load_sample_data(region_a)
                data_b = modules["region"].load_sample_data(region_b)
            else:
                collector = YouTubeCollector(os.environ.get("YOUTUBE_API_KEY", ""))
                data_a = collector.fetch_trending(region_a)
                data_b = collector.fetch_trending(region_b)

            comparison = modules["region"].compare_regions(data_a, data_b, region_a, region_b)

        st.markdown('<h3 class="section-header">Comparison Results</h3>', unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["Feature Comparison", "Engagement Metrics", "Heatmap"])

        with t1:
            fig1 = modules["visualizer"].plot_region_feature_comparison(
                comparison, region_a, region_b)
            st.pyplot(fig1)

        with t2:
            fig2 = modules["visualizer"].plot_engagement_comparison(
                data_a, data_b, region_a, region_b)
            st.pyplot(fig2)

        with t3:
            fig3 = modules["visualizer"].plot_feature_heatmap(comparison)
            st.pyplot(fig3)

        with st.expander("📋 Statistical Summary"):
            import pandas as pd
            st.dataframe(pd.DataFrame(comparison).T, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE: Trend Predictor
# ══════════════════════════════════════════════
elif page == "🤖 Trend Predictor":
    st.markdown('<h2 class="section-header">ML-Based Trend Prediction</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Upload & Metadata")
        uploaded_pred = st.file_uploader("Thumbnail", type=["jpg","jpeg","png","webp"], key="pred")
        title_pred    = st.text_input("Video Title", key="pred_title")
        region_pred   = st.selectbox("Target Region", config.SUPPORTED_REGIONS, key="pred_region")
        channel_subs  = st.number_input("Channel Subscribers", min_value=0, value=100000, step=1000)
        video_duration = st.slider("Video Duration (minutes)", 1, 60, 10)
        pred_btn = st.button("🤖 Predict Trend Score")

    with col2:
        if uploaded_pred and pred_btn:
            file_bytes = np.asarray(bytearray(uploaded_pred.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Running prediction model…"):
                features   = modules["analyzer"].extract_all_features(img_bgr)
                metadata   = {
                    "title": title_pred,
                    "region": region_pred,
                    "subscribers": channel_subs,
                    "duration_minutes": video_duration,
                }
                prediction = modules["predictor"].predict(features, metadata)

            # Score gauge
            score = prediction["trend_score"]
            color = "#00cc66" if score > 0.65 else "#ff9900" if score > 0.4 else "#ff4444"
            verdict_class = "verdict-safe" if score > 0.65 else "verdict-warning" if score > 0.4 else "verdict-danger"
            verdict_text  = "🟢 High Trending Potential" if score > 0.65 else \
                            "🟡 Moderate Potential" if score > 0.4 else "🔴 Low Trending Potential"

            st.markdown(f"""
            <div class="{verdict_class}">
              <h2 style="margin:0">{verdict_text}</h2>
              <p style="margin:.5rem 0 0; font-size:2rem; font-weight:700">{score:.0%}</p>
              <p style="margin:.5rem 0 0; font-size:.85rem">Confidence: {prediction['confidence']:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            fig_feat_imp = modules["visualizer"].plot_feature_importance(prediction["feature_importance"])
            st.pyplot(fig_feat_imp)

            with st.expander("🔍 Improvement Suggestions"):
                for tip in prediction.get("suggestions", []):
                    st.markdown(f"• {tip}")
        else:
            st.info("Fill in the form and click Predict.")

# ══════════════════════════════════════════════
# PAGE: Misleading Detector
# ══════════════════════════════════════════════
elif page == "🚨 Misleading Detector":
    st.markdown('<h2 class="section-header">Misleading Thumbnail Detection</h2>', unsafe_allow_html=True)
    st.caption("Detects mismatches between thumbnail content, title, and description using multi-modal analysis.")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_m  = st.file_uploader("Thumbnail", type=["jpg","jpeg","png","webp"], key="miss")
        title_m     = st.text_input("Video Title",       key="miss_title", placeholder="Actual video title…")
        desc_m      = st.text_area("Video Description", key="miss_desc",  placeholder="Video description…", height=100)
        tags_m      = st.text_input("Tags (comma-separated)", key="miss_tags")
        detect_btn  = st.button("🚨 Detect Misleading Content")

    with col2:
        if uploaded_m and detect_btn:
            file_bytes = np.asarray(bytearray(uploaded_m.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Input Thumbnail", use_container_width=True)

            with st.spinner("Running multi-modal analysis…"):
                result = modules["misleading"].detect(
                    img_bgr, title_m, desc_m, tags_m.split(",") if tags_m else [])

            score = result["misleading_score"]
            verdict_class = "verdict-danger" if score > 0.65 else \
                            "verdict-warning" if score > 0.35 else "verdict-safe"
            verdict_icon  = "🔴" if score > 0.65 else "🟡" if score > 0.35 else "🟢"
            verdict_label = "Likely Misleading" if score > 0.65 else \
                            "Possibly Misleading" if score > 0.35 else "Appears Authentic"

            st.markdown(f"""
            <div class="{verdict_class}">
              <h3 style="margin:0">{verdict_icon} {verdict_label}</h3>
              <p style="margin:.3rem 0 0">Misleading Score: <strong>{score:.0%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            # Breakdown
            for flag, detail in result["flags"].items():
                icon = "⚠️" if detail["triggered"] else "✅"
                st.markdown(f"{icon} **{flag.replace('_',' ').title()}** — {detail['reason']}")

            fig_miss = modules["visualizer"].plot_misleading_breakdown(result["scores"])
            st.pyplot(fig_miss)
        else:
            st.info("Upload a thumbnail and fill in metadata to detect misleading content.")

# ══════════════════════════════════════════════
# PAGE: Data Collector
# ══════════════════════════════════════════════
elif page == "📥 Data Collector":
    st.markdown('<h2 class="section-header">YouTube Trending Data Collection</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        regions_sel = st.multiselect("Regions to Collect", config.SUPPORTED_REGIONS,
                                      default=["IN", "US"])
        max_results = st.slider("Videos per Region", 10, 200, 50)
        collect_btn = st.button("📥 Collect Trending Data")

    with col2:
        if collect_btn:
            api_k = os.environ.get("YOUTUBE_API_KEY", "")
            if not api_k:
                st.error("⚠️ Please enter your YouTube API key in the sidebar.")
            else:
                collector = YouTubeCollector(api_k)
                all_data  = {}
                prog = st.progress(0)
                for i, region in enumerate(regions_sel):
                    with st.spinner(f"Fetching {region}…"):
                        all_data[region] = collector.fetch_trending(region, max_results)
                    prog.progress((i + 1) / len(regions_sel))

                st.success(f"✅ Collected data for {len(regions_sel)} region(s).")

                import pandas as pd
                for region, data in all_data.items():
                    with st.expander(f"📋 {region} — {len(data)} videos"):
                        st.dataframe(pd.DataFrame(data), use_container_width=True)

                # Download
                import json
                json_str = json.dumps(all_data, indent=2)
                st.download_button("⬇️ Download JSON", json_str,
                                   "trending_data.json", "application/json")

# ══════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<h2 class="section-header">About This Project</h2>', unsafe_allow_html=True)
    st.markdown("""
    ## Region-Based Analysis of YouTube Trending Videos using Thumbnails

    **A Final Year Mini-Project in Computer Science**

    ### 🎯 Objectives
    - Analyze how thumbnail visual features influence YouTube trending behavior
    - Compare thumbnail design patterns across regions (India, USA, UK, etc.)
    - Predict trending potential using ML on combined thumbnail + metadata features
    - Detect misleading thumbnails via multi-modal mismatch analysis

    ### 🏗️ Tech Stack
    | Layer | Technology |
    |-------|------------|
    | Frontend | Streamlit |
    | Image Processing | OpenCV, NumPy, Pillow |
    | Machine Learning | scikit-learn |
    | Visualization | Matplotlib, Seaborn |
    | Data Collection | YouTube Data API v3 |

    ### 📚 Research Insights Applied
    - **Multi-modal Analysis**: Combining visual features with text (title/description)
    - **Misleading Thumbnail Detection**: Identifying thumbnail-title mismatches
    - **Cultural Patterns**: Regional differences in color, brightness, and face usage
    - **Clickbait Indicators**: Face count, saturation, edge density correlations

    ### 📁 Module Architecture
    ```
    app.py                    ← Streamlit entry point
    config.py                 ← Constants and configuration
    modules/
      ├── youtube_collector.py  ← YouTube API wrapper
      ├── image_analyzer.py     ← OpenCV feature extraction
      ├── ml_model.py           ← scikit-learn ML pipeline
      ├── visualizer.py         ← Matplotlib charts
      ├── misleading_detector.py← Multi-modal mismatch detection
      └── region_analyzer.py    ← Regional statistical analysis
    data/
      ├── sample/               ← Sample datasets (no API key needed)
      └── collected/            ← Your collected data
    models/
      └── trend_predictor.pkl   ← Saved ML model
    ```
    """)

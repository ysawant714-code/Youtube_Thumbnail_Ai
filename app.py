import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Analyzer", layout="wide")

# 🎨 UI Title
st.title("📊 YouTube Thumbnail Analyzer (Advanced)")
st.markdown("Analyze thumbnails using AI 🔥")

# 🔘 Sidebar options
option = st.sidebar.selectbox(
    "Choose Input Type",
    ["YouTube Link", "Thumbnail URL", "Upload Image"]
)

# 🎥 Extract thumbnail from YouTube link
def get_thumbnail(youtube_url):
    try:
        video_id = youtube_url.split("v=")[1]
        return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    except:
        return None

# 🧠 Analyze function
def analyze_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return brightness, len(faces)

# 🎨 NEW FEATURE: Color Analysis
def get_dominant_color(img):
    img = np.array(img)
    avg_color = img.mean(axis=0).mean(axis=0)

    r, g, b = avg_color

    if r > g and r > b:
        return "Red 🔴"
    elif g > r and g > b:
        return "Green 🟢"
    else:
        return "Blue 🔵"

img = None

# 🔹 Option 1: YouTube link
if option == "YouTube Link":
    yt_link = st.text_input("Enter YouTube Video Link")

    if yt_link:
        thumb_url = get_thumbnail(yt_link)
        if thumb_url:
            response = requests.get(thumb_url)
            img = Image.open(BytesIO(response.content))
        else:
            st.error("Invalid YouTube link")

# 🔹 Option 2: Thumbnail URL
elif option == "Thumbnail URL":
    url = st.text_input("Enter Thumbnail URL")

    if url:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

# 🔹 Option 3: Upload Image
elif option == "Upload Image":
    file = st.file_uploader("Upload Thumbnail", type=["jpg", "png"])
    
    if file:
        img = Image.open(file)

# 🔍 Analyze button
if st.button("Analyze"):
    if img is not None:
        st.image(img, caption="Thumbnail Preview", use_column_width=True)

        brightness, faces = analyze_image(img)
        color = get_dominant_color(img)   # NEW

        col1, col2, col3 = st.columns(3)   # UPDATED
        col1.metric("Brightness", round(brightness, 2))
        col2.metric("Faces Detected", faces)
        col3.metric("Dominant Color", color)  # NEW

        # Prediction
        if brightness > 120:
            st.success("🔥 High Engagement Thumbnail")
        else:
            st.warning("⚠️ Low Engagement Thumbnail")

        # 📊 Graph
        st.subheader("📊 Analysis Graph")
        x = ["Brightness", "Faces"]
        y = [brightness, faces]

        fig, ax = plt.subplots()
        ax.bar(x, y)
        st.pyplot(fig)

        # 📄 NEW FEATURE: Download Report
        report = f"""
YouTube Thumbnail Report

Brightness: {brightness}
Faces Detected: {faces}
Dominant Color: {color}
"""

        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="report.txt",
            mime="text/plain"
        )

    else:
        st.error("Please provide input")

# 🌍 Region comparison (same as before)
st.subheader("🌍 Region Comparison (Demo)")

regions = ["India", "USA"]
values = [120, 150]

fig2, ax2 = plt.subplots()
ax2.bar(regions, values)
st.pyplot(fig2)

st.info("USA thumbnails generally have higher brightness (demo data)") ya code madhe youtube sathi tumbalie sathi
Choose Input Type

Upload Image  ya sathi try sathi data de jo ki mazya web ver takhun run krun bghu shkhte 
"""
modules/image_analyzer.py
──────────────────────────
OpenCV-based thumbnail feature extraction.

Features extracted:
  • Brightness, Contrast (RMS)
  • Saturation & Hue statistics (HSV)
  • Edge density (Canny)
  • Color channel ratios and diversity
  • Face count (Haar cascades)
  • Sharpness (Laplacian variance)
  • Approximate text-region density
"""

import cv2
import numpy as np
from typing import Optional
import config


class ImageAnalyzer:
    """Extracts a rich feature vector from a BGR thumbnail image."""

    def __init__(self):
        # Load Haar cascades (bundled with OpenCV)
        self.face_cascade    = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

    # ── Master method ─────────────────────────────────────────────────────────
    def extract_all_features(self, img_bgr: np.ndarray) -> dict:
        """
        Run all feature extractors and return a flat dict of numeric features.
        All values are Python floats/ints (safe for pandas / sklearn).
        """
        img = self._resize(img_bgr)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        brightness, contrast = self._brightness_contrast(gray)
        sat_mean, sat_std    = self._saturation_stats(hsv)
        hue_mean             = self._hue_mean(hsv)
        edge_density         = self._edge_density(gray)
        face_count           = self._count_faces(gray)
        r_ratio, g_ratio, b_ratio = self._color_ratios(img)
        sharpness            = self._sharpness(gray)
        color_diversity      = self._color_diversity(img)
        text_density         = self._text_region_density(gray)

        return {
            "brightness":       float(brightness),
            "contrast":         float(contrast),
            "saturation_mean":  float(sat_mean),
            "saturation_std":   float(sat_std),
            "hue_mean":         float(hue_mean),
            "edge_density":     float(edge_density),
            "face_count":       int(face_count),
            "has_face":         int(face_count > 0),
            "red_ratio":        float(r_ratio),
            "green_ratio":      float(g_ratio),
            "blue_ratio":       float(b_ratio),
            "sharpness":        float(sharpness),
            "color_diversity":  float(color_diversity),
            "text_region_density": float(text_density),
        }

    # ── Brightness & Contrast ─────────────────────────────────────────────────
    def _brightness_contrast(self, gray: np.ndarray) -> tuple[float, float]:
        """
        Brightness = mean pixel value (0–255).
        Contrast   = RMS contrast (std of pixel values).
        """
        brightness = float(np.mean(gray))
        contrast   = float(np.std(gray))
        return brightness, contrast

    # ── Saturation ────────────────────────────────────────────────────────────
    def _saturation_stats(self, hsv: np.ndarray) -> tuple[float, float]:
        sat = hsv[:, :, 1].astype(np.float32)
        return float(np.mean(sat)), float(np.std(sat))

    # ── Hue ───────────────────────────────────────────────────────────────────
    def _hue_mean(self, hsv: np.ndarray) -> float:
        return float(np.mean(hsv[:, :, 0]))

    # ── Edge Density ──────────────────────────────────────────────────────────
    def _edge_density(self, gray: np.ndarray) -> float:
        """
        Fraction of pixels that are edges (Canny).
        Higher → busier / more complex thumbnail.
        """
        lo, hi = config.EDGE_THRESHOLD
        edges   = cv2.Canny(gray, lo, hi)
        return float(np.sum(edges > 0) / edges.size)

    def detect_edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return the Canny edge map (for visualization)."""
        gray = cv2.cvtColor(self._resize(img_bgr), cv2.COLOR_BGR2GRAY)
        lo, hi = config.EDGE_THRESHOLD
        return cv2.Canny(gray, lo, hi)

    # ── Face Detection ────────────────────────────────────────────────────────
    def _count_faces(self, gray: np.ndarray) -> int:
        frontal = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        profile = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(frontal) + len(profile)

    def detect_faces(self, img_bgr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Detect faces and return (annotated_image_bgr, list_of_face_dicts).
        Each face dict: {x, y, w, h, area_ratio}.
        """
        img  = self._resize(img_bgr).copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        frontal = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        profile = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        faces_info = []
        all_faces  = list(frontal) + list(profile) if len(frontal) and len(profile) else \
                     list(frontal) if len(frontal) else list(profile)

        for (fx, fy, fw, fh) in all_faces:
            cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            faces_info.append({
                "x": fx, "y": fy, "w": fw, "h": fh,
                "area_ratio": round(fw * fh / (w * h), 4),
            })

        return img, faces_info

    # ── Color Ratios ──────────────────────────────────────────────────────────
    def _color_ratios(self, img: np.ndarray) -> tuple[float, float, float]:
        total = img.sum() + 1e-6
        b_ratio = img[:, :, 0].sum() / total
        g_ratio = img[:, :, 1].sum() / total
        r_ratio = img[:, :, 2].sum() / total
        return float(r_ratio), float(g_ratio), float(b_ratio)

    def get_color_histogram(self, img_bgr: np.ndarray, bins: int = None) -> dict:
        """Return per-channel histograms as dict of arrays."""
        bins = bins or config.COLOR_BINS
        img  = self._resize(img_bgr)
        hists = {}
        for i, ch in enumerate(["blue", "green", "red"]):
            hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
            hists[ch] = hist.flatten() / hist.sum()   # normalize
        return hists

    # ── Sharpness ─────────────────────────────────────────────────────────────
    def _sharpness(self, gray: np.ndarray) -> float:
        """Laplacian variance — higher means sharper image."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── Color Diversity ───────────────────────────────────────────────────────
    def _color_diversity(self, img: np.ndarray, k: int = 5) -> float:
        """
        Run k-means on a downsampled pixel set and return the mean
        inter-centroid distance — a proxy for color diversity.
        """
        small = cv2.resize(img, (64, 36))
        pixels = small.reshape(-1, 3).astype(np.float32)
        if len(pixels) < k:
            return 0.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        # Mean pairwise distance between centroids
        dists = []
        for i in range(k):
            for j in range(i + 1, k):
                dists.append(np.linalg.norm(centers[i] - centers[j]))
        return float(np.mean(dists)) if dists else 0.0

    # ── Text Region Density ───────────────────────────────────────────────────
    def _text_region_density(self, gray: np.ndarray) -> float:
        """
        Approximate fraction of pixels likely belonging to text/overlay regions
        using MSER (Maximally Stable Extremal Regions).
        """
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            if not regions:
                return 0.0
            mask = np.zeros_like(gray)
            for pts in regions:
                hull = cv2.convexHull(pts.reshape(-1, 1, 2))
                cv2.drawContours(mask, [hull], -1, 255, -1)
            return float(np.sum(mask > 0) / mask.size)
        except Exception:
            return 0.0

    # ── Dominant Colors ───────────────────────────────────────────────────────
    def get_dominant_colors(self, img_bgr: np.ndarray, k: int = 5) -> list[dict]:
        """Return the top-k dominant colors as RGB tuples with percentage."""
        small  = cv2.resize(img_bgr, (100, 56))
        pixels = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        colors = []
        for idx, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            c = centers[idx].astype(int)
            colors.append({
                "b": int(c[0]), "g": int(c[1]), "r": int(c[2]),
                "hex": f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}",
                "percentage": round(cnt / total * 100, 1),
            })
        return colors

    # ── Utility ───────────────────────────────────────────────────────────────
    @staticmethod
    def _resize(img: np.ndarray,
                w: int = config.THUMBNAIL_WIDTH,
                h: int = config.THUMBNAIL_HEIGHT) -> np.ndarray:
        """Resize to standard thumbnail dimensions (letterbox-safe)."""
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

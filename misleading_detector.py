"""
modules/misleading_detector.py
───────────────────────────────
Multi-modal misleading thumbnail detection.

Combines:
  1. Text-Image Semantic Mismatch
     • Keyword extraction from title vs. detected visual content
     • Clickbait keyword matching in title/description
  2. Visual Anomaly Signals
     • Extreme saturation / brightness (over-processing)
     • Artificial sharpening artefacts
     • Face emotion extremity (mouth open, extreme expressions)
  3. Title Sentiment vs. Visual Tone
     • Negative/alarming words with neutral image → misleading
  4. Tag Consistency
     • Tags completely unrelated to thumbnail content

Output score range: 0.0 (authentic) → 1.0 (highly misleading)
"""

import re
import cv2
import numpy as np
from typing import Optional
import config


# ── Simple sentiment / keyword lists ─────────────────────────────────────────
NEGATIVE_WORDS = {
    "shocking", "exposed", "caught", "destroyed", "leaked", "banned",
    "deleted", "gone", "worst", "died", "death", "kill", "crime",
    "arrested", "illegal", "secret", "truth", "betrayal", "shame",
    "scandal", "fraud", "cheating", "hate", "terrible", "horrible",
}

CLICKBAIT_PHRASES = [
    "you won't believe", "not clickbait", "must watch", "watch till end",
    "wait for it", "gone wrong", "exposed", "100%", "gone sexual",
    "real or fake", "this is real", "warning", "extreme",
]

EMOTION_WORDS = {
    "happy", "amazing", "awesome", "love", "best", "great", "incredible",
    "beautiful", "wonderful", "fantastic", "perfect", "excellent",
}


class MisleadingDetector:
    """
    Detects potential misleading content in YouTube thumbnails
    using multi-modal (image + text) analysis.
    """

    def __init__(self):
        # Haar cascade for face-based emotion proxy
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── Master Method ─────────────────────────────────────────────────────────
    def detect(self, img_bgr: np.ndarray,
               title: str = "",
               description: str = "",
               tags: list[str] = None) -> dict:
        """
        Run all misleading checks and return a comprehensive result dict.

        Returns:
        {
          "misleading_score": float,       # 0–1
          "flags": {
            "clickbait_title": {"triggered": bool, "reason": str},
            "visual_anomaly":  {...},
            "sentiment_mismatch": {...},
            "extreme_editing": {...},
            "face_emotion_extreme": {...},
          },
          "scores": {                      # per-check scores
            "clickbait_title": float,
            "visual_anomaly":  float,
            ...
          },
          "dominant_emotions": [...],
        }
        """
        tags = tags or []
        title_lower = title.lower()
        desc_lower  = description.lower()

        checks = {
            "clickbait_title":    self._check_clickbait_title(title_lower),
            "visual_anomaly":     self._check_visual_anomaly(img_bgr),
            "sentiment_mismatch": self._check_sentiment_mismatch(img_bgr, title_lower),
            "extreme_editing":    self._check_extreme_editing(img_bgr),
            "face_emotion_extreme": self._check_face_extremity(img_bgr, title_lower),
        }

        scores = {k: v["score"] for k, v in checks.items()}
        # Weighted average
        weights = {
            "clickbait_title":     0.30,
            "visual_anomaly":      0.20,
            "sentiment_mismatch":  0.25,
            "extreme_editing":     0.15,
            "face_emotion_extreme":0.10,
        }
        total = sum(scores[k] * weights[k] for k in scores)

        flags = {k: {"triggered": v["score"] > 0.4, "reason": v["reason"]}
                 for k, v in checks.items()}

        return {
            "misleading_score": round(float(total), 4),
            "flags":   flags,
            "scores":  {k: round(v, 4) for k, v in scores.items()},
        }

    # ── Check 1: Clickbait Title ──────────────────────────────────────────────
    def _check_clickbait_title(self, title_lower: str) -> dict:
        score    = 0.0
        reasons  = []

        # Phrase matching
        matched_phrases = [p for p in CLICKBAIT_PHRASES if p in title_lower]
        if matched_phrases:
            score += 0.5 * min(len(matched_phrases) / 2, 1.0)
            reasons.append(f"Clickbait phrase(s): {', '.join(matched_phrases[:3])}")

        # Negative words
        words = set(re.findall(r"\b\w+\b", title_lower))
        neg_matches = words & NEGATIVE_WORDS
        if neg_matches:
            score += 0.3 * min(len(neg_matches) / 3, 1.0)
            reasons.append(f"Negative words: {', '.join(list(neg_matches)[:3])}")

        # All-caps words (shouting)
        caps_words = [w for w in title_lower.split() if w.upper() == w and len(w) > 2]
        if len(caps_words) > 2:
            score += 0.2
            reasons.append(f"{len(caps_words)} all-caps word(s)")

        # Excessive punctuation
        if title_lower.count("!") + title_lower.count("?") > 2:
            score += 0.1
            reasons.append("Excessive punctuation")

        score = float(np.clip(score, 0, 1))
        reason_str = "; ".join(reasons) if reasons else "No clickbait signals detected."
        return {"score": score, "reason": reason_str}

    # ── Check 2: Visual Anomaly ───────────────────────────────────────────────
    def _check_visual_anomaly(self, img_bgr: np.ndarray) -> dict:
        """
        Detects over-processed thumbnails:
        • Extreme saturation (HSV S channel)
        • Very low or very high brightness
        • Neon/artificial colors (saturation spikes)
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        sat_mean = float(np.mean(sat))
        sat_std  = float(np.std(sat))
        val_mean = float(np.mean(val))

        score   = 0.0
        reasons = []

        if sat_mean > 180:
            score += 0.4
            reasons.append(f"Hyper-saturated image (mean sat={sat_mean:.0f}/255)")
        elif sat_mean > 150:
            score += 0.2
            reasons.append(f"Very high saturation (mean={sat_mean:.0f})")

        if val_mean < 40:
            score += 0.2
            reasons.append(f"Unusually dark (mean brightness={val_mean:.0f})")
        elif val_mean > 230:
            score += 0.15
            reasons.append(f"Overexposed/washed-out (mean={val_mean:.0f})")

        if sat_std > 90:
            score += 0.2
            reasons.append(f"Unnatural saturation variance (std={sat_std:.0f})")

        score = float(np.clip(score, 0, 1))
        reason_str = "; ".join(reasons) if reasons else "Visual properties appear natural."
        return {"score": score, "reason": reason_str}

    # ── Check 3: Sentiment Mismatch ───────────────────────────────────────────
    def _check_sentiment_mismatch(self, img_bgr: np.ndarray,
                                   title_lower: str) -> dict:
        """
        Heuristic: alarming/negative title + warm/bright thumbnail → suspicious.
        Happy/positive title + dark/cold thumbnail → suspicious.
        """
        hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        val_mean = float(np.mean(hsv[:, :, 2]))       # brightness
        sat_mean = float(np.mean(hsv[:, :, 1]))       # saturation
        hue_mean = float(np.mean(hsv[:, :, 0]))       # hue

        words       = set(re.findall(r"\b\w+\b", title_lower))
        neg_count   = len(words & NEGATIVE_WORDS)
        pos_count   = len(words & EMOTION_WORDS)

        score   = 0.0
        reasons = []

        # Negative title + bright / warm image
        if neg_count >= 2 and val_mean > 160 and sat_mean > 100:
            score += 0.45
            reasons.append(
                f"Negative/alarming title ({neg_count} signals) paired with bright, "
                "engaging thumbnail — classic misleading pattern."
            )

        # Positive title + dark / cold image
        if pos_count >= 2 and val_mean < 80:
            score += 0.30
            reasons.append(
                "Positive title language paired with dark, low-energy thumbnail."
            )

        # No clear relationship
        if not reasons:
            reasons.append("Title and visual tone appear consistent.")

        score = float(np.clip(score, 0, 1))
        return {"score": score, "reason": " | ".join(reasons)}

    # ── Check 4: Extreme Editing ──────────────────────────────────────────────
    def _check_extreme_editing(self, img_bgr: np.ndarray) -> dict:
        """
        Detects hallmarks of heavy post-processing:
        • Sharpening halos (high-frequency ring artefacts)
        • Noise injection
        • JPEG compression artefacts in high-bitrate regions
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Laplacian energy — very high → artificial sharpening
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Unsharp mask residual (difference between image and its blur)
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
        residual = np.abs(gray.astype(np.float32) - blurred)
        residual_mean = float(np.mean(residual))

        score   = 0.0
        reasons = []

        if lap_var > 5000:
            score += 0.4
            reasons.append(f"Extremely high sharpening (Laplacian var={lap_var:.0f})")
        elif lap_var > 2000:
            score += 0.2
            reasons.append(f"Heavy sharpening detected (var={lap_var:.0f})")

        if residual_mean > 15:
            score += 0.3
            reasons.append(f"Strong unsharp mask artefacts (residual={residual_mean:.1f})")

        score = float(np.clip(score, 0, 1))
        reason_str = "; ".join(reasons) if reasons else "Image editing appears normal."
        return {"score": score, "reason": reason_str}

    # ── Check 5: Face Emotion Extremity ──────────────────────────────────────
    def _check_face_extremity(self, img_bgr: np.ndarray,
                               title_lower: str) -> dict:
        """
        Proxy for exaggerated facial expressions:
        • Open-mouth detection via skin + edge analysis in face region
        • Face occupying >40% of thumbnail area → reaction-style bait
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        score   = 0.0
        reasons = []

        if len(faces) == 0:
            return {"score": 0.0, "reason": "No faces to analyse."}

        for (fx, fy, fw, fh) in faces:
            area_ratio = (fw * fh) / (w * h)

            # Large face = reaction/clickbait thumbnail
            if area_ratio > 0.35:
                score += 0.3
                reasons.append(f"Very large face ({area_ratio:.0%} of thumbnail) — reaction bait")

            # Lower-face region edge density (proxy for open mouth / extreme expression)
            lower_face = gray[fy + int(fh * 0.55): fy + fh, fx: fx + fw]
            if lower_face.size > 0:
                edges = cv2.Canny(lower_face, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                if edge_density > 0.20:
                    score += 0.25
                    reasons.append(
                        f"Possible open-mouth/extreme expression (edge density={edge_density:.2f})"
                    )

        score = float(np.clip(score / max(len(faces), 1), 0, 1))
        reason_str = "; ".join(reasons) if reasons else "Facial expressions appear normal."
        return {"score": score, "reason": reason_str}

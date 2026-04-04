"""
modules/region_analyzer.py
───────────────────────────
Computes and compares thumbnail feature statistics across YouTube regions.

Key capabilities:
  • Statistical aggregation of features per region
  • Region-vs-region comparison (mean, std, delta)
  • Sample dataset generation (no API key required)
  • Cultural pattern insights
"""

import os
import json
import random
import numpy as np
from typing import Optional
import config


# ── Regional cultural priors (from research literature) ──────────────────────
# Used to generate realistic synthetic data per region
REGIONAL_PRIORS = {
    "IN": {  # India: high brightness, high saturation, many faces
        "brightness":       (165, 20),
        "contrast":         (55,  12),
        "saturation_mean":  (155, 25),
        "edge_density":     (0.13, 0.04),
        "face_count":       (1.4, 0.8),
        "color_diversity":  (110, 20),
        "sharpness":        (700, 200),
        "views_mean":       (4_500_000, 1_500_000),
        "likes_ratio":      (0.040, 0.01),
        "comments_ratio":   (0.008, 0.003),
    },
    "US": {  # USA: moderate brightness, clean/bold design
        "brightness":       (145, 25),
        "contrast":         (60,  15),
        "saturation_mean":  (125, 30),
        "edge_density":     (0.11, 0.04),
        "face_count":       (1.1, 0.7),
        "color_diversity":  (95,  25),
        "sharpness":        (800, 250),
        "views_mean":       (5_000_000, 2_000_000),
        "likes_ratio":      (0.035, 0.012),
        "comments_ratio":   (0.006, 0.002),
    },
    "JP": {  # Japan: lower saturation, more text-heavy, distinctive aesthetics
        "brightness":       (138, 22),
        "contrast":         (50,  12),
        "saturation_mean":  (100, 28),
        "edge_density":     (0.16, 0.05),
        "face_count":       (0.8, 0.6),
        "color_diversity":  (80,  20),
        "sharpness":        (750, 200),
        "views_mean":       (3_000_000, 1_000_000),
        "likes_ratio":      (0.025, 0.008),
        "comments_ratio":   (0.004, 0.002),
    },
    "BR": {  # Brazil: vibrant colors, high saturation
        "brightness":       (158, 22),
        "contrast":         (58,  13),
        "saturation_mean":  (148, 28),
        "edge_density":     (0.12, 0.04),
        "face_count":       (1.3, 0.7),
        "color_diversity":  (115, 22),
        "sharpness":        (680, 180),
        "views_mean":       (3_500_000, 1_200_000),
        "likes_ratio":      (0.038, 0.011),
        "comments_ratio":   (0.009, 0.003),
    },
    "GB": {  # UK: muted palette, editorial style
        "brightness":       (132, 22),
        "contrast":         (62,  15),
        "saturation_mean":  (105, 28),
        "edge_density":     (0.10, 0.035),
        "face_count":       (1.0, 0.6),
        "color_diversity":  (88,  22),
        "sharpness":        (820, 220),
        "views_mean":       (2_800_000, 900_000),
        "likes_ratio":      (0.028, 0.009),
        "comments_ratio":   (0.005, 0.002),
    },
    "KR": {  # South Korea: high contrast, idol/face-heavy
        "brightness":       (155, 20),
        "contrast":         (65,  14),
        "saturation_mean":  (135, 25),
        "edge_density":     (0.13, 0.04),
        "face_count":       (1.5, 0.7),
        "color_diversity":  (105, 20),
        "sharpness":        (900, 220),
        "views_mean":       (4_000_000, 1_200_000),
        "likes_ratio":      (0.045, 0.012),
        "comments_ratio":   (0.010, 0.003),
    },
    "DE": {  # Germany: clean, functional
        "brightness":       (130, 20),
        "contrast":         (58,  13),
        "saturation_mean":  (95,  25),
        "edge_density":     (0.09, 0.03),
        "face_count":       (0.9, 0.6),
        "color_diversity":  (82,  18),
        "sharpness":        (850, 200),
        "views_mean":       (2_200_000, 800_000),
        "likes_ratio":      (0.022, 0.008),
        "comments_ratio":   (0.004, 0.002),
    },
    "FR": {
        "brightness":       (135, 22),
        "contrast":         (56,  13),
        "saturation_mean":  (102, 26),
        "edge_density":     (0.10, 0.035),
        "face_count":       (1.0, 0.6),
        "color_diversity":  (90,  20),
        "sharpness":        (780, 200),
        "views_mean":       (2_000_000, 700_000),
        "likes_ratio":      (0.024, 0.009),
        "comments_ratio":   (0.005, 0.002),
    },
    "MX": {
        "brightness":       (160, 22),
        "contrast":         (57,  13),
        "saturation_mean":  (145, 28),
        "edge_density":     (0.12, 0.04),
        "face_count":       (1.2, 0.7),
        "color_diversity":  (112, 22),
        "sharpness":        (670, 180),
        "views_mean":       (3_200_000, 1_100_000),
        "likes_ratio":      (0.036, 0.011),
        "comments_ratio":   (0.008, 0.003),
    },
    "AU": {
        "brightness":       (148, 22),
        "contrast":         (60,  14),
        "saturation_mean":  (118, 27),
        "edge_density":     (0.10, 0.035),
        "face_count":       (1.1, 0.7),
        "color_diversity":  (92,  21),
        "sharpness":        (800, 210),
        "views_mean":       (1_800_000, 700_000),
        "likes_ratio":      (0.030, 0.010),
        "comments_ratio":   (0.006, 0.002),
    },
}

_DEFAULT_PRIOR = {
    "brightness": (140, 25), "contrast": (55, 13),
    "saturation_mean": (120, 28), "edge_density": (0.11, 0.04),
    "face_count": (1.0, 0.7), "color_diversity": (95, 22),
    "sharpness": (750, 200), "views_mean": (3_000_000, 1_000_000),
    "likes_ratio": (0.030, 0.010), "comments_ratio": (0.005, 0.002),
}

FEATURE_KEYS = [
    "brightness", "contrast", "saturation_mean",
    "edge_density", "face_count", "color_diversity", "sharpness",
]


class RegionAnalyzer:
    """Statistical analysis of thumbnail features grouped by region."""

    # ── Sample Data Generation ────────────────────────────────────────────────
    def load_sample_data(self, region: str, n: int = 50) -> list[dict]:
        """
        Generate or load synthetic per-region video data
        that approximates real cultural patterns.
        """
        # Check for cached real data first
        cache_path = os.path.join(config.SAMPLE_DATA_DIR, f"{region}_sample.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        return self._generate_synthetic(region, n)

    def _generate_synthetic(self, region: str, n: int) -> list[dict]:
        """Generate n synthetic video records for a region."""
        priors = REGIONAL_PRIORS.get(region, _DEFAULT_PRIOR)
        random.seed(hash(region) % 2**31)
        np.random.seed(hash(region) % 2**31)

        records = []
        for _ in range(n):
            def sample(key):
                mu, sigma = priors.get(key, _DEFAULT_PRIOR.get(key, (100, 20)))
                return float(np.clip(np.random.normal(mu, sigma), 0, None))

            views = max(int(sample("views_mean")), 10000)
            likes_ratio    = float(np.clip(np.random.normal(*priors.get("likes_ratio", (0.03, 0.01))), 0.001, 0.2))
            comments_ratio = float(np.clip(np.random.normal(*priors.get("comments_ratio", (0.005, 0.002))), 0.001, 0.05))

            records.append({
                "region":          region,
                "views":           views,
                "likes":           int(views * likes_ratio),
                "comments":        int(views * comments_ratio),
                "brightness":      sample("brightness"),
                "contrast":        sample("contrast"),
                "saturation_mean": sample("saturation_mean"),
                "edge_density":    float(np.clip(np.random.normal(
                    *priors.get("edge_density", (0.11, 0.04))), 0.01, 0.4)),
                "face_count":      max(0, int(round(sample("face_count")))),
                "has_face":        int(sample("face_count") >= 0.8),
                "color_diversity": sample("color_diversity"),
                "sharpness":       sample("sharpness"),
            })

        return records

    # ── Feature Aggregation ───────────────────────────────────────────────────
    def aggregate(self, data: list[dict]) -> dict:
        """Compute mean of each feature across a list of video dicts."""
        if not data:
            return {}
        agg = {}
        for key in FEATURE_KEYS:
            vals = [d.get(key, 0) for d in data if key in d]
            agg[key] = round(float(np.mean(vals)), 3) if vals else 0.0
        return agg

    # ── Region Comparison ─────────────────────────────────────────────────────
    def compare_regions(self,
                         data_a: list[dict], data_b: list[dict],
                         region_a: str, region_b: str) -> dict:
        """
        Compare mean feature values for two regions.

        Returns:
        {
          feature_name: {
            region_a: mean_value,
            region_b: mean_value,
            delta: a - b,
            delta_pct: (a-b)/b * 100,
          },
          ...
        }
        """
        agg_a = self.aggregate(data_a)
        agg_b = self.aggregate(data_b)

        result = {}
        for key in FEATURE_KEYS:
            va = agg_a.get(key, 0)
            vb = agg_b.get(key, 0)
            delta = va - vb
            delta_pct = (delta / vb * 100) if vb != 0 else 0.0
            result[key] = {
                region_a:    round(va, 3),
                region_b:    round(vb, 3),
                "delta":     round(delta, 3),
                "delta_pct": round(delta_pct, 1),
            }
        return result

    # ── Cultural Insights ─────────────────────────────────────────────────────
    def generate_insights(self, comparison: dict,
                           region_a: str, region_b: str) -> list[str]:
        """Return human-readable insights from a comparison dict."""
        insights = []

        # Brightness
        delta = comparison.get("brightness", {}).get("delta", 0)
        if abs(delta) > 10:
            higher = region_a if delta > 0 else region_b
            insights.append(
                f"🌟 **{higher}** thumbnails are significantly brighter "
                f"({abs(delta):.1f} pts), typical of warmer-climate / entertainment-heavy markets."
            )

        # Saturation
        delta = comparison.get("saturation_mean", {}).get("delta", 0)
        if abs(delta) > 15:
            higher = region_a if delta > 0 else region_b
            insights.append(
                f"🎨 **{higher}** uses more saturated colors ({abs(delta):.1f} pts delta). "
                "High saturation is correlated with clickbait and youth-oriented content."
            )

        # Faces
        delta = comparison.get("face_count", {}).get("delta", 0)
        if abs(delta) > 0.2:
            higher = region_a if delta > 0 else region_b
            insights.append(
                f"😊 **{higher}** thumbnails feature more faces on average. "
                "Face-heavy thumbnails drive higher CTR in personality-driven content cultures."
            )

        # Edge density
        delta = comparison.get("edge_density", {}).get("delta", 0)
        if abs(delta) > 0.02:
            higher = region_a if delta > 0 else region_b
            insights.append(
                f"📐 **{higher}** thumbnails are visually busier (edge density +{abs(delta):.3f}). "
                "Higher edge density may indicate more text overlays or complex compositions."
            )

        if not insights:
            insights.append("Both regions show similar thumbnail design patterns.")

        return insights

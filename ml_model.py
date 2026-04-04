"""
modules/ml_model.py
────────────────────
scikit-learn pipeline for predicting YouTube trending potential.

Models used:
  • RandomForestClassifier  (primary — handles non-linear feature interactions)
  • LogisticRegression       (baseline — interpretable coefficients)
  • GradientBoostingClassifier (optional boost)

The pipeline:
    raw features → StandardScaler → RandomForest → trending probability

Training:
    Uses synthetic data by default (no YouTube API key needed).
    Call trainer.train_on_real_data(videos, images) with actual data.
"""

import os
import json
import math
import pickle
import numpy as np
from typing import Optional
import config

# sklearn imports (gracefully degrade if not installed)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TrendPredictor:
    """
    Predicts whether a video will trend based on thumbnail features + metadata.

    Output:
        {
          "trend_score":        0.0–1.0,   # probability of trending
          "label":              "trending" | "not trending",
          "confidence":         0.0–1.0,
          "feature_importance": {feature: importance, ...},
          "suggestions":        [str, ...],
        }
    """

    FEATURE_NAMES = config.FEATURE_COLUMNS

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.feature_names = self.FEATURE_NAMES
        self._load_or_train()

    # ── Load / Train ──────────────────────────────────────────────────────────
    def _load_or_train(self):
        """Load saved model or train on synthetic data."""
        if os.path.exists(config.MODEL_PATH):
            try:
                with open(config.MODEL_PATH, "rb") as f:
                    self.pipeline = pickle.load(f)
                print("[TrendPredictor] Loaded saved model.")
                return
            except Exception:
                pass
        self._train_synthetic()

    def _train_synthetic(self):
        """
        Train on synthetically generated data that approximates known
        patterns from YouTube trend research:
          - Higher brightness + saturation → more likely to trend
          - Face presence → strong positive signal
          - Edge density (moderate) → positive; extremes → negative
          - Longer titles with numbers → positive
        """
        if not SKLEARN_AVAILABLE:
            print("[TrendPredictor] scikit-learn not available. Predictions will be rule-based.")
            return

        np.random.seed(config.RANDOM_STATE)
        n = 2000

        def make_sample():
            brightness      = np.random.uniform(60, 220)
            contrast        = np.random.uniform(20, 90)
            saturation_mean = np.random.uniform(40, 200)
            saturation_std  = np.random.uniform(10, 80)
            hue_mean        = np.random.uniform(0, 180)
            edge_density    = np.random.uniform(0.02, 0.30)
            face_count      = np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.15, 0.10])
            has_face        = int(face_count > 0)
            red_ratio       = np.random.uniform(0.2, 0.5)
            green_ratio     = np.random.uniform(0.15, 0.45)
            blue_ratio      = 1.0 - red_ratio - green_ratio
            sharpness       = np.random.uniform(50, 2000)
            color_diversity = np.random.uniform(30, 150)
            text_density    = np.random.uniform(0.01, 0.4)

            title_length       = np.random.randint(5, 80)
            title_has_numbers  = np.random.choice([0, 1], p=[0.5, 0.5])
            title_has_caps     = np.random.choice([0, 1], p=[0.6, 0.4])
            duration_minutes   = np.random.uniform(1, 60)
            subscribers_log    = np.random.uniform(3, 8)

            # Label: trending if several positive signals align
            score = (
                0.20 * (brightness / 220) +
                0.15 * (saturation_mean / 200) +
                0.15 * has_face +
                0.10 * (1.0 - abs(edge_density - 0.12) / 0.12) +
                0.10 * (sharpness / 2000) +
                0.08 * (color_diversity / 150) +
                0.05 * (title_has_numbers) +
                0.05 * (title_has_caps) +
                0.07 * (subscribers_log / 8) +
                0.05 * (1.0 - abs(duration_minutes - 12) / 48)
            )
            label = int(score + np.random.normal(0, 0.08) > 0.52)

            return [
                brightness, contrast, saturation_mean, saturation_std,
                hue_mean, edge_density, face_count, has_face,
                red_ratio, green_ratio, blue_ratio, sharpness,
                color_diversity, text_density, title_length,
                title_has_numbers, title_has_caps, duration_minutes, subscribers_log,
            ], label

        X_list, y_list = zip(*[make_sample() for _ in range(n)])
        X = np.array(X_list)
        y = np.array(y_list)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            )),
        ])
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"[TrendPredictor] Synthetic model trained | AUC: {auc:.3f}")

        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        with open(config.MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, image_features: dict, metadata: dict) -> dict:
        """
        Predict trending potential.

        Args:
            image_features: dict from ImageAnalyzer.extract_all_features()
            metadata: {title, region, subscribers, duration_minutes}

        Returns:
            prediction dict (see class docstring)
        """
        feature_vec = self._build_feature_vector(image_features, metadata)

        if self.pipeline is not None and SKLEARN_AVAILABLE:
            prob = self.pipeline.predict_proba([feature_vec])[0][1]
            importance = self._get_feature_importance(feature_vec)
        else:
            prob = self._rule_based_score(image_features, metadata)
            importance = self._rule_based_importance(image_features)

        suggestions = self._generate_suggestions(image_features, metadata, prob)

        return {
            "trend_score":        round(float(prob), 4),
            "label":              "trending" if prob > 0.5 else "not trending",
            "confidence":         round(abs(prob - 0.5) * 2, 4),
            "feature_importance": importance,
            "suggestions":        suggestions,
        }

    # ── Feature Vector Construction ───────────────────────────────────────────
    def _build_feature_vector(self, img_feats: dict, meta: dict) -> list:
        title = meta.get("title", "")
        subs  = max(meta.get("subscribers", 1000), 1)
        return [
            img_feats.get("brightness",          128.0),
            img_feats.get("contrast",             40.0),
            img_feats.get("saturation_mean",     100.0),
            img_feats.get("saturation_std",       30.0),
            img_feats.get("hue_mean",             90.0),
            img_feats.get("edge_density",          0.1),
            img_feats.get("face_count",              0),
            img_feats.get("has_face",                0),
            img_feats.get("red_ratio",             0.33),
            img_feats.get("green_ratio",           0.33),
            img_feats.get("blue_ratio",            0.33),
            img_feats.get("sharpness",            200.0),
            img_feats.get("color_diversity",       80.0),
            img_feats.get("text_region_density",   0.1),
            len(title),
            int(any(c.isdigit() for c in title)),
            int(sum(1 for w in title.split() if w.isupper()) > 0),
            meta.get("duration_minutes", 10.0),
            math.log10(subs),
        ]

    # ── Feature Importance ────────────────────────────────────────────────────
    def _get_feature_importance(self, feature_vec: list) -> dict:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            raw = clf.feature_importances_
        else:
            raw = np.ones(len(self.feature_names)) / len(self.feature_names)

        return {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(self.feature_names, raw),
                key=lambda x: -x[1]
            )
        }

    def _rule_based_score(self, img_feats: dict, meta: dict) -> float:
        score = (
            0.20 * (img_feats.get("brightness", 128) / 220) +
            0.15 * (img_feats.get("saturation_mean", 100) / 200) +
            0.15 * img_feats.get("has_face", 0) +
            0.10 * (img_feats.get("sharpness", 200) / 2000) +
            0.10 * (img_feats.get("color_diversity", 80) / 150) +
            0.10 * min(math.log10(max(meta.get("subscribers", 1000), 1)) / 7, 1.0)
        )
        return float(np.clip(score, 0, 1))

    def _rule_based_importance(self, img_feats: dict) -> dict:
        return {
            "brightness":      0.20, "saturation_mean": 0.15,
            "has_face":        0.15, "sharpness":       0.10,
            "color_diversity": 0.10, "edge_density":    0.08,
            "subscribers_log": 0.08, "contrast":        0.07,
            "title_length":    0.05, "duration_minutes":0.02,
        }

    # ── Suggestions ───────────────────────────────────────────────────────────
    def _generate_suggestions(self, img_feats: dict, meta: dict, score: float) -> list[str]:
        tips = []
        if img_feats.get("brightness", 128) < 100:
            tips.append("📸 Increase thumbnail brightness — darker thumbnails perform worse.")
        if img_feats.get("saturation_mean", 100) < 80:
            tips.append("🎨 Boost color saturation — vibrant thumbnails attract more clicks.")
        if img_feats.get("has_face", 0) == 0:
            tips.append("😊 Consider adding a face — human faces dramatically increase CTR.")
        if img_feats.get("sharpness", 200) < 100:
            tips.append("🔍 Improve image sharpness — blurry thumbnails signal low quality.")
        if img_feats.get("edge_density", 0.1) > 0.25:
            tips.append("✂️ Simplify the thumbnail — too many elements reduce focus.")
        if len(meta.get("title", "")) < 30:
            tips.append("📝 Expand your title — 50–70 character titles tend to perform best.")
        if score > 0.7:
            tips.append("✅ Strong trending potential! Publish during peak hours (2–4 PM local time).")
        if not tips:
            tips.append("💡 Thumbnail looks good overall. A/B test with a face vs no-face variant.")
        return tips

    # ── Training on Real Data ─────────────────────────────────────────────────
    def train_on_real_data(self, videos: list[dict],
                            image_features_list: list[dict]) -> dict:
        """
        Retrain on real collected data.
        `videos` must have 'views' key; top-50% by views = trending (label=1).

        Returns evaluation metrics dict.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed"}

        view_counts = [v.get("views", 0) for v in videos]
        median_views = np.median(view_counts)

        X, y = [], []
        for video, img_feats in zip(videos, image_features_list):
            meta = {
                "title":            video.get("title", ""),
                "subscribers":      video.get("subscribers", 1000),
                "duration_minutes": video.get("duration", 0) / 60,
            }
            X.append(self._build_feature_vector(img_feats, meta))
            y.append(int(video.get("views", 0) >= median_views))

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

        self.pipeline.fit(X_train, y_train)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        self._save()
        return {
            "auc":            round(auc, 3),
            "n_samples":      len(X),
            "n_trending":     int(y.sum()),
            "n_not_trending": int((1 - y).sum()),
        }

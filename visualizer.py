"""
modules/visualizer.py
──────────────────────
All Matplotlib/Seaborn charts for the Streamlit dashboard.

Charts:
  • Color histogram (BGR channels)
  • Edge map overlay
  • Face detection display
  • Feature radar chart
  • Region feature comparison (bar chart)
  • Engagement metrics comparison
  • Feature heatmap (region × feature)
  • Feature importance (horizontal bar)
  • Misleading score breakdown (pie/bar)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
from typing import Optional
import config

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f0f0f",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#333",
    "axes.labelcolor":   "#ccc",
    "xtick.color":       "#888",
    "ytick.color":       "#888",
    "text.color":        "#eee",
    "grid.color":        "#2a2a2a",
    "grid.linestyle":    "--",
    "font.family":       "monospace",
    "axes.titlecolor":   "#ff4444",
})

CHANNEL_COLORS = {"red": "#ff4444", "green": "#44ff88", "blue": "#4488ff"}
REGION_PALETTE = ["#ff4444", "#4488ff", "#44ff88", "#ffaa44", "#aa44ff",
                  "#ff88aa", "#88aaff", "#aaffaa", "#ffcc44", "#cc88ff"]


class Visualizer:

    # ── Color Histogram ───────────────────────────────────────────────────────
    def plot_color_histogram(self, img_bgr: np.ndarray,
                              bins: int = 64) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("Color Distribution per Channel", fontsize=13, color="#ff4444")

        channel_info = [
            (0, "Blue",  "#4488ff"),
            (1, "Green", "#44ff88"),
            (2, "Red",   "#ff4444"),
        ]
        for ax, (ch_idx, name, color) in zip(axes, channel_info):
            hist = cv2.calcHist([img_bgr], [ch_idx], None, [bins], [0, 256])
            hist = hist.flatten() / hist.sum()
            ax.plot(hist, color=color, linewidth=1.5)
            ax.fill_between(range(bins), hist, alpha=0.3, color=color)
            ax.set_title(name, color=color)
            ax.set_xlabel("Pixel Value (0–255)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ── Edge Map ──────────────────────────────────────────────────────────────
    def plot_edge_map(self, img_rgb: np.ndarray,
                       edges: np.ndarray) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Edge Density Analysis", fontsize=13, color="#ff4444")

        ax1.imshow(img_rgb)
        ax1.set_title("Original Thumbnail")
        ax1.axis("off")

        ax2.imshow(edges, cmap="hot")
        ax2.set_title(f"Canny Edge Map  |  Density = {np.sum(edges>0)/edges.size:.3f}")
        ax2.axis("off")

        fig.tight_layout()
        return fig

    # ── Face Detection ────────────────────────────────────────────────────────
    def plot_face_detection(self, img_rgb_annotated: np.ndarray,
                             face_data: list[dict]) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Face Detection Results", fontsize=13, color="#ff4444")

        ax1.imshow(img_rgb_annotated)
        ax1.set_title(f"{len(face_data)} Face(s) Detected")
        ax1.axis("off")

        # Area ratios bar
        if face_data:
            labels = [f"Face {i+1}" for i in range(len(face_data))]
            areas  = [f["area_ratio"] * 100 for f in face_data]
            colors = REGION_PALETTE[:len(face_data)]
            ax2.barh(labels, areas, color=colors)
            ax2.set_xlabel("Face Area (% of thumbnail)")
            ax2.set_title("Face Size Distribution")
            ax2.grid(True, axis="x", alpha=0.3)
            for i, v in enumerate(areas):
                ax2.text(v + 0.2, i, f"{v:.1f}%", va="center", color="#eee")
        else:
            ax2.text(0.5, 0.5, "No faces detected\n\nConsider adding a human\nface for higher CTR",
                     ha="center", va="center", fontsize=13, color="#888",
                     transform=ax2.transAxes)
            ax2.axis("off")

        fig.tight_layout()
        return fig

    # ── Feature Radar ─────────────────────────────────────────────────────────
    def plot_feature_radar(self, features: dict) -> plt.Figure:
        """Radar chart of normalised feature values."""
        keys = ["brightness", "contrast", "saturation_mean",
                "edge_density", "sharpness", "color_diversity"]
        norms = {
            "brightness": 255, "contrast": 100, "saturation_mean": 255,
            "edge_density": 0.3, "sharpness": 2000, "color_diversity": 200,
        }
        labels = ["Brightness", "Contrast", "Saturation",
                  "Edge Density", "Sharpness", "Color Diversity"]
        values = [min(features.get(k, 0) / norms[k], 1.0) for k in keys]
        values += values[:1]  # close polygon

        angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(6, 6))
        ax  = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#0f0f0f")

        ax.plot(angles, values, "o-", linewidth=2, color="#ff4444")
        ax.fill(angles, values, alpha=0.25, color="#ff4444")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9, color="#ccc")
        ax.set_ylim(0, 1)
        ax.set_title("Feature Profile", size=13, color="#ff4444", pad=20)
        ax.grid(color="#333", linestyle="--", alpha=0.5)

        fig.tight_layout()
        return fig

    # ── Region Feature Comparison ─────────────────────────────────────────────
    def plot_region_feature_comparison(self, comparison: dict,
                                        region_a: str, region_b: str) -> plt.Figure:
        features = list(comparison.keys())
        vals_a   = [comparison[f][region_a] for f in features]
        vals_b   = [comparison[f][region_b] for f in features]

        x     = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars_a = ax.bar(x - width/2, vals_a, width, label=region_a, color="#ff4444", alpha=0.85)
        bars_b = ax.bar(x + width/2, vals_b, width, label=region_b, color="#4488ff", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in features], rotation=0, ha="center")
        ax.set_title(f"Thumbnail Feature Comparison: {region_a} vs {region_b}",
                     fontsize=13, color="#ff4444")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        # Value labels
        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7, color="#ccc")

        fig.tight_layout()
        return fig

    # ── Engagement Comparison ─────────────────────────────────────────────────
    def plot_engagement_comparison(self, data_a: list[dict], data_b: list[dict],
                                    region_a: str, region_b: str) -> plt.Figure:
        def avg(lst, key): return np.mean([d.get(key, 0) for d in lst]) if lst else 0

        metrics = ["views", "likes", "comments"]
        a_vals  = [avg(data_a, m) for m in metrics]
        b_vals  = [avg(data_b, m) for m in metrics]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f"Engagement Metrics: {region_a} vs {region_b}",
                     fontsize=13, color="#ff4444")

        for ax, metric, va, vb in zip(axes, metrics, a_vals, b_vals):
            bars = ax.bar([region_a, region_b], [va, vb],
                          color=["#ff4444", "#4488ff"], width=0.5, alpha=0.85)
            ax.set_title(metric.capitalize(), color="#ccc")
            ax.set_ylabel(f"Avg {metric.capitalize()}")
            ax.grid(True, axis="y", alpha=0.3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                        f"{h:,.0f}", ha="center", va="bottom", fontsize=9, color="#eee")

        fig.tight_layout()
        return fig

    # ── Feature Heatmap ───────────────────────────────────────────────────────
    def plot_feature_heatmap(self, comparison: dict) -> plt.Figure:
        import matplotlib.colors as mcolors

        features = list(comparison.keys())
        regions  = list(next(iter(comparison.values())).keys())

        matrix = np.array([
            [comparison[f].get(r, 0) for r in regions]
            for f in features
        ])

        # Row-normalize for visual comparison
        row_max = matrix.max(axis=1, keepdims=True)
        matrix_norm = np.where(row_max > 0, matrix / row_max, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(regions) * 1.2), max(5, len(features) * 0.6)))
        im = ax.imshow(matrix_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=30, ha="right")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace("_", " ").title() for f in features])
        ax.set_title("Region × Feature Heatmap (row-normalised)", color="#ff4444", fontsize=13)

        for i in range(len(features)):
            for j in range(len(regions)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if matrix_norm[i,j] > 0.5 else "white")

        plt.colorbar(im, ax=ax, label="Normalised Value")
        fig.tight_layout()
        return fig

    # ── Feature Importance ────────────────────────────────────────────────────
    def plot_feature_importance(self, importance: dict) -> plt.Figure:
        items  = sorted(importance.items(), key=lambda x: x[1])[-15:]
        labels = [k.replace("_", " ").title() for k, _ in items]
        values = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [config.ACCENT_COLOR if v > np.median(values) else "#4488ff" for v in values]
        bars = ax.barh(labels, values, color=colors, alpha=0.85)

        ax.set_title("Feature Importance (Trend Prediction Model)", color="#ff4444", fontsize=13)
        ax.set_xlabel("Importance Score")
        ax.grid(True, axis="x", alpha=0.3)

        for bar, v in zip(bars, values):
            ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=8, color="#ccc")

        fig.tight_layout()
        return fig

    # ── Misleading Breakdown ──────────────────────────────────────────────────
    def plot_misleading_breakdown(self, scores: dict) -> plt.Figure:
        labels = [k.replace("_", " ").title() for k in scores]
        values = list(scores.values())
        colors = ["#ff4444" if v > 0.5 else "#ffaa44" if v > 0.3 else "#44cc88"
                  for v in values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Misleading Content Analysis Breakdown", color="#ff4444", fontsize=13)

        # Bar chart
        bars = ax1.barh(labels, values, color=colors, alpha=0.85)
        ax1.set_xlim(0, 1)
        ax1.axvline(0.5, color="#888", linestyle="--", alpha=0.6)
        ax1.set_xlabel("Score (0 = authentic, 1 = misleading)")
        ax1.grid(True, axis="x", alpha=0.3)
        for bar, v in zip(bars, values):
            ax1.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{v:.2f}", va="center", fontsize=9, color="#eee")

        # Gauge (pie)
        total = np.mean(values)
        wedge_sizes = [total, 1 - total]
        wedge_colors = ["#ff4444", "#1a1a2e"]
        ax2.pie(wedge_sizes, colors=wedge_colors,
                startangle=90, counterclock=False,
                wedgeprops={"width": 0.4, "edgecolor": "#333"})
        ax2.text(0, 0, f"{total:.0%}", ha="center", va="center",
                 fontsize=22, fontweight="bold", color="#ff4444")
        ax2.set_title("Overall Misleading Score", color="#ff4444")

        fig.tight_layout()
        return fig

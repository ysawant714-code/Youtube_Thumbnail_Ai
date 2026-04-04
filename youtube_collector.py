"""
modules/youtube_collector.py
────────────────────────────
Wraps the YouTube Data API v3 to fetch trending video data,
including thumbnails, metadata, and engagement metrics.

Usage:
    collector = YouTubeCollector(api_key="YOUR_KEY")
    videos    = collector.fetch_trending("IN", max_results=50)
    thumbnail = collector.download_thumbnail(video["thumbnail_url"])
"""

import os
import time
import json
import requests
import numpy as np
import cv2
from typing import Optional
import config


class YouTubeCollector:
    """Fetches YouTube trending data using the Data API v3."""

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    # YouTube video category IDs → human names
    CATEGORY_MAP = {
        "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
        "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
        "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
        "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
        "25": "News & Politics", "26": "Howto & Style", "27": "Education",
        "28": "Science & Technology", "29": "Nonprofits & Activism",
    }

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or config.YOUTUBE_API_KEY
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ── Core API call ─────────────────────────────────────────────────────────
    def _get(self, endpoint: str, params: dict) -> dict:
        """Generic GET to the YouTube API."""
        params["key"] = self.api_key
        resp = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Trending Videos ───────────────────────────────────────────────────────
    def fetch_trending(self, region_code: str = "US", max_results: int = 50) -> list[dict]:
        """
        Fetch currently trending YouTube videos for a given region.

        Returns a list of dicts with:
            video_id, title, channel, views, likes, comments,
            category, duration, thumbnail_url, region, published_at
        """
        if not self.api_key:
            print("[YouTubeCollector] No API key — returning empty list.")
            return []

        videos   = []
        page_token = None

        while len(videos) < max_results:
            batch = min(50, max_results - len(videos))
            params = {
                "part":            "snippet,statistics,contentDetails",
                "chart":           "mostPopular",
                "regionCode":      region_code,
                "maxResults":      batch,
                "videoCategoryId": "",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._get("videos", params)
            except requests.HTTPError as exc:
                print(f"[YouTubeCollector] HTTP error: {exc}")
                break

            for item in data.get("items", []):
                snippet     = item.get("snippet", {})
                stats       = item.get("statistics", {})
                content     = item.get("contentDetails", {})
                thumbnails  = snippet.get("thumbnails", {})

                # Best thumbnail quality available
                thumb_url = (
                    thumbnails.get("maxres", {}).get("url") or
                    thumbnails.get("standard", {}).get("url") or
                    thumbnails.get("high", {}).get("url") or
                    thumbnails.get("medium", {}).get("url") or ""
                )

                videos.append({
                    "video_id":       item.get("id", ""),
                    "title":          snippet.get("title", ""),
                    "channel":        snippet.get("channelTitle", ""),
                    "description":    snippet.get("description", "")[:500],
                    "tags":           snippet.get("tags", []),
                    "category_id":    snippet.get("categoryId", ""),
                    "category":       self.CATEGORY_MAP.get(snippet.get("categoryId", ""), "Unknown"),
                    "published_at":   snippet.get("publishedAt", ""),
                    "views":          int(stats.get("viewCount", 0)),
                    "likes":          int(stats.get("likeCount", 0)),
                    "comments":       int(stats.get("commentCount", 0)),
                    "duration":       self._parse_duration(content.get("duration", "PT0S")),
                    "thumbnail_url":  thumb_url,
                    "region":         region_code,
                    "url":            f"https://www.youtube.com/watch?v={item.get('id','')}",
                })

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.2)  # be polite to the quota

        return videos

    # ── Thumbnail Download ────────────────────────────────────────────────────
    def download_thumbnail(self, url: str) -> Optional[np.ndarray]:
        """
        Download a YouTube thumbnail and return as an OpenCV BGR image.
        Returns None on failure.
        """
        if not url:
            return None
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as exc:
            print(f"[YouTubeCollector] Failed to download thumbnail: {exc}")
            return None

    def download_thumbnails_batch(self, videos: list[dict],
                                  save_dir: str = "") -> dict[str, np.ndarray]:
        """
        Download thumbnails for a list of video dicts.
        Optionally saves JPEGs to `save_dir`.
        Returns {video_id: bgr_image}.
        """
        images = {}
        for video in videos:
            vid = video.get("video_id", "")
            img = self.download_thumbnail(video.get("thumbnail_url", ""))
            if img is not None:
                images[vid] = img
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    path = os.path.join(save_dir, f"{vid}.jpg")
                    cv2.imwrite(path, img)
            time.sleep(0.05)
        return images

    # ── Channel Info ──────────────────────────────────────────────────────────
    def fetch_channel_stats(self, channel_id: str) -> dict:
        """Fetch subscriber count and other stats for a channel."""
        try:
            data = self._get("channels", {
                "part": "statistics",
                "id":   channel_id,
            })
            stats = data["items"][0]["statistics"]
            return {
                "subscribers":  int(stats.get("subscriberCount", 0)),
                "total_views":  int(stats.get("viewCount", 0)),
                "video_count":  int(stats.get("videoCount", 0)),
            }
        except Exception:
            return {"subscribers": 0, "total_views": 0, "video_count": 0}

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_duration(iso_duration: str) -> float:
        """Convert ISO 8601 duration (PT4M13S) to total seconds."""
        import re
        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        m = re.match(pattern, iso_duration)
        if not m:
            return 0.0
        hours   = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = int(m.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def save_to_json(data: list[dict], path: str) -> None:
        """Save collected video data to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[YouTubeCollector] Saved {len(data)} videos → {path}")

    @staticmethod
    def load_from_json(path: str) -> list[dict]:
        """Load previously saved video data from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

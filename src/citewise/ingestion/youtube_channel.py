"""Discover recent public YouTube channel videos with yt-dlp."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urlparse


CHANNEL_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}
CHANNEL_PATH_PATTERN = re.compile(r"^/(?:@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)(?:/videos)?/?$")


class YouTubeChannelError(RuntimeError):
    """The supplied channel could not be read."""


class InvalidYouTubeChannelError(YouTubeChannelError):
    """The supplied value is not a supported channel reference."""


@dataclass(frozen=True)
class YouTubeChannelVideo:
    """A lightweight video reference returned by a channel listing."""

    video_id: str
    title: str
    url: str


@dataclass(frozen=True)
class YouTubeChannel:
    """A public channel and its most recent video references."""

    title: str
    url: str
    videos: tuple[YouTubeChannelVideo, ...]


class YouTubeChannelClient:
    """List recent public channel videos without requiring a YouTube Data API key."""

    def __init__(self, youtube_dl_factory: Any | None = None) -> None:
        self.youtube_dl_factory = youtube_dl_factory

    def fetch_channel(self, channel: str, limit: int = 10) -> YouTubeChannel:
        """Return up to ``limit`` recent videos, newest first, from a channel reference."""
        if limit < 1:
            raise InvalidYouTubeChannelError("--limit must be at least 1.")
        channel_url = self.normalize_channel_url(channel)
        factory = self.youtube_dl_factory or self._youtube_dl_factory()
        options = {
            "extract_flat": True,
            "playlistend": limit,
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        try:
            with factory(options) as downloader:
                info = downloader.extract_info(channel_url, download=False)
        except Exception as error:
            raise YouTubeChannelError(
                "Could not retrieve this YouTube channel's video list. Check the channel reference "
                "and your connection, then try again."
            ) from error

        entries = info.get("entries") if isinstance(info, dict) else None
        if not entries:
            raise YouTubeChannelError("This channel has no public videos to ingest.")
        videos = tuple(
            video
            for entry in entries
            if (video := self._video_from_entry(entry)) is not None
        )
        if not videos:
            raise YouTubeChannelError("This channel has no usable public video references.")
        return YouTubeChannel(
            title=str(info.get("title") or info.get("uploader") or channel_url),
            url=channel_url,
            videos=videos[:limit],
        )

    @staticmethod
    def is_channel_reference(value: str) -> bool:
        candidate = value.strip()
        if candidate.startswith("@") and len(candidate) > 1:
            return True
        parsed = urlparse(candidate)
        return (
            parsed.netloc.lower().split(":")[0] in CHANNEL_HOSTS
            and bool(CHANNEL_PATH_PATTERN.fullmatch(parsed.path))
        )

    @classmethod
    def normalize_channel_url(cls, value: str) -> str:
        """Accept a bare handle or a canonical channel/handle URL."""
        candidate = value.strip()
        if candidate.startswith("@") and len(candidate) > 1:
            return f"https://www.youtube.com/{candidate}/videos"
        if not cls.is_channel_reference(candidate):
            raise InvalidYouTubeChannelError(
                "Enter a YouTube channel URL or handle, for example @GoogleDevelopers or "
                "https://www.youtube.com/@GoogleDevelopers."
            )
        parsed = urlparse(candidate)
        base_url = f"https://www.youtube.com{parsed.path.rstrip('/')}"
        return base_url if base_url.endswith("/videos") else f"{base_url}/videos"

    @staticmethod
    def _video_from_entry(entry: object) -> YouTubeChannelVideo | None:
        if not isinstance(entry, dict):
            return None
        video_id = str(entry.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            return None
        title = str(entry.get("title") or video_id).strip()
        return YouTubeChannelVideo(
            video_id=video_id,
            title=title,
            url=f"https://www.youtube.com/watch?v={video_id}",
        )

    @staticmethod
    def _youtube_dl_factory() -> Any:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as error:
            raise YouTubeChannelError(
                "Channel ingestion requires yt-dlp. Install the project's dependencies and try again."
            ) from error
        return YoutubeDL

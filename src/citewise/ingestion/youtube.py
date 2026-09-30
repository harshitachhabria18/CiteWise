"""Fetch a single public YouTube video's caption transcript."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApiException,
    YouTubeRequestFailed,
)


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
OEMBED_URL = "https://www.youtube.com/oembed"


class YouTubeIngestionError(RuntimeError):
    """Base error for a readable YouTube ingestion failure."""


class InvalidYouTubeVideoError(YouTubeIngestionError):
    """The supplied URL or ID cannot identify a YouTube video."""


class TranscriptUnavailableError(YouTubeIngestionError):
    """The video has no captions that the transcript service can retrieve."""


class VideoUnavailableError(YouTubeIngestionError):
    """The video is private, unavailable, or cannot be played publicly."""


class RequestedLanguageUnavailableError(TranscriptUnavailableError):
    """The video has captions, but not in the requested language."""


@dataclass(frozen=True)
class YouTubeTranscriptSegment:
    """One caption segment with its position in the video."""

    text: str
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class YouTubeVideo:
    """A public video and its normalized caption segments."""

    video_id: str
    title: str
    url: str
    segments: tuple[YouTubeTranscriptSegment, ...]
    caption_language: str = ""
    caption_language_name: str = ""
    caption_selection_notice: str | None = None


class YouTubeClient:
    """Fetch one public video's captions in a requested language without an API key."""

    def __init__(self, timeout_seconds: int = 15, session: requests.Session | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.transcript_api: Any = YouTubeTranscriptApi(http_client=self.session)

    def fetch_video(self, video: str, language: str | None = None) -> YouTubeVideo:
        """Return a requested-language transcript, or automatically select a useful track."""
        video_id = self.extract_video_id(video)
        url = self.video_url(video_id)
        requested_language = self._normalize_language(language) if language else None
        try:
            transcript_list = self.transcript_api.list(video_id)
            available_transcripts = list(transcript_list)
        except TranscriptsDisabled as error:
            raise TranscriptUnavailableError(
                "This video has no captions or transcript available in any language."
            ) from error
        except (VideoUnavailable, VideoUnplayable, AgeRestricted) as error:
            raise VideoUnavailableError(
                "This video is private, unavailable, age-restricted, or cannot be played publicly."
            ) from error
        except (RequestBlocked, IpBlocked, YouTubeRequestFailed, requests.RequestException) as error:
            raise YouTubeIngestionError(
                "Could not retrieve the list of YouTube captions. Check your connection or "
                "whether YouTube is blocking this request, then try again."
            ) from error
        except YouTubeTranscriptApiException as error:
            raise YouTubeIngestionError(
                "YouTube could not provide this video's transcript. Please try another video."
            ) from error

        if not available_transcripts:
            raise TranscriptUnavailableError(
                "This video has no captions or transcript available in any language."
            )
        available_languages = self._available_languages(available_transcripts)
        selection_notice: str | None = None
        if requested_language:
            try:
                selected_transcript = transcript_list.find_transcript([requested_language])
            except NoTranscriptFound as error:
                raise RequestedLanguageUnavailableError(
                    f"Captions are available, but not in the requested language '{requested_language}'. "
                    f"Available languages: {available_languages}."
                ) from error
        else:
            english_tracks = [
                transcript
                for transcript in available_transcripts
                if str(getattr(transcript, "language_code", "")).lower() == "en"
                or str(getattr(transcript, "language_code", "")).lower().startswith("en-")
            ]
            selected_transcript = english_tracks[0] if english_tracks else available_transcripts[0]
            if not english_tracks:
                selection_notice = (
                    "English captions are not available; using "
                    f"{self._track_label(selected_transcript)} captions instead."
                )

        try:
            transcript = selected_transcript.fetch()
        except (RequestBlocked, IpBlocked, YouTubeRequestFailed, requests.RequestException) as error:
            raise YouTubeIngestionError(
                "Could not download the selected YouTube captions. Check your connection or "
                "whether YouTube is blocking this request, then try again."
            ) from error
        except YouTubeTranscriptApiException as error:
            raise YouTubeIngestionError(
                "YouTube could not download the selected transcript. Please try another video."
            ) from error

        segments = tuple(
            YouTubeTranscriptSegment(
                text=str(snippet.text).strip(),
                start_seconds=float(snippet.start),
                duration_seconds=float(snippet.duration),
            )
            for snippet in transcript
            if str(snippet.text).strip()
        )
        if not segments:
            raise TranscriptUnavailableError("This video has an empty transcript.")

        return YouTubeVideo(
            video_id=video_id,
            title=self._fetch_title(url),
            url=url,
            segments=segments,
            caption_language=str(getattr(selected_transcript, "language_code", "")),
            caption_language_name=str(getattr(selected_transcript, "language", "")),
            caption_selection_notice=selection_notice,
        )

    @staticmethod
    def extract_video_id(video: str) -> str:
        """Accept a canonical URL, supported share URL, or bare 11-character video ID."""
        candidate = video.strip()
        if VIDEO_ID_PATTERN.fullmatch(candidate):
            return candidate

        parsed = urlparse(candidate)
        host = parsed.netloc.lower().split(":")[0]
        path_parts = [part for part in parsed.path.split("/") if part]
        video_id = ""
        if host == "youtu.be" and path_parts:
            video_id = path_parts[0]
        elif host in YOUTUBE_HOSTS:
            if parsed.path == "/watch":
                video_id = parse_qs(parsed.query).get("v", [""])[0]
            elif len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
                video_id = path_parts[1]
        if not VIDEO_ID_PATTERN.fullmatch(video_id):
            raise InvalidYouTubeVideoError(
                "Enter a YouTube video URL or 11-character video ID (for example, "
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ)."
            )
        return video_id

    @staticmethod
    def video_url(video_id: str) -> str:
        return f"https://www.youtube.com/watch?v={video_id}"

    @staticmethod
    def _normalize_language(language: str) -> str:
        normalized = language.strip().lower()
        if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]+)*", normalized):
            raise InvalidYouTubeVideoError(
                "Enter a caption language code such as 'en', 'hi', or 'pt-br'."
            )
        return normalized

    @staticmethod
    def _available_languages(transcripts: list[Any]) -> str:
        """Format track names and codes without hiding manual versus generated captions."""
        return "; ".join(YouTubeClient._track_label(transcript) for transcript in transcripts)

    @staticmethod
    def _track_label(transcript: Any) -> str:
        language = str(getattr(transcript, "language", "Unknown"))
        code = str(getattr(transcript, "language_code", "unknown"))
        kind = "auto-generated" if getattr(transcript, "is_generated", False) else "manual"
        return f"{language} ({code}, {kind})"

    def _fetch_title(self, video_url: str) -> str:
        """Use YouTube's public oEmbed endpoint for the exact canonical video URL."""
        try:
            response = self.session.get(
                OEMBED_URL,
                params={"url": video_url, "format": "json"},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise YouTubeIngestionError(
                "Could not reach YouTube for the video title. Check your connection and try again."
            ) from error
        if not response.ok:
            raise VideoUnavailableError(
                "YouTube could not retrieve public details for this video. It may be private or unavailable."
            )
        try:
            title = str(response.json().get("title", "")).strip()
        except (ValueError, AttributeError) as error:
            raise YouTubeIngestionError("YouTube returned an unexpected video-details response.") from error
        if not title:
            raise YouTubeIngestionError("YouTube did not return a title for this video.")
        return title

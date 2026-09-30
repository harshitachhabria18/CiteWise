from typing import Any

import pytest

import citewise.services.youtube_pipeline as pipeline_module
from citewise.ingestion.youtube import TranscriptUnavailableError, YouTubeTranscriptSegment, YouTubeVideo
from citewise.ingestion.youtube_channel import (
    InvalidYouTubeChannelError,
    YouTubeChannel,
    YouTubeChannelClient,
    YouTubeChannelVideo,
)
from citewise.processing.youtube_chunking import build_youtube_documents
from citewise.services.youtube_pipeline import YouTubeIngestionReport, ingest_youtube_channel


class FakeDownloader:
    def __init__(self, _options: dict[str, Any]) -> None:
        pass

    def __enter__(self) -> "FakeDownloader":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, Any]:
        assert url == "https://www.youtube.com/@smallchannel/videos"
        assert not download
        return {
            "title": "Small Channel",
            "entries": [
                {"id": "dQw4w9WgXcQ", "title": "Newest video"},
                {"id": "9bZkp7q19f0", "title": "Older video"},
            ],
        }


def test_channel_client_accepts_handle_and_returns_video_references() -> None:
    channel = YouTubeChannelClient(youtube_dl_factory=FakeDownloader).fetch_channel("@smallchannel", limit=5)

    assert channel.title == "Small Channel"
    assert [video.video_id for video in channel.videos] == ["dQw4w9WgXcQ", "9bZkp7q19f0"]
    assert channel.videos[0].url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_invalid_channel_reference_has_a_clear_error() -> None:
    with pytest.raises(InvalidYouTubeChannelError, match="channel URL or handle"):
        YouTubeChannelClient.normalize_channel_url("https://example.com/not-youtube")


def test_channel_ingestion_reuses_single_video_ingestion_and_skips_missing_transcripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = YouTubeChannel(
        title="Small Channel",
        url="https://www.youtube.com/@smallchannel/videos",
        videos=(
            YouTubeChannelVideo("dQw4w9WgXcQ", "Captioned", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            YouTubeChannelVideo("9bZkp7q19f0", "No captions", "https://www.youtube.com/watch?v=9bZkp7q19f0"),
        ),
    )

    class FakeChannelClient:
        def fetch_channel(self, _input: str, limit: int) -> YouTubeChannel:
            assert limit == 5
            return channel

    video = YouTubeVideo("dQw4w9WgXcQ", "Captioned", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", ())
    expected_report = YouTubeIngestionReport(video, stored_chunks=3, source_counts={"youtube_transcript": 3})

    def fake_ingest(video_id: str, language: str | None = None) -> YouTubeIngestionReport:
        assert language == "en"
        if video_id == "9bZkp7q19f0":
            raise TranscriptUnavailableError("no captions")
        return expected_report

    monkeypatch.setattr(pipeline_module, "YouTubeChannelClient", FakeChannelClient)
    monkeypatch.setattr(pipeline_module, "ingest_youtube_video", fake_ingest)

    report = ingest_youtube_channel("@smallchannel", limit=5, language="en")

    assert report.stored_chunks == 3
    assert report.ingested_videos == (expected_report,)
    assert report.skipped_videos == ("Skipping No captions: no transcript available.",)


def test_chunks_from_multiple_channel_videos_keep_distinct_video_metadata() -> None:
    first_video = YouTubeVideo(
        "dQw4w9WgXcQ",
        "First",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        (YouTubeTranscriptSegment("First transcript.", 1.0, 1.0),),
    )
    second_video = YouTubeVideo(
        "9bZkp7q19f0",
        "Second",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        (YouTubeTranscriptSegment("Second transcript.", 2.0, 1.0),),
    )

    documents = [*build_youtube_documents(first_video), *build_youtube_documents(second_video)]

    assert {document.metadata["video_id"] for document in documents} == {"dQw4w9WgXcQ", "9bZkp7q19f0"}
    assert "YouTube: First" in documents[0].metadata["citation"]
    assert "YouTube: Second" in documents[1].metadata["citation"]

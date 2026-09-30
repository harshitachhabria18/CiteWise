from dataclasses import dataclass
from typing import Any

import pytest

from citewise.ingestion.youtube import (
    InvalidYouTubeVideoError,
    RequestedLanguageUnavailableError,
    YouTubeClient,
    YouTubeTranscriptSegment,
    YouTubeVideo,
)
from citewise.processing.youtube_chunking import build_youtube_documents, format_timestamp
from citewise.services.youtube_pipeline import (
    YOUTUBE_LIST_RETRIEVAL_K,
    retrieve_youtube_documents,
)
from langchain_core.documents import Document


@dataclass
class FakeSnippet:
    text: str
    start: float
    duration: float


class FakeTranscriptApi:
    def list(self, video_id: str) -> "FakeTranscriptList":
        assert video_id == "dQw4w9WgXcQ"
        return FakeTranscriptList()


class FakeTrack:
    language = "English"
    language_code = "en"
    is_generated = False

    def fetch(self) -> list[FakeSnippet]:
        return [
            FakeSnippet("The first caption line.", 12.4, 2.0),
            FakeSnippet("The second caption line.", 15.0, 2.0),
        ]


class FakeTranscriptList:
    def __iter__(self) -> Any:
        return iter([FakeTrack()])

    def find_transcript(self, languages: list[str]) -> FakeTrack:
        assert languages == ["en"]
        return FakeTrack()


class HindiOnlyTranscriptList:
    def __iter__(self) -> Any:
        return iter([self])

    language = "Hindi"
    language_code = "hi"
    is_generated = True

    def find_transcript(self, languages: list[str]) -> Any:
        from youtube_transcript_api._errors import NoTranscriptFound

        raise NoTranscriptFound("dQw4w9WgXcQ", languages, self)

    def fetch(self) -> list[FakeSnippet]:
        return [FakeSnippet("Hindi caption line.", 12.4, 2.0)]


class HindiOnlyTranscriptApi:
    def list(self, video_id: str) -> HindiOnlyTranscriptList:
        return HindiOnlyTranscriptList()


class FakeResponse:
    ok = True

    def json(self) -> dict[str, str]:
        return {"title": "Caption Test Video"}


class FakeSession:
    headers: dict[str, str] = {}

    def __init__(self) -> None:
        self.request_url = ""
        self.request_params: dict[str, str] = {}

    def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.request_url = str(args[0])
        self.request_params = kwargs["params"]
        return FakeResponse()


def test_fetch_video_normalizes_url_and_keeps_caption_timestamps() -> None:
    session = FakeSession()
    client = YouTubeClient(session=session)  # type: ignore[arg-type]
    client.transcript_api = FakeTranscriptApi()

    video = client.fetch_video("https://youtu.be/dQw4w9WgXcQ?feature=share")

    assert video.video_id == "dQw4w9WgXcQ"
    assert video.title == "Caption Test Video"
    assert video.segments[0].start_seconds == 12.4
    assert session.request_params["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_invalid_video_input_has_a_clear_error() -> None:
    with pytest.raises(InvalidYouTubeVideoError, match="YouTube video URL or 11-character video ID"):
        YouTubeClient.extract_video_id("not-a-valid-video")


def test_unavailable_requested_language_lists_available_caption_tracks() -> None:
    client = YouTubeClient(session=FakeSession())  # type: ignore[arg-type]
    client.transcript_api = HindiOnlyTranscriptApi()

    with pytest.raises(RequestedLanguageUnavailableError, match=r"Available languages: Hindi \(hi, auto-generated\)"):
        client.fetch_video("dQw4w9WgXcQ", language="en")


def test_unspecified_language_prefers_an_available_non_english_track() -> None:
    client = YouTubeClient(session=FakeSession())  # type: ignore[arg-type]
    client.transcript_api = HindiOnlyTranscriptApi()

    video = client.fetch_video("dQw4w9WgXcQ")

    assert video.caption_language == "hi"
    assert video.caption_selection_notice == (
        "English captions are not available; using Hindi (hi, auto-generated) captions instead."
    )


def test_youtube_chunks_cite_title_and_start_timestamp() -> None:
    video = YouTubeVideo(
        video_id="dQw4w9WgXcQ",
        title="Caption Test Video",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        segments=(
            YouTubeTranscriptSegment("First idea in the video.", 90.0, 3.0),
            YouTubeTranscriptSegment("Second idea in the video.", 94.0, 3.0),
        ),
    )

    documents = build_youtube_documents(video)

    assert len(documents) == 1
    assert documents[0].metadata["timestamp"] == "1:30"
    assert documents[0].metadata["timestamp_url"].endswith("&t=90s")
    assert documents[0].metadata["citation"] == (
        "YouTube: Caption Test Video @ 1:30 "
        "(https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s)"
    )
    assert format_timestamp(3661) == "1:01:01"


def test_range_list_retrieval_uses_a_wider_pool_and_promotes_exact_range_chunks() -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.k: int | None = None
            self.metadata_filter: dict[str, object] | None = None

        def similarity_search(
            self, _question: str, k: int, metadata_filter: dict[str, object]
        ) -> list[Document]:
            self.k = k
            self.metadata_filter = metadata_filter
            return [
                Document(page_content="Phones under ₹20k and above ₹30k."),
                Document(page_content="The ₹20k-₹30k range includes Phone A and Phone B."),
            ]

    store = FakeStore()

    documents = retrieve_youtube_documents(
        store, "What are all phones between ₹20k and ₹30k?", ["dQw4w9WgXcQ"]
    )

    assert store.k == YOUTUBE_LIST_RETRIEVAL_K
    assert store.metadata_filter == {"video_id": {"$in": ["dQw4w9WgXcQ"]}}
    assert documents[0].page_content.startswith("The ₹20k-₹30k")

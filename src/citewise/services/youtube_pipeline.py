"""Run the single-video or channel YouTube transcript -> RAG workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re

from citewise.ingestion.youtube import (
    TranscriptUnavailableError,
    YouTubeClient,
    YouTubeIngestionError,
    YouTubeVideo,
)
from citewise.ingestion.youtube_channel import (
    YouTubeChannelClient,
    YouTubeChannelError,
)
from citewise.processing.youtube_chunking import build_youtube_documents, youtube_document_ids
from citewise.retrieval.github_answer import AnswerGenerationError, generate_grounded_answer
from citewise.storage.chroma_store import LocalChromaStore


YOUTUBE_COLLECTION = "youtube_content"
YOUTUBE_RETRIEVAL_K = 20
YOUTUBE_LIST_RETRIEVAL_K = 40
YOUTUBE_CHANNEL_DEFAULT_LIMIT = 10

_LIST_ENUMERATION_TERMS = re.compile(
    r"\b(?:all|list|enumerate|which|name|show|give)\b", re.IGNORECASE
)
_RANGE_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand)?\s*"
    r"(?:-|–|—|to|and)\s*(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class YouTubeIngestionReport:
    video: YouTubeVideo
    stored_chunks: int
    source_counts: dict[str, int]


@dataclass(frozen=True)
class YouTubeChannelIngestionReport:
    channel_title: str
    channel_url: str
    stored_chunks: int
    ingested_videos: tuple[YouTubeIngestionReport, ...]
    skipped_videos: tuple[str, ...]


def ingest_youtube_video(
    video_input: str, language: str | None = None
) -> YouTubeIngestionReport:
    """Fetch, chunk, and persist one video's caption transcript."""
    video = YouTubeClient().fetch_video(video_input, language=language)
    documents = build_youtube_documents(video)
    store = LocalChromaStore(collection_name=YOUTUBE_COLLECTION)
    store.add_documents(documents, youtube_document_ids(documents))
    return YouTubeIngestionReport(
        video=video,
        stored_chunks=len(documents),
        source_counts=store.count_by_source_type(metadata_filter={"video_id": video.video_id}),
    )


def ingest_youtube_channel(
    channel_input: str, limit: int = YOUTUBE_CHANNEL_DEFAULT_LIMIT, language: str | None = None
) -> YouTubeChannelIngestionReport:
    """Discover a channel's recent videos and reuse single-video ingestion for each one."""
    channel = YouTubeChannelClient().fetch_channel(channel_input, limit=limit)
    ingested_videos: list[YouTubeIngestionReport] = []
    skipped_videos: list[str] = []
    for channel_video in channel.videos:
        try:
            ingested_videos.append(ingest_youtube_video(channel_video.video_id, language=language))
        except TranscriptUnavailableError:
            skipped_videos.append(f"Skipping {channel_video.title}: no transcript available.")
        except YouTubeIngestionError as error:
            skipped_videos.append(f"Skipping {channel_video.title}: {error}")

    return YouTubeChannelIngestionReport(
        channel_title=channel.title,
        channel_url=channel.url,
        stored_chunks=sum(report.stored_chunks for report in ingested_videos),
        ingested_videos=tuple(ingested_videos),
        skipped_videos=tuple(skipped_videos),
    )


def retrieve_youtube_documents(
    store: LocalChromaStore, question: str, video_ids: list[str]
) -> list:
    """Retrieve extra candidates for range-list questions, then put exact-range chunks first.

    The answer prompt remains responsible for excluding unsupported items.  Reordering
    makes the wider candidate set useful despite the answer context's fixed character
    budget: chunks that explicitly mention the requested range reach Groq before less
    targeted semantic matches.
    """
    requested_range = _requested_range(question)
    retrieval_k = YOUTUBE_LIST_RETRIEVAL_K if requested_range else YOUTUBE_RETRIEVAL_K
    documents = store.similarity_search(
        question,
        k=retrieval_k,
        metadata_filter={"video_id": {"$in": video_ids}},
    )
    if not requested_range:
        return documents

    range_documents = [
        document
        for document in documents
        if requested_range in _ranges_in_text(document.page_content)
    ]
    other_documents = [document for document in documents if document not in range_documents]
    return [*range_documents, *other_documents]


def _requested_range(question: str) -> tuple[float, float] | None:
    """Return a numeric question range only for an enumeration-style request."""
    if not _LIST_ENUMERATION_TERMS.search(question):
        return None
    match = _RANGE_PATTERN.search(question)
    if not match:
        return None
    lower = _range_number(match.group(1), match.group(2))
    upper = _range_number(match.group(3), match.group(4))
    return tuple(sorted((lower, upper)))


def _ranges_in_text(text: str) -> set[tuple[float, float]]:
    """Extract explicit numeric ranges from a transcript chunk."""
    return {
        tuple(sorted((_range_number(match.group(1), match.group(2)), _range_number(match.group(3), match.group(4)))))
        for match in _RANGE_PATTERN.finditer(text)
    }


def _range_number(value: str, suffix: str | None) -> float:
    number = float(value.replace(",", ""))
    return number * 1_000 if suffix and suffix.casefold() in {"k", "thousand"} else number


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a YouTube video or channel's captions into local Chroma.")
    parser.add_argument("input", help="YouTube video URL/ID, channel URL, or @handle")
    parser.add_argument(
        "--limit",
        type=int,
        default=YOUTUBE_CHANNEL_DEFAULT_LIMIT,
        help=f"Maximum recent channel videos, default: {YOUTUBE_CHANNEL_DEFAULT_LIMIT}",
    )
    parser.add_argument("--query", help="Optional question to ask after ingestion")
    parser.add_argument(
        "--language",
        help="Optional caption language override; otherwise English is preferred, then another available track",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=YOUTUBE_RETRIEVAL_K,
        help=f"Number of chunks to retrieve for a query, default: {YOUTUBE_RETRIEVAL_K}",
    )
    args = parser.parse_args()

    is_channel = YouTubeChannelClient.is_channel_reference(args.input)
    try:
        if is_channel:
            channel_report = ingest_youtube_channel(args.input, limit=args.limit, language=args.language)
            video_ids = [report.video.video_id for report in channel_report.ingested_videos]
            print(
                f"Stored {channel_report.stored_chunks} chunks from "
                f"{len(channel_report.ingested_videos)} videos in: {channel_report.channel_title}"
            )
            for message in channel_report.skipped_videos:
                print(message)
            print(f"Chunks by type: {{'youtube_transcript': {channel_report.stored_chunks}}}")
        else:
            report = ingest_youtube_video(args.input, language=args.language)
            video_ids = [report.video.video_id]
            print(f"Stored {report.stored_chunks} chunks from: {report.video.title}")
            if report.video.caption_selection_notice:
                print(f"Notice: {report.video.caption_selection_notice}")
            print(f"Chunks by type: {report.source_counts}")
    except (YouTubeIngestionError, YouTubeChannelError) as error:
        print(f"Ingestion failed: {error}")
        raise SystemExit(1) from error

    if args.query and video_ids:
        retrieved_documents = LocalChromaStore(collection_name=YOUTUBE_COLLECTION).similarity_search(
            args.query,
            k=max(1, args.k),
            metadata_filter={"video_id": {"$in": video_ids}},
        )
        print("\nGrounded answer:")
        try:
            print(generate_grounded_answer(args.query, retrieved_documents))
        except AnswerGenerationError as error:
            print(f"Answer generation unavailable: {error}")

        print("\nTop matching chunks:")
        for document in retrieved_documents:
            print(f"- {document.metadata['citation']}")
    elif args.query:
        print("\nNo videos with stored transcripts are available to query.")


if __name__ == "__main__":
    _main()

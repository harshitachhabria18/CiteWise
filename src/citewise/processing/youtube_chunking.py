"""Chunk timestamped YouTube captions into citation-ready documents."""

from __future__ import annotations

from bisect import bisect_right
from hashlib import sha256
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citewise.ingestion.youtube import YouTubeVideo


# Spoken captions need enough surrounding dialogue for a complete thought, while still
# keeping a citation close to the moment a viewer can verify in the video.
TRANSCRIPT_CHUNK_SIZE = 1000
TRANSCRIPT_CHUNK_OVERLAP = 150


def build_youtube_documents(video: YouTubeVideo) -> list[Document]:
    """Split caption text and attach the timestamp of each chunk's first segment."""
    full_text, segment_offsets, segment_starts = _transcript_text_with_offsets(video)
    if not full_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TRANSCRIPT_CHUNK_SIZE,
        chunk_overlap=TRANSCRIPT_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)
    documents: list[Document] = []
    search_from = 0
    for chunk_index, chunk in enumerate(chunks):
        chunk_offset = full_text.find(chunk, search_from)
        if chunk_offset < 0:
            chunk_offset = full_text.find(chunk)
        search_from = max(0, chunk_offset + len(chunk) - TRANSCRIPT_CHUNK_OVERLAP)
        segment_index = max(0, bisect_right(segment_offsets, chunk_offset) - 1)
        start_seconds = segment_starts[segment_index]
        timestamp = format_timestamp(start_seconds)
        timestamp_url = f"{video.url}&t={int(start_seconds)}s"
        citation = f"YouTube: {video.title} @ {timestamp} ({timestamp_url})"
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_type": "youtube_transcript",
                    "video_id": video.video_id,
                    "video_title": video.title,
                    "video_url": video.url,
                    "timestamp_seconds": start_seconds,
                    "timestamp": timestamp,
                    "timestamp_url": timestamp_url,
                    "chunk_index": chunk_index,
                    "citation": citation,
                },
            )
        )
    return documents


def youtube_document_ids(documents: Iterable[Document]) -> list[str]:
    """Return stable IDs so re-ingesting a video updates its existing chunks."""
    return [
        sha256(
            f"{document.metadata.get('video_id')}:{document.metadata.get('chunk_index')}:"
            f"{document.page_content}".encode()
        ).hexdigest()
        for document in documents
    ]


def format_timestamp(seconds: float) -> str:
    """Format a video offset as H:MM:SS or M:SS for readable citations."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes}:{remaining_seconds:02d}"


def _transcript_text_with_offsets(video: YouTubeVideo) -> tuple[str, list[int], list[float]]:
    text_parts: list[str] = []
    offsets: list[int] = []
    starts: list[float] = []
    current_offset = 0
    for segment in video.segments:
        text = segment.text.strip()
        if not text:
            continue
        if text_parts:
            current_offset += 1  # Space inserted between caption segments.
        offsets.append(current_offset)
        starts.append(segment.start_seconds)
        text_parts.append(text)
        current_offset += len(text)
    return " ".join(text_parts), offsets, starts

"""Run the Hacker News search -> discussion -> RAG workflow."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from citewise.ingestion.hackernews import HackerNewsClient, HackerNewsIngestionError, HackerNewsStory
from citewise.processing.hackernews_chunking import build_hackernews_documents, hackernews_document_ids
from citewise.retrieval.github_answer import AnswerGenerationError, generate_grounded_answer
from citewise.storage.chroma_store import LocalChromaStore


HACKERNEWS_COLLECTION = "hackernews_content"
HACKERNEWS_DEFAULT_LIMIT = 10


@dataclass(frozen=True)
class HackerNewsIngestionReport:
    topic: str
    stories: tuple[HackerNewsStory, ...]
    stored_chunks: int
    source_counts: dict[str, int]


def ingest_hackernews_topic(topic: str, limit: int = HACKERNEWS_DEFAULT_LIMIT) -> HackerNewsIngestionReport:
    """Search a topic, persist story/discussion chunks, and retain citation metadata."""
    stories = HackerNewsClient().search_stories(topic, limit=limit)
    documents = build_hackernews_documents(stories)
    store = LocalChromaStore(collection_name=HACKERNEWS_COLLECTION)
    store.add_documents(documents, hackernews_document_ids(documents))
    return HackerNewsIngestionReport(
        topic=topic,
        stories=tuple(stories),
        stored_chunks=len(documents),
        source_counts=dict(Counter(document.metadata["source_type"] for document in documents)),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Search and ingest Hacker News stories and discussions.")
    parser.add_argument("topic", help="Search term or topic, for example 'rust programming'")
    parser.add_argument("--limit", type=int, default=HACKERNEWS_DEFAULT_LIMIT, help="Maximum matching stories")
    parser.add_argument("--query", help="Optional grounded question after ingestion")
    args = parser.parse_args()

    try:
        report = ingest_hackernews_topic(args.topic, args.limit)
    except HackerNewsIngestionError as error:
        print(f"Ingestion failed: {error}")
        raise SystemExit(1) from error

    story_ids = [story.id for story in report.stories]
    print(f"Stored {report.stored_chunks} chunks from {len(report.stories)} Hacker News stories for: {report.topic}")
    print(f"Chunks by type: {report.source_counts}")

    if args.query:
        retrieved_documents = LocalChromaStore(collection_name=HACKERNEWS_COLLECTION).similarity_search(
            args.query, metadata_filter={"story_id": {"$in": story_ids}}
        )
        print("\nGrounded answer:")
        try:
            print(generate_grounded_answer(args.query, retrieved_documents))
        except AnswerGenerationError as error:
            print(f"Answer generation unavailable: {error}")
        print("\nTop matching chunks:")
        for document in retrieved_documents:
            print(f"- {document.metadata['citation']}")


if __name__ == "__main__":
    _main()

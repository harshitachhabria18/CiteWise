"""Run the Wikipedia search -> section chunk -> RAG workflow."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from citewise.ingestion.wikipedia import WikipediaArticle, WikipediaClient, WikipediaIngestionError
from citewise.processing.wikipedia_chunking import build_wikipedia_documents, wikipedia_document_ids
from citewise.retrieval.github_answer import AnswerGenerationError, generate_grounded_answer
from citewise.storage.chroma_store import LocalChromaStore


WIKIPEDIA_COLLECTION = "wikipedia_content"
WIKIPEDIA_DEFAULT_LIMIT = 4
WIKIPEDIA_RETRIEVAL_CANDIDATES = 50
WIKIPEDIA_CONTEXT_CHUNKS = 8
_ACHIEVEMENT_QUERY_TERMS = frozenset({"achievement", "achievements", "accomplishment", "award", "awards", "record", "records", "title", "titles"})
_ACHIEVEMENT_SECTION_TERMS = frozenset({"achievement", "achievements", "record", "records", "captain", "captaincy", "cup", "award", "awards", "honour", "honor", "statistic", "statistics", "championship", "title", "titles"})


@dataclass(frozen=True)
class WikipediaIngestionReport:
    topic: str
    articles: tuple[WikipediaArticle, ...]
    stored_chunks: int
    source_counts: dict[str, int]


def ingest_wikipedia_topic(topic: str, limit: int = WIKIPEDIA_DEFAULT_LIMIT) -> WikipediaIngestionReport:
    """Search, section-chunk, embed, and persist Wikipedia articles locally."""
    articles = WikipediaClient().search_articles(topic, limit=limit)
    documents = build_wikipedia_documents(articles)
    store = LocalChromaStore(collection_name=WIKIPEDIA_COLLECTION)
    for article in articles:
        store.delete_documents_by_metadata_filter({"page_id": article.page_id})
    store.add_documents(documents, wikipedia_document_ids(documents))
    return WikipediaIngestionReport(
        topic=topic,
        articles=tuple(articles),
        stored_chunks=len(documents),
        source_counts=dict(Counter(document.metadata["source_type"] for document in documents)),
    )


def retrieve_wikipedia_documents(
    store: LocalChromaStore, topic: str, articles: tuple[WikipediaArticle, ...], query: str
) -> list:
    """Prefer an exact-title article and avoid repeated chunks from one section.

    Wikipedia search can return similarly named films, albums, and brands alongside a
    person or subject. When a result title is an exact normalized match for the topic,
    it is the clearest primary source for a question about that topic. A wider candidate
    pool then lets section-diverse context include more than the first long section.
    """
    normalized_topic = _normalize_title(topic)
    primary_page_ids = [
        article.page_id for article in articles if _normalize_title(article.title) == normalized_topic
    ]
    page_ids = primary_page_ids or [article.page_id for article in articles]
    candidates = store.similarity_search(
        query,
        k=WIKIPEDIA_RETRIEVAL_CANDIDATES,
        metadata_filter={"page_id": {"$in": page_ids}},
    )
    return _select_section_diverse_documents(candidates, query)


def _normalize_title(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _select_section_diverse_documents(documents: list, query: str) -> list:
    """Keep the best-ranked chunk per section, favoring achievement-bearing headings."""
    question_terms = {term.strip("?.,!;:").casefold() for term in query.split()}
    achievement_query = bool(question_terms & _ACHIEVEMENT_QUERY_TERMS)

    def ranking_key(indexed_document: tuple[int, object]) -> tuple[int, int, int]:
        index, document = indexed_document
        section = str(document.metadata.get("section", "")).casefold()
        section_terms = set(section.replace("-", " ").split())
        achievement_bonus = int(
            achievement_query and bool(section_terms & _ACHIEVEMENT_SECTION_TERMS)
        )
        direct_matches = len(question_terms & section_terms)
        return (-achievement_bonus, -direct_matches, index)

    selected = []
    seen_sections: set[tuple[object, object]] = set()
    for _, document in sorted(enumerate(documents), key=ranking_key):
        section_key = (document.metadata.get("page_id"), document.metadata.get("section"))
        if section_key in seen_sections:
            continue
        selected.append(document)
        seen_sections.add(section_key)
        if len(selected) == WIKIPEDIA_CONTEXT_CHUNKS:
            break
    return selected


def _main() -> None:
    parser = argparse.ArgumentParser(description="Search and ingest Wikipedia articles by topic.")
    parser.add_argument("topic", help="Search term or topic, for example 'quantum computing'")
    parser.add_argument(
        "--limit",
        type=int,
        default=WIKIPEDIA_DEFAULT_LIMIT,
        help=f"Maximum matching articles, default: {WIKIPEDIA_DEFAULT_LIMIT}",
    )
    parser.add_argument("--query", help="Optional grounded question after ingestion")
    args = parser.parse_args()

    try:
        report = ingest_wikipedia_topic(args.topic, args.limit)
    except WikipediaIngestionError as error:
        print(f"Ingestion failed: {error}")
        raise SystemExit(1) from error

    print(f"Stored {report.stored_chunks} chunks from {len(report.articles)} Wikipedia articles for: {report.topic}")
    print(f"Chunks by type: {report.source_counts}")

    if args.query:
        retrieved_documents = retrieve_wikipedia_documents(
            LocalChromaStore(collection_name=WIKIPEDIA_COLLECTION),
            args.topic,
            report.articles,
            args.query,
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

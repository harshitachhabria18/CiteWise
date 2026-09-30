"""Run the complete GitHub fetch -> chunk -> embed -> Chroma workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from citewise.ingestion.github import GitHubClient, GitHubIngestionError
from citewise.processing.github_chunking import build_github_documents, document_ids
from citewise.retrieval.github_answer import AnswerGenerationError, generate_grounded_answer
from citewise.storage.chroma_store import LocalChromaStore


@dataclass(frozen=True)
class GitHubIngestionReport:
    repository: str
    stored_chunks: int
    source_counts: dict[str, int]
    notices: tuple[str, ...]


def ingest_github_repository(repository: str, limit: int = 20) -> GitHubIngestionReport:
    """Fetch all supported GitHub context and persist citation-ready chunks locally."""
    client = GitHubClient()
    notices = [client.warning] if client.warning else []
    issues = client.fetch_issues(repository, limit)
    discussions = client.fetch_discussions(repository, limit)
    if client.last_notice:
        notices.append(client.last_notice)
    readme = client.fetch_readme(repository)
    code_files = client.fetch_code_files(repository)
    documents = build_github_documents([*issues, *discussions], readme, code_files)

    store = LocalChromaStore()
    store.add_documents(documents, document_ids(documents))
    return GitHubIngestionReport(
        repository=repository,
        stored_chunks=len(documents),
        source_counts=store.count_by_source_type(),
        notices=tuple(notice for notice in notices if notice),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ingest public GitHub content into local Chroma.")
    parser.add_argument("repository", help="Public repository in owner/repository form")
    parser.add_argument("--limit", type=int, default=20, help="Maximum issues and discussions")
    parser.add_argument("--query", help="Optional test query after ingestion")
    args = parser.parse_args()

    try:
        report = ingest_github_repository(args.repository, args.limit)
    except GitHubIngestionError as error:
        print(f"Ingestion failed: {error}")
        raise SystemExit(1) from error

    for notice in report.notices:
        print(f"Notice: {notice}")
    print(f"Stored {report.stored_chunks} chunks from {report.repository}.")
    print(f"Chunks by type: {report.source_counts}")

    if args.query:
        retrieved_documents = LocalChromaStore().similarity_search(
            args.query, repository=report.repository
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

"""Persist and query source chunks with local Chroma and sentence-transformer embeddings."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"


class LocalChromaStore:
    """A small local vector store using all-MiniLM-L6-v2 by default."""

    def __init__(
        self,
        persist_directory: Path = DEFAULT_CHROMA_DIRECTORY,
        collection_name: str = "github_content",
        embedding_function: Any | None = None,
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        embeddings = embedding_function or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Upsert chunks with stable IDs so re-ingestion does not silently duplicate them."""
        if documents:
            self.vector_store.add_documents(documents=documents, ids=ids)

    def delete_documents_by_metadata_filter(self, metadata_filter: dict[str, Any]) -> None:
        """Delete all chunks matching metadata before their source is re-ingested.

        Upserts cannot remove chunks that were intentionally omitted from a newer
        ingestion (for example, a newly excluded Wikipedia References section), so
        a source refresh must first remove its prior records.
        """
        result = self.vector_store.get(include=[], where=metadata_filter)
        ids = result["ids"]
        if ids:
            self.vector_store.delete(ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 8,
        repository: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        ensure_source_type: str | None = None,
        candidate_k: int | None = None,
    ) -> list[Document]:
        """Return similar chunks, optionally reserving one source type from a larger candidate pool."""
        if repository and metadata_filter:
            raise ValueError("Use either repository or metadata_filter, not both.")
        filter_by = metadata_filter or ({"repository": repository} if repository else None)
        if not ensure_source_type:
            return self.vector_store.similarity_search(query, k=k, filter=filter_by)

        scored_candidates = self.vector_store.similarity_search_with_score(
            query,
            k=max(k, candidate_k or k),
            filter=filter_by,
        )
        documents = [document for document, _score in scored_candidates[:k]]
        preferred_document = next(
            (
                document
                for document, _score in scored_candidates
                if document.metadata.get("source_type") == ensure_source_type
            ),
            None,
        )
        if preferred_document and preferred_document not in documents:
            return [preferred_document, *documents[:-1]]
        return documents

    def count_by_source_type(
        self, metadata_filter: dict[str, Any] | None = None
    ) -> dict[str, int]:
        """Count source types, optionally within one metadata scope."""
        result = self.vector_store.get(include=["metadatas"], where=metadata_filter)
        return dict(Counter(metadata.get("source_type", "unknown") for metadata in result["metadatas"]))

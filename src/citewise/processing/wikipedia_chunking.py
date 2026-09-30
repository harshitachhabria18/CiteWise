"""Chunk Wikipedia article sections while retaining section-specific citations."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citewise.ingestion.wikipedia import WikipediaArticle


WIKIPEDIA_CHUNK_SIZE = 900
WIKIPEDIA_CHUNK_OVERLAP = 150
NON_CONTENT_SECTION_TITLES = frozenset(
    {"references", "external links", "see also", "further reading", "bibliography", "notes"}
)


def build_wikipedia_documents(articles: Iterable[WikipediaArticle]) -> list[Document]:
    """Split each section separately so retrieval preserves its article context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=WIKIPEDIA_CHUNK_SIZE, chunk_overlap=WIKIPEDIA_CHUNK_OVERLAP
    )
    documents: list[Document] = []
    for article in articles:
        for section in article.sections:
            if section.title.strip().casefold() in NON_CONTENT_SECTION_TITLES:
                continue
            citation = f"Wikipedia: {article.title} - {section.title} - {article.url}"
            for chunk_index, chunk in enumerate(splitter.split_text(section.text)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_type": "wikipedia_article",
                            "page_id": article.page_id,
                            "article_title": article.title,
                            "article_url": article.url,
                            "section": section.title,
                            "section_index": section.index,
                            "chunk_index": chunk_index,
                            "citation": citation,
                        },
                    )
                )
    return documents


def wikipedia_document_ids(documents: Iterable[Document]) -> list[str]:
    """Produce stable IDs so re-ingesting an article updates matching chunks."""
    return [
        sha256(
            f"{document.metadata['page_id']}:{document.metadata['section_index']}:"
            f"{document.metadata['chunk_index']}:{document.page_content}".encode()
        ).hexdigest()
        for document in documents
    ]

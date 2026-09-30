"""Chunk GitHub records while retaining source-specific citation metadata."""

from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from citewise.ingestion.github import GitHubCodeFile, GitHubContent, GitHubReadme


TEXT_CHUNK_SIZE = 900
TEXT_CHUNK_OVERLAP = 150
CODE_WHOLE_FILE_LINE_THRESHOLD = 400


def build_github_documents(
    content_items: Iterable[GitHubContent],
    readme: GitHubReadme,
    code_files: Iterable[GitHubCodeFile],
) -> list[Document]:
    """Create citation-ready chunks for issues, discussions, README, and code files."""
    documents: list[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP
    )

    for item in content_items:
        label = "Issue" if item.content_type == "issue" else "Discussion"
        citation = f"GitHub {label} #{item.number}: {item.title} ({item.repository})"
        source_text = f"{label} #{item.number}: {item.title}\n\n{item.body}".strip()
        documents.extend(
            _split_text(
                source_text,
                text_splitter,
                {
                    "source_type": f"github_{item.content_type}",
                    "repository": item.repository,
                    "title": item.title,
                    "number": item.number,
                    "author": item.author or "unknown",
                    "source_url": item.url,
                    "citation": citation,
                },
            )
        )

    documents.extend(
        _split_text(
            readme.content,
            text_splitter,
            {
                "source_type": "github_readme",
                "repository": readme.repository,
                "file_path": readme.path,
                "file_name": PurePosixPath(readme.path).name,
                "source_url": readme.url,
                "citation": f"GitHub README: {readme.path} ({readme.repository})",
            },
        )
    )

    for code_file in code_files:
        metadata = {
            "source_type": "github_file",
            "repository": code_file.repository,
            "file_path": code_file.path,
            "file_name": code_file.name,
            "source_url": code_file.url,
            "citation": f"GitHub File: {code_file.path} ({code_file.repository})",
        }
        if code_file.line_count <= CODE_WHOLE_FILE_LINE_THRESHOLD:
            documents.extend(
                _with_chunk_metadata([Document(page_content=code_file.content, metadata=metadata)])
            )
        else:
            documents.extend(_split_text(code_file.content, _code_splitter(code_file.path), metadata))

    return documents


def document_ids(documents: Iterable[Document]) -> list[str]:
    """Return stable IDs so repeated ingestion updates the same Chroma records."""
    return [
        sha256(
            f"{document.metadata.get('citation')}:{document.metadata.get('chunk_index')}:{document.page_content}".encode()
        ).hexdigest()
        for document in documents
    ]


def _split_text(text: str, splitter: RecursiveCharacterTextSplitter, metadata: dict[str, object]) -> list[Document]:
    if not text.strip():
        return []
    return _with_chunk_metadata(splitter.create_documents([text], metadatas=[metadata]))


def _with_chunk_metadata(documents: list[Document]) -> list[Document]:
    for index, document in enumerate(documents):
        document.metadata["chunk_index"] = index
    return documents


def _code_splitter(path: str) -> RecursiveCharacterTextSplitter:
    """Prefer language boundaries for common code types; otherwise use the text splitter."""
    language_by_suffix = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".java": Language.JAVA,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".cs": Language.CSHARP,
        ".cpp": Language.CPP,
        ".hpp": Language.CPP,
        ".c": Language.C,
        ".h": Language.C,
        ".kt": Language.KOTLIN,
        ".kts": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".swift": Language.SWIFT,
        ".html": Language.HTML,
    }
    language = language_by_suffix.get(PurePosixPath(path).suffix.lower())
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP
        )
    return RecursiveCharacterTextSplitter(chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP)

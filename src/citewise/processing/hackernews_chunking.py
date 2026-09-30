"""Chunk Hacker News stories and comments with source-specific citations."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citewise.ingestion.hackernews import HackerNewsStory


HN_CHUNK_SIZE = 900
HN_CHUNK_OVERLAP = 150


def build_hackernews_documents(stories: Iterable[HackerNewsStory]) -> list[Document]:
    """Build short-form, citation-ready documents from stories and bounded discussions."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=HN_CHUNK_SIZE, chunk_overlap=HN_CHUNK_OVERLAP)
    documents: list[Document] = []
    for story in stories:
        story_text = "\n\n".join(part for part in (story.title, story.text) if part).strip()
        documents.extend(
            _chunk_content(
                splitter, story_text, story, content_id=story.id, content_type="story", author=story.author,
                item_url=story.hn_url, depth=0,
            )
        )
        for comment in story.comments:
            documents.extend(
                _chunk_content(
                    splitter, comment.text, story, content_id=comment.id,
                    content_type="reply" if comment.depth else "comment", author=comment.author,
                    item_url=comment.url, depth=comment.depth,
                )
            )
    return documents


def hackernews_document_ids(documents: Iterable[Document]) -> list[str]:
    """Produce stable IDs so repeated topic searches update existing source chunks."""
    return [
        sha256(
            f"{document.metadata.get('story_id')}:{document.metadata.get('item_id')}:"
            f"{document.metadata.get('chunk_index')}:{document.page_content}".encode()
        ).hexdigest()
        for document in documents
    ]


def _chunk_content(
    splitter: RecursiveCharacterTextSplitter,
    text: str,
    story: HackerNewsStory,
    content_id: str,
    content_type: str,
    author: str,
    item_url: str,
    depth: int,
) -> list[Document]:
    if not text:
        return []
    source_label = "story" if content_type == "story" else f"{content_type} by {author}"
    citation = f"Hacker News: {story.title} ({source_label}) - {item_url}"
    return [
        Document(
            page_content=chunk,
            metadata={
                "source_type": f"hackernews_{content_type}",
                "story_id": story.id,
                "story_title": story.title,
                "story_url": story.url,
                "item_id": content_id,
                "item_url": item_url,
                "author": author,
                "score": story.score,
                "comment_depth": depth,
                "chunk_index": chunk_index,
                "citation": citation,
            },
        )
        for chunk_index, chunk in enumerate(splitter.split_text(text))
    ]

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from citewise.ingestion.hackernews import HackerNewsClient, HackerNewsNoResultsError
from citewise.processing.hackernews_chunking import build_hackernews_documents, hackernews_document_ids
from citewise.storage.chroma_store import LocalChromaStore


class FakeEmbeddings:
    """Fast deterministic vectors for an actual local Chroma write in this test."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


@dataclass
class FakeResponse:
    payload: object

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.search_params: dict[str, object] = {}
        self.items: dict[str, dict[str, Any]] = {
            "1": {"id": 1, "type": "story", "title": "Rust programming discussion", "text": "A <b>story</b> body.", "url": "https://example.com/rust", "by": "submitter", "score": 42, "kids": ["2"]},
            "2": {"id": 2, "type": "comment", "by": "alice", "text": "Top <i>comment</i>", "kids": [3]},
            "3": {"id": 3, "type": "comment", "by": "bob", "text": "A reply", "kids": []},
        }

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if "algolia" in url:
            self.search_params = kwargs["params"]
            return FakeResponse({"hits": [{"objectID": "1"}]})
        item_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return FakeResponse(self.items.get(item_id))


def test_search_fetches_story_comments_and_one_reply_level() -> None:
    session = FakeSession()
    stories = HackerNewsClient(session=session).search_stories("rust programming", limit=5)

    assert session.search_params == {"query": "rust programming", "tags": "story", "hitsPerPage": 5}
    assert len(stories) == 1
    assert stories[0].text == "A story body."
    assert [(comment.author, comment.depth) for comment in stories[0].comments] == [("alice", 0), ("bob", 1)]


def test_empty_search_has_a_clear_error() -> None:
    class EmptySearchSession(FakeSession):
        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            if "algolia" in url:
                return FakeResponse({"hits": []})
            return super().get(url, **kwargs)

    with pytest.raises(HackerNewsNoResultsError, match="No Hacker News stories matched"):
        HackerNewsClient(session=EmptySearchSession()).search_stories("no results")


def test_hackernews_chunks_keep_story_comment_and_reply_citations() -> None:
    story = HackerNewsClient(session=FakeSession()).search_stories("rust", limit=1)[0]
    documents = build_hackernews_documents([story])

    assert {document.metadata["source_type"] for document in documents} == {
        "hackernews_story", "hackernews_comment", "hackernews_reply"
    }
    comment_document = next(document for document in documents if document.metadata["source_type"] == "hackernews_comment")
    assert comment_document.metadata["author"] == "alice"
    assert comment_document.metadata["citation"] == (
        "Hacker News: Rust programming discussion (comment by alice) - "
        "https://news.ycombinator.com/item?id=2"
    )
    assert len(hackernews_document_ids(documents)) == len(set(hackernews_document_ids(documents)))


def test_hackernews_documents_can_be_persisted_by_chroma(tmp_path: Path) -> None:
    """Reject nullable HN metadata before it can break a live ingestion."""
    story = HackerNewsClient(session=FakeSession()).search_stories("rust", limit=1)[0]
    documents = build_hackernews_documents([story])
    store = LocalChromaStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_hackernews_content",
        embedding_function=FakeEmbeddings(),
    )

    store.add_documents(documents, hackernews_document_ids(documents))

    assert store.count_by_source_type() == {
        "hackernews_story": 1,
        "hackernews_comment": 1,
        "hackernews_reply": 1,
    }
    assert all(value is not None for document in documents for value in document.metadata.values())

"""Tests for Wikipedia discovery, section-aware chunking, and Chroma persistence."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from citewise.ingestion.wikipedia import (
    WikipediaClient,
    WikipediaDisambiguationError,
)
from citewise.ingestion.wikipedia import WikipediaArticle
from citewise.processing.wikipedia_chunking import (
    build_wikipedia_documents,
    wikipedia_document_ids,
)
from citewise.services.wikipedia_pipeline import retrieve_wikipedia_documents
from citewise.storage.chroma_store import LocalChromaStore
from langchain_core.documents import Document


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


@dataclass
class FakeResponse:
    payload: object = None
    text: str = ""
    status_code: int = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.search_params: dict[str, object] = {}

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("w/api.php"):
            self.search_params = kwargs["params"]
            return FakeResponse(
                {
                    "query": {
                        "search": [
                            {"pageid": 7, "title": "Rust (programming language)"},
                        ]
                    }
                }
            )
        return FakeResponse(
            text=(
                "<p>Rust is a systems programming language.</p>"
                "<h2>Memory safety</h2><p>Rust uses ownership and borrowing.</p>"
                "<h2>History</h2><p>Rust was first released in 2010.</p>"
                "<h2>rEfErEnCeS</h2><p>Reference list entry.</p>"
                "<h2>EXTERNAL LINKS</h2><p>External website list.</p>"
            )
        )


def test_search_fetches_article_and_preserves_section_boundaries() -> None:
    session = FakeSession()
    articles = WikipediaClient(session=session).search_articles("rust programming", limit=3)

    assert session.search_params == {
        "action": "query",
        "list": "search",
        "srsearch": "rust programming",
        "srlimit": 3,
        "format": "json",
        "formatversion": 2,
    }
    assert articles[0].title == "Rust (programming language)"
    assert [(section.title, section.text) for section in articles[0].sections] == [
        ("Introduction", "Rust is a systems programming language."),
        ("Memory safety", "Rust uses ownership and borrowing."),
        ("History", "Rust was first released in 2010."),
        ("rEfErEnCeS", "Reference list entry."),
        ("EXTERNAL LINKS", "External website list."),
    ]


def test_disambiguation_only_results_have_a_clear_error() -> None:
    class DisambiguationSession(FakeSession):
        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            if url.endswith("w/api.php"):
                return FakeResponse(
                    {"query": {"search": [{"pageid": 9, "title": "Mercury (disambiguation)"}]}}
                )
            raise AssertionError("A title-marked disambiguation page should not be fetched")

    with pytest.raises(WikipediaDisambiguationError, match="more specific topic"):
        WikipediaClient(session=DisambiguationSession()).search_articles("mercury")


def test_wikipedia_documents_keep_section_citations_and_persist(tmp_path: Path) -> None:
    article = WikipediaClient(session=FakeSession()).search_articles("rust", limit=1)[0]
    documents = build_wikipedia_documents([article])
    memory_safety = next(document for document in documents if document.metadata["section"] == "Memory safety")
    assert memory_safety.metadata["citation"] == (
        "Wikipedia: Rust (programming language) - Memory safety - "
        "https://en.wikipedia.org/wiki/Rust_%28programming_language%29"
    )
    assert {document.metadata["section"] for document in documents} == {
        "Introduction", "Memory safety", "History"
    }
    assert all(
        document.metadata["section"].casefold() not in {"references", "external links"}
        for document in documents
    )
    assert len(wikipedia_document_ids(documents)) == len(set(wikipedia_document_ids(documents)))

    store = LocalChromaStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_wikipedia_content",
        embedding_function=FakeEmbeddings(),
    )
    store.add_documents(documents, wikipedia_document_ids(documents))

    assert store.count_by_source_type() == {"wikipedia_article": 3}

    store.delete_documents_by_metadata_filter({"page_id": article.page_id})

    assert store.count_by_source_type() == {}


def test_retrieval_prefers_exact_topic_article_and_diversifies_sections() -> None:
    primary = WikipediaArticle(1, "MS Dhoni", "https://example.test/dhoni", ())
    film = WikipediaArticle(2, "M.S. Dhoni: The Untold Story", "https://example.test/film", ())
    candidates = [
        Document(page_content="Film plot", metadata={"page_id": 2, "section": "Plot"}),
        Document(page_content="Youth one", metadata={"page_id": 1, "section": "Youth career"}),
        Document(page_content="Youth two", metadata={"page_id": 1, "section": "Youth career"}),
        Document(page_content="Captaincy", metadata={"page_id": 1, "section": "Captaincy"}),
        Document(page_content="Records", metadata={"page_id": 1, "section": "Records and achievements"}),
    ]

    class FakeStore:
        def __init__(self) -> None:
            self.filter: dict[str, object] | None = None

        def similarity_search(self, _query: str, k: int, metadata_filter: dict[str, object]) -> list[Document]:
            assert k == 50
            self.filter = metadata_filter
            return candidates[1:]

    store = FakeStore()
    documents = retrieve_wikipedia_documents(
        store, "MS Dhoni", (primary, film), "What are MS Dhoni's major achievements in cricket?"
    )

    assert store.filter == {"page_id": {"$in": [1]}}
    assert [document.metadata["section"] for document in documents] == [
        "Records and achievements", "Captaincy", "Youth career"
    ]

from pathlib import Path

from langchain_core.documents import Document

from citewise.storage.chroma_store import LocalChromaStore


class FakeEmbeddings:
    """Fast deterministic vectors for testing storage without downloading a model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


def test_chroma_tracks_all_github_source_types(tmp_path: Path) -> None:
    documents = [
        Document(page_content="issue", metadata={"source_type": "github_issue", "citation": "Issue"}),
        Document(page_content="discussion", metadata={"source_type": "github_discussion", "citation": "Discussion"}),
        Document(page_content="readme", metadata={"source_type": "github_readme", "citation": "README"}),
        Document(page_content="code", metadata={"source_type": "github_file", "citation": "File"}),
    ]
    store = LocalChromaStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_github_content",
        embedding_function=FakeEmbeddings(),
    )

    store.add_documents(documents, ["issue", "discussion", "readme", "file"])

    assert store.count_by_source_type() == {
        "github_issue": 1,
        "github_discussion": 1,
        "github_readme": 1,
        "github_file": 1,
    }
    assert len(store.similarity_search("code", k=4)) == 4


def test_chroma_can_limit_retrieval_to_one_repository(tmp_path: Path) -> None:
    store = LocalChromaStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_repository_filter",
        embedding_function=FakeEmbeddings(),
    )
    store.add_documents(
        [
            Document(
                page_content="target repository content",
                metadata={"repository": "owner/target", "citation": "Target"},
            ),
            Document(
                page_content="other repository content",
                metadata={"repository": "owner/other", "citation": "Other"},
            ),
        ],
        ["target", "other"],
    )

    results = store.similarity_search("repository content", repository="owner/target")

    assert len(results) == 1
    assert results[0].metadata["repository"] == "owner/target"


def test_retrieval_can_reserve_a_readme_from_the_candidate_pool(tmp_path: Path) -> None:
    store = LocalChromaStore(
        persist_directory=tmp_path / "chroma",
        collection_name="readme_diversity",
        embedding_function=FakeEmbeddings(),
    )
    store.add_documents(
        [
            Document(
                page_content="query",
                metadata={"repository": "owner/repo", "source_type": "github_discussion"},
            ),
            Document(
                page_content="query",
                metadata={"repository": "owner/repo", "source_type": "github_discussion"},
            ),
            Document(
                page_content="query extra",
                metadata={"repository": "owner/repo", "source_type": "github_readme"},
            ),
        ],
        ["discussion-1", "discussion-2", "readme-1"],
    )

    results = store.similarity_search(
        "query",
        k=2,
        repository="owner/repo",
        ensure_source_type="github_readme",
        candidate_k=3,
    )

    assert len(results) == 2
    assert results[0].metadata["source_type"] == "github_readme"

from citewise.ingestion.github import GitHubCodeFile, GitHubContent, GitHubReadme
from citewise.processing.github_chunking import _code_splitter, build_github_documents


def test_build_github_documents_preserves_all_source_types() -> None:
    issue = GitHubContent(
        id="issue-1", repository="owner/repo", content_type="issue", number=12,
        title="Fix example", body="Issue body", state="open", author="octocat",
        url="https://github.com/owner/repo/issues/12", comments_count=0,
        created_at=None, updated_at=None,
    )
    discussion = GitHubContent(
        id="discussion-1", repository="owner/repo", content_type="discussion", number=7,
        title="Design discussion", body="Discussion body", state="open", author="hubot",
        url="https://github.com/owner/repo/discussions/7", comments_count=0,
        created_at=None, updated_at=None,
    )
    readme = GitHubReadme("owner/repo", "README.md", "# Example", "https://github.com/owner/repo#readme")
    code_file = GitHubCodeFile(
        "owner/repo", "src/example.py", "example.py", "def example():\n    return True\n",
        "https://github.com/owner/repo/blob/main/src/example.py",
    )

    documents = build_github_documents([issue, discussion], readme, [code_file])
    metadata = [document.metadata for document in documents]

    assert {item["source_type"] for item in metadata} == {
        "github_issue", "github_discussion", "github_readme", "github_file"
    }
    file_metadata = next(item for item in metadata if item["source_type"] == "github_file")
    assert file_metadata["file_path"] == "src/example.py"
    assert file_metadata["file_name"] == "example.py"
    assert file_metadata["citation"] == "GitHub File: src/example.py (owner/repo)"


def test_code_splitter_falls_back_for_unsupported_language_extensions() -> None:
    """SQL/CSS have no matching Language enum member in the installed LangChain version."""
    assert _code_splitter("schema.sql")
    assert _code_splitter("styles.css")
    assert _code_splitter("program.py")

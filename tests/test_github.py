from typing import Any

from citewise.ingestion.github import GitHubClient


class FakeResponse:
    status_code = 200
    ok = True
    headers: dict[str, str] = {}

    def json(self) -> list[dict[str, Any]]:
        return [
            {
                "node_id": "I_123",
                "number": 7,
                "title": "Example issue",
                "body": "Example issue body.",
                "state": "open",
                "user": {"login": "octocat"},
                "html_url": "https://github.com/octocat/Hello-World/issues/7",
                "comments": 2,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
            },
            {"pull_request": {"url": "https://api.github.com/pulls/8"}},
        ]


class FakeSession:
    def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()


class DiscussionsGoneResponse:
    status_code = 410
    ok = False
    headers: dict[str, str] = {}


class DiscussionsGoneSession:
    def get(self, *args: Any, **kwargs: Any) -> DiscussionsGoneResponse:
        return DiscussionsGoneResponse()


def test_fetch_issues_parses_issues_and_excludes_pull_requests() -> None:
    client = GitHubClient()
    client.session = FakeSession()  # type: ignore[assignment]

    issues = client.fetch_issues("octocat/Hello-World", limit=5)

    assert len(issues) == 1
    assert issues[0].content_type == "issue"
    assert issues[0].title == "Example issue"
    assert issues[0].author == "octocat"
    assert issues[0].comments_count == 2


def test_fetch_discussions_uses_the_shared_content_shape() -> None:
    client = GitHubClient()
    client.session = FakeSession()  # type: ignore[assignment]

    discussions = client.fetch_discussions("octocat/Hello-World", limit=5)

    assert discussions[0].content_type == "discussion"
    assert discussions[0].title == "Example issue"


def test_fetch_discussions_skips_a_gone_rest_endpoint() -> None:
    client = GitHubClient()
    client.session = DiscussionsGoneSession()  # type: ignore[assignment]

    discussions = client.fetch_discussions("octocat/Hello-World", limit=5)

    assert discussions == []
    assert client.last_notice == "Discussions are disabled for this repository, skipping."


def test_github_token_adds_authorization_header(monkeypatch: Any) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    client = GitHubClient()

    assert client.session.headers["Authorization"] == "Bearer test-token"
    assert client.warning is None

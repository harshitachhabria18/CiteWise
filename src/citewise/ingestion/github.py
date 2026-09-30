"""Fetch public GitHub issues and discussions through GitHub's REST API."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import os
import re
from typing import Any, Literal

import requests
from dotenv import load_dotenv

from citewise.config.settings import ENV_FILE, get_secret


GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".kt", ".kts",
    ".scala", ".sh", ".sql", ".html", ".css", ".scss", ".vue", ".svelte",
}
SKIPPED_DIRECTORIES = {".git", ".github", ".venv", "venv", "node_modules", "vendor", "dist", "build", "coverage", "__pycache__"}
SKIPPED_FILENAMES = {".gitignore", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock", "composer.lock"}


class GitHubIngestionError(RuntimeError):
    """Base error for a readable GitHub ingestion failure."""


class InvalidRepositoryError(GitHubIngestionError):
    """The repository is not in the expected owner/repository format."""


class RepositoryNotFoundError(GitHubIngestionError):
    """GitHub could not find a public repository at the supplied path."""


class GitHubRateLimitError(GitHubIngestionError):
    """GitHub's unauthenticated request allowance has been exhausted."""


class GitHubDiscussionsUnavailable(GitHubIngestionError):
    """The requested repository does not have GitHub Discussions enabled."""


@dataclass(frozen=True)
class GitHubReadme:
    """Raw README text, intentionally kept as one source record before chunking."""

    repository: str
    path: str
    content: str
    url: str


@dataclass(frozen=True)
class GitHubCodeFile:
    """One eligible text source file fetched in full from a repository."""

    repository: str
    path: str
    name: str
    content: str
    url: str

    @property
    def line_count(self) -> int:
        return len(self.content.splitlines())


@dataclass(frozen=True)
class GitHubContent:
    """A normalized issue or discussion, ready for later chunking and citation."""

    id: str
    repository: str
    content_type: Literal["issue", "discussion"]
    number: int
    title: str
    body: str
    state: str
    author: str | None
    url: str
    comments_count: int
    created_at: str | None
    updated_at: str | None


class GitHubClient:
    """Small client for public repository issues and discussions.

    It uses GitHub's official REST API without a token. Unauthenticated requests have
    a lower rate limit, so callers should request only the content they need.
    """

    def __init__(self, timeout_seconds: int = 15) -> None:
        load_dotenv(dotenv_path=ENV_FILE)
        self.timeout_seconds = timeout_seconds
        self.last_notice: str | None = None
        self.warning: str | None = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
                "User-Agent": "CiteWise/0.1",
            }
        )
        token = get_secret("GITHUB_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        else:
            self.warning = (
                "GITHUB_TOKEN is not configured; using GitHub's lower unauthenticated rate limit."
            )

    def fetch_issues(
        self, repository: str, limit: int = 10, state: str = "all"
    ) -> list[GitHubContent]:
        """Fetch issues, deliberately excluding pull requests from GitHub's mixed list."""
        self.last_notice = None
        repo = self._validate_repository(repository)
        payload = self._get_list(
            f"/repos/{repo}/issues",
            repo,
            {"state": state, "per_page": self._safe_limit(limit), "sort": "updated"},
        )
        return [
            self._to_content(item, repo, "issue")
            for item in payload
            if "pull_request" not in item
        ]

    def fetch_discussions(self, repository: str, limit: int = 10) -> list[GitHubContent]:
        """Best-effort discussion fetch; gracefully skip the retired REST route."""
        self.last_notice = None
        repo = self._validate_repository(repository)
        try:
            payload = self._get_list(
                f"/repos/{repo}/discussions",
                repo,
                {"per_page": self._safe_limit(limit), "direction": "desc"},
            )
        except GitHubDiscussionsUnavailable:
            self.last_notice = "Discussions are disabled for this repository, skipping."
            return []
        return [self._to_content(item, repo, "discussion") for item in payload]

    def fetch_readme(self, repository: str) -> GitHubReadme:
        """Fetch the repository README as raw text for the normal text chunking pipeline."""
        repo = self._validate_repository(repository)
        data = self._get_object(f"/repos/{repo}/readme", repo)
        content = self._decode_file_content(data, "README")
        return GitHubReadme(
            repository=repo,
            path=str(data.get("path", "README")),
            content=content,
            url=str(data.get("html_url", f"https://github.com/{repo}#readme")),
        )

    def fetch_code_files(self, repository: str) -> list[GitHubCodeFile]:
        """Recursively fetch all eligible code files, preserving every included file in full.

        Non-code, binary, generated/build, and lock/config files are explicitly skipped.
        An API failure is raised rather than silently dropping an eligible source file.
        """
        repo = self._validate_repository(repository)
        files: list[GitHubCodeFile] = []
        self._walk_code_path(repo, "", files)
        return files

    def _walk_code_path(
        self, repository: str, path: str, files: list[GitHubCodeFile]
    ) -> None:
        payload = self._get_object(
            f"/repos/{repository}/contents/{path}" if path else f"/repos/{repository}/contents",
            repository,
        )
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("type")
            entry_path = str(entry.get("path", ""))
            entry_name = str(entry.get("name", ""))
            if entry_type == "dir":
                if entry_name not in SKIPPED_DIRECTORIES:
                    self._walk_code_path(repository, entry_path, files)
            elif entry_type == "file" and self._is_eligible_code_file(entry_path, entry_name):
                file_data = self._get_object(
                    f"/repos/{repository}/contents/{entry_path}", repository
                )
                try:
                    content = self._decode_file_content(file_data, entry_path)
                except UnicodeDecodeError:
                    continue  # Binary content is deliberately not ingested.
                files.append(
                    GitHubCodeFile(
                        repository=repository,
                        path=entry_path,
                        name=entry_name,
                        content=content,
                        url=str(file_data.get("html_url", entry.get("html_url", ""))),
                    )
                )

    @staticmethod
    def _is_eligible_code_file(path: str, name: str) -> bool:
        if name.lower() in SKIPPED_FILENAMES or name.endswith(".lock"):
            return False
        return os.path.splitext(path)[1].lower() in CODE_EXTENSIONS

    def _get_list(
        self, path: str, repository: str, params: dict[str, str | int]
    ) -> list[dict[str, Any]]:
        try:
            response = self.session.get(
                f"{GITHUB_API_BASE_URL}{path}", params=params, timeout=self.timeout_seconds
            )
        except requests.Timeout as error:
            raise GitHubIngestionError(
                "GitHub took too long to respond. Please try again in a moment."
            ) from error
        except requests.RequestException as error:
            raise GitHubIngestionError(
                "Could not reach GitHub. Check your internet connection and try again."
            ) from error

        self._raise_for_status(response, repository)
        try:
            payload = response.json()
        except ValueError as error:
            raise GitHubIngestionError(
                "GitHub returned an unexpected response. Please try again later."
            ) from error
        if not isinstance(payload, list):
            raise GitHubIngestionError("GitHub returned an unexpected data format.")
        return [item for item in payload if isinstance(item, dict)]

    def _get_object(self, path: str, repository: str) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            response = self.session.get(
                f"{GITHUB_API_BASE_URL}{path}", timeout=self.timeout_seconds
            )
        except requests.Timeout as error:
            raise GitHubIngestionError("GitHub took too long to respond. Please try again in a moment.") from error
        except requests.RequestException as error:
            raise GitHubIngestionError("Could not reach GitHub. Check your internet connection and try again.") from error
        self._raise_for_status(response, repository)
        try:
            payload = response.json()
        except ValueError as error:
            raise GitHubIngestionError("GitHub returned an unexpected response. Please try again later.") from error
        if not isinstance(payload, (dict, list)):
            raise GitHubIngestionError("GitHub returned an unexpected data format.")
        return payload

    @staticmethod
    def _decode_file_content(data: dict[str, Any] | list[dict[str, Any]], label: str) -> str:
        if not isinstance(data, dict) or data.get("encoding") != "base64":
            raise GitHubIngestionError(f"GitHub did not return readable content for {label}.")
        encoded_content = str(data.get("content", "")).replace("\n", "")
        try:
            return base64.b64decode(encoded_content).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise UnicodeDecodeError("utf-8", b"", 0, 0, f"{label} is not UTF-8 text") from error

    @staticmethod
    def _validate_repository(repository: str) -> str:
        normalized = repository.strip().strip("/")
        if not REPOSITORY_PATTERN.fullmatch(normalized):
            raise InvalidRepositoryError(
                "Enter a repository as owner/repository (for example, psf/requests)."
            )
        return normalized

    @staticmethod
    def _safe_limit(limit: int) -> int:
        return max(1, min(limit, 100))

    @staticmethod
    def _raise_for_status(response: requests.Response, repository: str) -> None:
        if response.status_code == 410:
            raise GitHubDiscussionsUnavailable(
                "Discussions are disabled for this repository, skipping."
            )
        if response.status_code == 404:
            raise RepositoryNotFoundError(
                f"{repository} was not found, is private, or does not have this feature enabled."
            )
        if response.status_code in (403, 429):
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            details = ""
            if remaining == "0":
                details = f" Rate limit resets at Unix time {reset}." if reset else ""
            raise GitHubRateLimitError(
                "GitHub denied the request or the unauthenticated rate limit was reached."
                f"{details}"
            )
        if not response.ok:
            raise GitHubIngestionError(
                f"GitHub returned HTTP {response.status_code}. Please try again later."
            )

    @staticmethod
    def _to_content(
        data: dict[str, Any], repository: str, content_type: Literal["issue", "discussion"]
    ) -> GitHubContent:
        user = data.get("user") or data.get("author") or {}
        author = user.get("login") if isinstance(user, dict) else None
        comments_count = data.get("comments", 0)
        if isinstance(comments_count, list):
            comments_count = len(comments_count)

        return GitHubContent(
            id=str(data.get("node_id") or data.get("id", "")),
            repository=repository,
            content_type=content_type,
            number=int(data.get("number", 0)),
            title=str(data.get("title", "")),
            body=str(data.get("body") or ""),
            state=str(data.get("state", "open")),
            author=str(author) if author else None,
            url=str(data.get("html_url", "")),
            comments_count=int(comments_count or 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


def _main() -> None:
    """Provide a manual check once the user supplies a public owner/repository."""
    parser = argparse.ArgumentParser(description="Fetch GitHub repository issues and discussions.")
    parser.add_argument("repository", help="Public repository in owner/repository form")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    client = GitHubClient()

    issues: list[GitHubContent] = []
    discussions: list[GitHubContent] = []
    failures: list[GitHubIngestionError] = []

    try:
        issues = client.fetch_issues(args.repository, args.limit)
    except GitHubIngestionError as error:
        print(f"Could not fetch issues from {args.repository}: {error}")
        failures.append(error)

    try:
        discussions = client.fetch_discussions(args.repository, args.limit)
    except GitHubIngestionError as error:
        print(f"Could not fetch discussions from {args.repository}: {error}")
        failures.append(error)

    if client.last_notice:
        print(client.last_notice)

    if not issues and not discussions and failures:
        raise SystemExit(1)

    for content in [*issues, *discussions]:
        print(f"[{content.content_type}] #{content.number}: {content.title}\n  {content.url}")


if __name__ == "__main__":
    _main()

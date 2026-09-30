"""DEPRECATED: former low-volume fallback client for Reddit JSON responses.

Reddit app registration is blocked by a persistent reCAPTCHA issue. Hacker News replaced
Reddit as the supported discussion source; this module remains only as historical reference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
from typing import Any

import requests


REDDIT_BASE_URL = "https://old.reddit.com"
SUBREDDIT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,21}$")
DEFAULT_USER_AGENT = "CiteWise/0.1 (learning project; contact: local developer)"


class RedditIngestionError(RuntimeError):
    """Base error for a user-friendly Reddit ingestion failure."""


class InvalidSubredditError(RedditIngestionError):
    """The supplied subreddit name cannot be used in a Reddit URL."""


class SubredditNotFoundError(RedditIngestionError):
    """Reddit did not find the requested public subreddit."""


class RedditRateLimitError(RedditIngestionError):
    """Reddit asked the client to slow down."""


@dataclass(frozen=True)
class RedditPost:
    """The post fields needed by the later text-chunking and citation steps."""

    id: str
    subreddit: str
    title: str
    selftext: str
    score: int
    permalink: str
    url: str
    author: str | None
    created_utc: float | None


class RedditJsonClient:
    """Fetch recent public subreddit posts without OAuth credentials.

    The client intentionally makes one request per call. It does not retry rate-limit
    responses, because immediately retrying would make the limit worse.
    """

    def __init__(self, user_agent: str | None = None, timeout_seconds: int = 15) -> None:
        configured_user_agent = os.getenv("REDDIT_USER_AGENT", "").strip()
        self.user_agent = user_agent or configured_user_agent or DEFAULT_USER_AGENT
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def fetch_recent_posts(self, subreddit: str, limit: int = 10) -> list[RedditPost]:
        """Fetch up to ``limit`` current posts and convert them into stable records."""
        normalized_subreddit = self._validate_subreddit(subreddit)
        safe_limit = max(1, min(limit, 100))
        url = f"{REDDIT_BASE_URL}/r/{normalized_subreddit}/.json"

        try:
            response = self.session.get(
                url,
                params={"limit": safe_limit},
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as error:
            raise RedditIngestionError(
                "Reddit took too long to respond. Please try again in a moment."
            ) from error
        except requests.RequestException as error:
            raise RedditIngestionError(
                "Could not reach Reddit. Check your internet connection and try again."
            ) from error

        self._raise_for_status(response, normalized_subreddit)

        try:
            payload = response.json()
            children = payload["data"]["children"]
        except (KeyError, TypeError, ValueError) as error:
            raise RedditIngestionError(
                "Reddit returned an unexpected response. Please try again later."
            ) from error

        return [
            self._to_post(child["data"], normalized_subreddit)
            for child in children
            if child.get("kind") == "t3" and isinstance(child.get("data"), dict)
        ]

    @staticmethod
    def _validate_subreddit(subreddit: str) -> str:
        normalized = subreddit.strip().removeprefix("r/").removeprefix("R/")
        if not SUBREDDIT_NAME_PATTERN.fullmatch(normalized):
            raise InvalidSubredditError(
                "Enter a subreddit name using 3–21 letters, numbers, or underscores "
                "(for example, personalfinance)."
            )
        return normalized

    @staticmethod
    def _raise_for_status(response: requests.Response, subreddit: str) -> None:
        if response.status_code == 404:
            raise SubredditNotFoundError(
                f"r/{subreddit} was not found or is not publicly available."
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_message = f" Wait {retry_after} seconds before trying again." if retry_after else ""
            raise RedditRateLimitError(
                "Reddit is rate-limiting requests. Please slow down and try again later."
                f"{wait_message}"
            )
        if response.status_code == 403:
            raise RedditIngestionError(
                "Reddit denied this public request. The subreddit may be private, or "
                "Reddit may require approved API/OAuth access for this request."
            )
        if not response.ok:
            raise RedditIngestionError(
                f"Reddit returned HTTP {response.status_code}. Please try again later."
            )

    @staticmethod
    def _to_post(data: dict[str, Any], subreddit: str) -> RedditPost:
        permalink = str(data.get("permalink", ""))
        full_url = f"{REDDIT_BASE_URL}{permalink}" if permalink.startswith("/") else permalink
        author = data.get("author")

        return RedditPost(
            id=str(data.get("id", "")),
            subreddit=str(data.get("subreddit", subreddit)),
            title=str(data.get("title", "")),
            selftext=str(data.get("selftext", "")),
            score=int(data.get("score", 0) or 0),
            permalink=permalink,
            url=full_url,
            author=str(author) if author else None,
            created_utc=float(data["created_utc"]) if data.get("created_utc") else None,
        )


def _main() -> None:
    """Provide a small manual smoke test without needing API credentials."""
    parser = argparse.ArgumentParser(description="Fetch public Reddit posts as JSON.")
    parser.add_argument("subreddit", nargs="?", default="personalfinance")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    try:
        posts = RedditJsonClient().fetch_recent_posts(args.subreddit, args.limit)
    except RedditIngestionError as error:
        print(f"Could not fetch r/{args.subreddit}: {error}")
        raise SystemExit(1) from error

    print(f"Fetched {len(posts)} posts from r/{args.subreddit}.")
    for post in posts:
        print(f"- [{post.score}] {post.title}\n  {post.url}")


if __name__ == "__main__":
    _main()

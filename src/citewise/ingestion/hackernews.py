"""Search Hacker News stories and fetch their discussion through public APIs."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any

import requests


HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"
HN_ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={item_id}"
MAX_TOP_LEVEL_COMMENTS = 10
MAX_REPLIES_PER_COMMENT = 3


class HackerNewsIngestionError(RuntimeError):
    """Base error for a readable Hacker News ingestion failure."""


class HackerNewsNoResultsError(HackerNewsIngestionError):
    """A topic search returned no usable stories."""


@dataclass(frozen=True)
class HackerNewsComment:
    """One comment or direct reply attached to an HN story."""

    id: str
    story_id: str
    story_title: str
    author: str
    text: str
    depth: int
    url: str


@dataclass(frozen=True)
class HackerNewsStory:
    """A searched HN story and a bounded portion of its discussion."""

    id: str
    title: str
    text: str
    url: str
    hn_url: str
    author: str
    score: int
    comments: tuple[HackerNewsComment, ...]


class _HtmlTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "br", "li", "pre", "blockquote"}:
            self.parts.append("\n")

    def text(self) -> str:
        return " ".join("".join(self.parts).split())


class HackerNewsClient:
    """Use Algolia discovery plus the canonical public Hacker News item API."""

    def __init__(self, timeout_seconds: int = 15, session: requests.Session | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def search_stories(self, topic: str, limit: int = 10) -> list[HackerNewsStory]:
        """Find matching stories, then load each story and bounded comment context."""
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise HackerNewsIngestionError("Enter a non-empty Hacker News search topic.")
        if limit < 1:
            raise HackerNewsIngestionError("--limit must be at least 1.")
        try:
            response = self.session.get(
                HN_ALGOLIA_SEARCH_URL,
                params={"query": normalized_topic, "tags": "story", "hitsPerPage": limit},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])
        except (requests.RequestException, ValueError, AttributeError) as error:
            raise HackerNewsIngestionError(
                "Could not search Hacker News. Check your connection and try again."
            ) from error

        stories: list[HackerNewsStory] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            item_id = str(hit.get("objectID") or hit.get("story_id") or "")
            if not item_id.isdigit():
                continue
            story = self.fetch_story(item_id)
            if story is not None:
                stories.append(story)
        if not stories:
            raise HackerNewsNoResultsError(
                f"No Hacker News stories matched '{normalized_topic}'. Try another topic."
            )
        return stories

    def fetch_story(self, story_id: str) -> HackerNewsStory | None:
        """Load a story plus top-level comments and one bounded reply level."""
        item = self._fetch_item(story_id)
        if not self._is_live_item(item, "story"):
            return None
        assert item is not None
        title = self.html_to_text(str(item.get("title") or "")).strip()
        if not title:
            return None
        canonical_url = HN_ITEM_URL.format(item_id=story_id)
        comments: list[HackerNewsComment] = []
        for comment_id in self._item_kids(item)[:MAX_TOP_LEVEL_COMMENTS]:
            comment_item = self._fetch_item(comment_id)
            comment = self._comment_from_item(comment_item, comment_id, story_id, title, depth=0)
            if comment is None:
                continue
            comments.append(comment)
            for reply_id in self._item_kids(comment_item)[:MAX_REPLIES_PER_COMMENT]:
                reply_item = self._fetch_item(reply_id)
                reply = self._comment_from_item(reply_item, reply_id, story_id, title, depth=1)
                if reply is not None:
                    comments.append(reply)
        return HackerNewsStory(
            id=story_id,
            title=title,
            text=self.html_to_text(str(item.get("text") or "")),
            url=str(item.get("url") or canonical_url),
            hn_url=canonical_url,
            author=str(item.get("by") or "unknown"),
            score=int(item.get("score") or 0),
            comments=tuple(comments),
        )

    def _comment_from_item(
        self, item: dict[str, Any] | None, comment_id: str, story_id: str, story_title: str, depth: int
    ) -> HackerNewsComment | None:
        if not self._is_live_item(item, "comment"):
            return None
        assert item is not None
        text = self.html_to_text(str(item.get("text") or ""))
        if not text:
            return None
        return HackerNewsComment(
            id=comment_id,
            story_id=story_id,
            story_title=story_title,
            author=str(item.get("by") or "unknown"),
            text=text,
            depth=depth,
            url=HN_ITEM_URL.format(item_id=comment_id),
        )

    def _fetch_item(self, item_id: str) -> dict[str, Any] | None:
        try:
            response = self.session.get(
                f"{HN_API_BASE_URL}/item/{item_id}.json", timeout=self.timeout_seconds
            )
            response.raise_for_status()
            item = response.json()
        except (requests.RequestException, ValueError) as error:
            raise HackerNewsIngestionError(
                "Could not retrieve a Hacker News story or comment. Please try again."
            ) from error
        return item if isinstance(item, dict) else None

    @staticmethod
    def html_to_text(value: str) -> str:
        parser = _HtmlTextParser()
        parser.feed(unescape(value))
        return parser.text()

    @staticmethod
    def _is_live_item(item: dict[str, Any] | None, item_type: str) -> bool:
        return bool(item) and item.get("type") == item_type and not item.get("deleted") and not item.get("dead")

    @staticmethod
    def _item_kids(item: dict[str, Any]) -> list[str]:
        kids = item.get("kids", [])
        return [str(item_id) for item_id in kids if isinstance(item_id, int) or str(item_id).isdigit()]

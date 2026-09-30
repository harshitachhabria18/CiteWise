"""Search English Wikipedia and retrieve article text grouped by section."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

import requests


WIKIPEDIA_ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_HTML_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"
WIKIPEDIA_ARTICLE_URL = "https://en.wikipedia.org/wiki/{title}"


class WikipediaIngestionError(RuntimeError):
    """Base error for readable Wikipedia ingestion failures."""


class WikipediaNoResultsError(WikipediaIngestionError):
    """A Wikipedia topic search returned no usable articles."""


class WikipediaDisambiguationError(WikipediaIngestionError):
    """Search results resolve only to disambiguation pages."""


@dataclass(frozen=True)
class WikipediaSection:
    """Plain article text under one visible Wikipedia section heading."""

    title: str
    text: str
    index: int


@dataclass(frozen=True)
class WikipediaArticle:
    """A Wikipedia article with its readable text grouped by section."""

    page_id: int
    title: str
    url: str
    sections: tuple[WikipediaSection, ...]


class _WikipediaSectionParser(HTMLParser):
    """Extract human-readable prose while retaining h2-h6 section boundaries."""

    _CONTENT_TAGS = {"p", "li", "dd", "dt", "blockquote", "pre"}
    _HEADING_TAGS = {"h2", "h3", "h4", "h5", "h6"}
    _IGNORED_TAGS = {"script", "style", "sup", "table", "math", "figure", "nav"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[WikipediaSection] = []
        self.current_title = "Introduction"
        self.current_parts: list[str] = []
        self.current_heading: list[str] | None = None
        self.content_depth = 0
        self.ignored_depth = 0
        self.is_disambiguation = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "disambig":
            self.is_disambiguation = True
        if tag in self._IGNORED_TAGS:
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return
        if tag in self._HEADING_TAGS:
            self._flush_section()
            self.current_heading = []
        elif tag in self._CONTENT_TAGS:
            self.content_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED_TAGS:
            self.ignored_depth = max(0, self.ignored_depth - 1)
            return
        if self.ignored_depth:
            return
        if tag in self._HEADING_TAGS and self.current_heading is not None:
            heading = " ".join("".join(self.current_heading).split())
            self.current_title = heading or "Introduction"
            self.current_heading = None
        elif tag in self._CONTENT_TAGS and self.content_depth:
            self.content_depth -= 1
            self.current_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self.current_heading is not None:
            self.current_heading.append(data)
        elif self.content_depth:
            self.current_parts.append(data)

    def close_sections(self) -> tuple[WikipediaSection, ...]:
        self._flush_section()
        return tuple(self.sections)

    def _flush_section(self) -> None:
        text = " ".join("".join(self.current_parts).split())
        if text:
            self.sections.append(WikipediaSection(self.current_title, text, len(self.sections)))
        self.current_parts = []


class WikipediaClient:
    """Use Wikipedia's public Action API for search and REST HTML for article sections."""

    def __init__(self, timeout_seconds: int = 20, session: requests.Session | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers["User-Agent"] = "CiteWise/1.0 (educational local RAG project)"

    def search_articles(self, topic: str, limit: int = 4) -> list[WikipediaArticle]:
        """Find topic matches and fetch the full text for each non-disambiguation article."""
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise WikipediaIngestionError("Enter a non-empty Wikipedia search topic.")
        if limit < 1:
            raise WikipediaIngestionError("--limit must be at least 1.")
        try:
            response = self.session.get(
                WIKIPEDIA_ACTION_API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": normalized_topic,
                    "srlimit": limit,
                    "format": "json",
                    "formatversion": 2,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            results = response.json().get("query", {}).get("search", [])
        except (requests.RequestException, ValueError, AttributeError) as error:
            raise WikipediaIngestionError(
                "Could not search Wikipedia. Check your connection and try again."
            ) from error

        articles: list[WikipediaArticle] = []
        disambiguations = 0
        for result in results:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "").strip()
            page_id = result.get("pageid")
            if not title or not isinstance(page_id, int):
                continue
            try:
                article = self.fetch_article(title, page_id)
            except WikipediaDisambiguationError:
                disambiguations += 1
                continue
            if article is not None:
                articles.append(article)
        if articles:
            return articles
        if disambiguations:
            raise WikipediaDisambiguationError(
                f"Wikipedia found only disambiguation pages for '{normalized_topic}'. Try a more specific topic."
            )
        raise WikipediaNoResultsError(
            f"No usable Wikipedia articles matched '{normalized_topic}'. Try another topic."
        )

    def fetch_article(self, title: str, page_id: int) -> WikipediaArticle | None:
        """Fetch an article's official REST HTML and turn visible headings into sections."""
        if title.lower().endswith("(disambiguation)"):
            raise WikipediaDisambiguationError(title)
        encoded_title = quote(title.replace(" ", "_"), safe="")
        try:
            response = self.session.get(
                WIKIPEDIA_REST_HTML_URL.format(title=encoded_title), timeout=self.timeout_seconds
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
        except requests.RequestException as error:
            raise WikipediaIngestionError(
                "Could not retrieve a Wikipedia article. Please try again."
            ) from error
        parser = _WikipediaSectionParser()
        parser.feed(response.text)
        sections = parser.close_sections()
        if parser.is_disambiguation:
            raise WikipediaDisambiguationError(title)
        if not sections:
            return None
        return WikipediaArticle(
            page_id=page_id,
            title=title,
            url=WIKIPEDIA_ARTICLE_URL.format(title=encoded_title),
            sections=sections,
        )

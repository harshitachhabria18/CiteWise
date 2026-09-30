from typing import Any

from citewise.ingestion.reddit_json import RedditJsonClient


class FakeResponse:
    status_code = 200
    ok = True
    headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "abc123",
                            "subreddit": "personalfinance",
                            "title": "Example savings question",
                            "selftext": "Example post body.",
                            "score": 42,
                            "permalink": "/r/personalfinance/comments/abc123/example/",
                            "author": "example_author",
                            "created_utc": 1_700_000_000,
                        },
                    }
                ]
            }
        }


class FakeSession:
    def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()


def test_fetch_recent_posts_parses_reddit_listing() -> None:
    """Verify post fields without making a fragile live network request."""
    client = RedditJsonClient()
    client.session = FakeSession()  # type: ignore[assignment]

    posts = client.fetch_recent_posts("personalfinance", limit=1)

    assert len(posts) == 1
    assert posts[0].title == "Example savings question"
    assert posts[0].selftext == "Example post body."
    assert posts[0].score == 42
    assert posts[0].permalink == "/r/personalfinance/comments/abc123/example/"
    assert posts[0].url == "https://old.reddit.com/r/personalfinance/comments/abc123/example/"

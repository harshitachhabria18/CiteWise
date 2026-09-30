"""DEPRECATED: reserved OAuth/PRAW alternative retained only as historical reference.

Reddit app registration is blocked by a persistent reCAPTCHA issue. Hacker News is the
supported discussion source, so this module must not be used by active pipelines.
"""

# Planned implementation once credentials are approved:
#
# import praw
# from citewise.config import get_settings
#
# def create_reddit_client() -> praw.Reddit:
#     settings = get_settings()
#     return praw.Reddit(
#         client_id=settings.reddit_client_id,
#         client_secret=settings.reddit_client_secret,
#         user_agent=settings.reddit_user_agent,
#     )
#
# The PRAW client should convert its submissions into the same RedditPost dataclass
# defined in reddit_json.py. That keeps all downstream code unchanged.

"""Load and validate private configuration from the local .env file."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(ValueError):
    """Raised when a required application credential has not been configured."""


@dataclass(frozen=True)
class Settings:
    """Validated credentials used by the Reddit client and Groq LLM client."""

    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    groq_api_key: str


def get_settings() -> Settings:
    """Return validated settings, or explain exactly what the developer must add."""
    load_dotenv(dotenv_path=ENV_FILE)

    required_values = {
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID", "").strip(),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT", "").strip(),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "").strip(),
    }
    missing_keys = [key for key, value in required_values.items() if not value]

    if missing_keys:
        missing_list = ", ".join(missing_keys)
        raise ConfigurationError(
            "Missing required configuration: "
            f"{missing_list}. Add these values to {ENV_FILE.name}; "
            "use .env.example as the template."
        )

    return Settings(
        reddit_client_id=required_values["REDDIT_CLIENT_ID"],
        reddit_client_secret=required_values["REDDIT_CLIENT_SECRET"],
        reddit_user_agent=required_values["REDDIT_USER_AGENT"],
        groq_api_key=required_values["GROQ_API_KEY"],
    )

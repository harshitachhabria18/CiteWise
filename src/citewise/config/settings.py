"""Load and validate private configuration from the local .env file."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(ValueError):
    """Raised when a required application credential has not been configured."""


def get_secret(name: str) -> str:
    """Read a credential from the environment, then Streamlit Cloud secrets if available."""
    environment_value = os.getenv(name, "").strip()
    if environment_value:
        return environment_value
    try:
        import streamlit as st

        secret_value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(secret_value).strip()


@dataclass(frozen=True)
class Settings:
    """Deprecated legacy settings retained while the package name is migrated."""

    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    groq_api_key: str


def get_settings() -> Settings:
    """Return validated settings, or explain exactly what the developer must add."""
    load_dotenv(dotenv_path=ENV_FILE)

    required_values = {
        "REDDIT_CLIENT_ID": get_secret("REDDIT_CLIENT_ID"),
        "REDDIT_CLIENT_SECRET": get_secret("REDDIT_CLIENT_SECRET"),
        "REDDIT_USER_AGENT": get_secret("REDDIT_USER_AGENT"),
        "GROQ_API_KEY": get_secret("GROQ_API_KEY"),
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

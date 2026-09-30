import sys
from types import SimpleNamespace

from citewise.config.settings import get_secret


def test_get_secret_prefers_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("CITEWISE_TEST_SECRET", "environment-value")
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={"CITEWISE_TEST_SECRET": "cloud-value"}))

    assert get_secret("CITEWISE_TEST_SECRET") == "environment-value"


def test_get_secret_falls_back_to_streamlit_secrets(monkeypatch) -> None:
    monkeypatch.delenv("CITEWISE_TEST_SECRET", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={"CITEWISE_TEST_SECRET": "cloud-value"}))

    assert get_secret("CITEWISE_TEST_SECRET") == "cloud-value"

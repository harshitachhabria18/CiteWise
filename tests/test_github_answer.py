from typing import Any

import pytest
from langchain_core.documents import Document

import citewise.retrieval.github_answer as answer_module
from citewise.retrieval.github_answer import (
    ANSWER_PROMPT,
    DEBUG_CONTEXT_ENVIRONMENT_VARIABLE,
    GROQ_MAX_TOKENS,
    GROQ_REASONING_EFFORT,
    AnswerGenerationError,
    INSUFFICIENT_CONTEXT_ANSWER,
    MissingGroqApiKeyError,
    _answer_from_response,
    format_context,
    generate_grounded_answer,
)


def test_context_uses_readable_citations_as_chunk_labels() -> None:
    context = format_context(
        [
            Document(
                page_content="Argument parsing uses decorators.",
                metadata={"citation": "GitHub File: src/click/core.py (pallets/click)"},
            )
        ]
    )

    assert context.startswith("[GitHub File: src/click/core.py (pallets/click)]")
    assert "Argument parsing uses decorators." in context
    assert "[Source " not in context


def test_context_budget_keeps_the_highest_ranked_chunks_first() -> None:
    context = format_context(
        [
            Document(page_content="first-ranked", metadata={"citation": "First"}),
            Document(page_content="second-ranked", metadata={"citation": "Second"}),
        ],
        max_characters=30,
    )

    assert context.startswith("[First]\nfirst-ranked")
    assert len(context) == 30
    assert "[Second]" in context


def test_missing_groq_key_has_clear_error(monkeypatch: Any) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(answer_module, "load_dotenv", lambda **_: False)

    with pytest.raises(MissingGroqApiKeyError, match="GROQ_API_KEY is missing"):
        generate_grounded_answer("How are arguments handled?", [Document(page_content="x")])


def test_rate_limit_retries_once_with_optional_fallback_key(monkeypatch: Any) -> None:
    class RateLimitError(Exception):
        status_code = 429

    class Response:
        content = "Grounded answer"

    keys_used: list[str] = []

    def fake_invoke(api_key: str, _prompt_values: dict[str, str]) -> object:
        keys_used.append(api_key)
        if api_key == "primary":
            raise RateLimitError()
        return Response()

    monkeypatch.setenv("GROQ_API_KEY", "primary")
    monkeypatch.setenv("GROQ_API_KEY_2", "fallback")
    monkeypatch.setattr(answer_module, "load_dotenv", lambda **_: False)
    monkeypatch.setattr(answer_module, "_invoke_groq", fake_invoke)

    assert generate_grounded_answer("Question", [Document(page_content="source")]) == "Grounded answer"
    assert keys_used == ["primary", "fallback"]


def test_empty_groq_response_has_a_visible_insufficient_context_fallback() -> None:
    class EmptyResponse:
        content = "   "

    assert _answer_from_response(EmptyResponse()) == INSUFFICIENT_CONTEXT_ANSWER


def test_answer_normalizes_full_width_and_square_source_label_brackets() -> None:
    class Response:
        content = (
            "README details 【GitHub README: README.md (josharsh/100LinesOfCode)】. "
            "More details [GitHub File: app.py (josharsh/100LinesOfCode)]."
        )

    assert _answer_from_response(Response()) == (
        "README details (GitHub README: README.md (josharsh/100LinesOfCode)). "
        "More details (GitHub File: app.py (josharsh/100LinesOfCode))."
    )


def test_prompt_synthesizes_relevant_advice_when_the_question_uses_different_words() -> None:
    messages = ANSWER_PROMPT.invoke({"question": "What are the do's and don'ts?", "context": "Advice"}).messages

    assert "does not use the question's exact wording or format" in messages[0].content
    assert "turn supported interview advice into do's and don'ts" in messages[0].content
    assert "Render every inline citation with ordinary parentheses only" in messages[0].content


def test_prompt_forbids_citing_generic_advice_that_is_absent_from_context() -> None:
    messages = ANSWER_PROMPT.invoke({"question": "What advice is given?", "context": "Advice"}).messages

    assert "Every claim in your answer must be directly traceable" in messages[0].content
    assert "do not attach a real citation to an unsupported nearby claim" in messages[0].content
    assert "arrive on time or dress appropriately" in messages[0].content


def test_prompt_requires_each_detail_in_a_sentence_to_be_independently_supported() -> None:
    messages = ANSWER_PROMPT.invoke({"question": "What advice is given?", "context": "Advice"}).messages

    assert "every distinct detail within a bullet or sentence must independently be supported" in messages[0].content
    assert "If you are not 100% certain" in messages[0].content
    assert "do not combine that with arriving on time" in messages[0].content
    assert "closed-off gestures or eye contact" in messages[0].content


def test_prompt_names_arrive_on_time_and_labeled_inferences_as_forbidden_when_unstated() -> None:
    messages = ANSWER_PROMPT.invoke({"question": "What advice is given?", "context": "Advice"}).messages

    assert "never write the exact phrase 'arrive on time'" in messages[0].content
    assert "do not pair it with handshake advice" in messages[0].content
    assert "Do not add inferences either" in messages[0].content
    assert "do not make an unstated conclusion grounded" in messages[0].content


def test_debug_context_is_opt_in(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setenv(DEBUG_CONTEXT_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(answer_module, "load_dotenv", lambda **_: False)
    monkeypatch.setattr(answer_module, "_invoke_groq", lambda *_: type("Response", (), {"content": "Answer"})())

    assert generate_grounded_answer("Question", [Document(page_content="Transcript text")]) == "Answer"
    output = capsys.readouterr().err
    assert "[RAG DEBUG] BEGIN GROQ QUESTION" in output
    assert "Transcript text" in output


def test_rag_completion_budget_is_large_enough_for_reasoning_and_an_answer() -> None:
    assert GROQ_MAX_TOKENS == 4096
    assert GROQ_REASONING_EFFORT == "low"


def test_prompt_allows_english_answers_from_non_english_context() -> None:
    messages = ANSWER_PROMPT.invoke({"question": "What does this say?", "context": "हिंदी"}).messages

    assert "answer in English when the question is in English" in messages[0].content
    assert "Never return a blank answer" in messages[0].content

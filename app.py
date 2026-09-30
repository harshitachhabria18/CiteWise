"""Tabbed Streamlit interface for the project's four grounded RAG sources."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import streamlit as st


# Allow `streamlit run app.py` from the repository root without requiring an
# editable package installation.
SRC_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from citewise.ingestion.github import GitHubIngestionError
from citewise.ingestion.hackernews import HackerNewsIngestionError
from citewise.ingestion.youtube import YouTubeIngestionError
from citewise.ingestion.youtube_channel import YouTubeChannelClient, YouTubeChannelError
from citewise.ingestion.wikipedia import WikipediaIngestionError
from citewise.retrieval.github_answer import AnswerGenerationError, generate_grounded_answer
from citewise.services.github_pipeline import ingest_github_repository
from citewise.services.hackernews_pipeline import (
    HACKERNEWS_COLLECTION,
    HACKERNEWS_DEFAULT_LIMIT,
    ingest_hackernews_topic,
)
from citewise.services.wikipedia_pipeline import (
    WIKIPEDIA_COLLECTION,
    WIKIPEDIA_DEFAULT_LIMIT,
    ingest_wikipedia_topic,
    retrieve_wikipedia_documents,
)
from citewise.services.youtube_pipeline import (
    YOUTUBE_CHANNEL_DEFAULT_LIMIT,
    YOUTUBE_COLLECTION,
    YOUTUBE_RETRIEVAL_K,
    ingest_youtube_channel,
    ingest_youtube_video,
)
from citewise.storage.chroma_store import LocalChromaStore


CitationRetriever = Callable[[str], list]
GITHUB_RETRIEVAL_K = 16
GITHUB_RETRIEVAL_CANDIDATES = 20


def _initialize_source_state(source: str) -> None:
    """Create state keys without sharing chat history between source tabs."""
    st.session_state.setdefault(f"{source}_chat_history", [])
    st.session_state.setdefault(f"{source}_ingested", False)
    st.session_state.setdefault(f"{source}_scope", None)


def _citations(documents: list) -> list[str]:
    """Return readable, de-duplicated source labels in retrieval order."""
    return list(
        dict.fromkeys(
            str(document.metadata.get("citation", "Unknown source")) for document in documents
        )
    )


def _show_sources(citations: list[str]) -> None:
    with st.expander("Sources"):
        if citations:
            for citation in citations:
                st.markdown(f"- {citation}")
        else:
            st.caption("No matching source chunks were retrieved.")


def _render_chat(source: str, retriever: CitationRetriever) -> None:
    """Render one tab's isolated chat history and grounded-answer interaction."""
    history_key = f"{source}_chat_history"
    for message in st.session_state[history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _show_sources(message.get("citations", []))

    question = st.chat_input("Ask a question about the ingested content", key=f"{source}_chat_input")
    if not question:
        return

    with st.spinner("Retrieving grounded context and generating an answer..."):
        try:
            documents = retriever(question)
            answer = generate_grounded_answer(question, documents)
        except AnswerGenerationError as error:
            st.error(str(error))
            return
        except Exception:
            st.error("Could not retrieve source context. Please try again.")
            return

    citations = _citations(documents)
    st.session_state[history_key].extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "citations": citations},
        ]
    )
    st.rerun()


def _github_tab() -> None:
    source = "github"
    _initialize_source_state(source)
    st.caption("Ask questions about a GitHub repository's issues, discussions, README, and code.")
    repository = st.text_input("Repository (owner/repo)", key="github_repository_input")
    limit = st.number_input("Issues and discussions limit", min_value=1, value=20, key="github_limit")

    if st.button("Ingest GitHub repository", key="github_ingest"):
        with st.spinner("Fetching, chunking, and storing GitHub content..."):
            try:
                report = ingest_github_repository(repository, limit=int(limit))
            except GitHubIngestionError as error:
                st.error(str(error))
            except Exception:
                st.error("GitHub ingestion could not complete. Check your connection and try again.")
            else:
                st.session_state["github_ingested"] = True
                st.session_state["github_scope"] = report.repository
                st.session_state["github_chat_history"] = []
                st.success(f"Stored {report.stored_chunks} chunks from {report.repository}.")
                for notice in report.notices:
                    st.warning(notice)

    if st.session_state["github_ingested"]:
        _render_chat(
            source,
            lambda question: LocalChromaStore().similarity_search(
                question,
                k=GITHUB_RETRIEVAL_K,
                repository=st.session_state["github_scope"],
                ensure_source_type="github_readme",
                candidate_k=GITHUB_RETRIEVAL_CANDIDATES,
            ),
        )


def _youtube_tab() -> None:
    source = "youtube"
    _initialize_source_state(source)
    st.caption("Ask questions about a YouTube video transcript or a channel's recent videos.")
    video_or_channel = st.text_input("Video URL/ID or channel handle (@handle)", key="youtube_input")
    limit = st.number_input(
        "Recent videos limit (used for channels)",
        min_value=1,
        value=YOUTUBE_CHANNEL_DEFAULT_LIMIT,
        key="youtube_limit",
    )

    if st.button("Ingest YouTube content", key="youtube_ingest"):
        with st.spinner("Fetching captions, chunking, and storing YouTube content..."):
            try:
                if YouTubeChannelClient.is_channel_reference(video_or_channel):
                    report = ingest_youtube_channel(video_or_channel, limit=int(limit))
                    video_ids = [video_report.video.video_id for video_report in report.ingested_videos]
                    summary = (
                        f"Stored {report.stored_chunks} chunks from {len(video_ids)} videos in "
                        f"{report.channel_title}."
                    )
                    notices = list(report.skipped_videos)
                else:
                    report = ingest_youtube_video(video_or_channel)
                    video_ids = [report.video.video_id]
                    summary = f"Stored {report.stored_chunks} chunks from {report.video.title}."
                    notices = [report.video.caption_selection_notice] if report.video.caption_selection_notice else []
            except (YouTubeIngestionError, YouTubeChannelError) as error:
                st.error(str(error))
            except Exception:
                st.error("YouTube ingestion could not complete. Check your connection and try again.")
            else:
                st.session_state["youtube_ingested"] = bool(video_ids)
                st.session_state["youtube_scope"] = video_ids
                st.session_state["youtube_chat_history"] = []
                st.success(summary)
                for notice in notices:
                    st.warning(notice)
                if not video_ids:
                    st.warning("No videos with usable transcripts were stored, so there is nothing to query yet.")

    if st.session_state["youtube_ingested"]:
        _render_chat(
            source,
            lambda question: LocalChromaStore(collection_name=YOUTUBE_COLLECTION).similarity_search(
                question,
                k=YOUTUBE_RETRIEVAL_K,
                metadata_filter={"video_id": {"$in": st.session_state["youtube_scope"]}},
            ),
        )


def _hackernews_tab() -> None:
    source = "hackernews"
    _initialize_source_state(source)
    st.caption("Ask questions about Hacker News stories and their discussions for a topic.")
    topic = st.text_input("Topic or search term", key="hackernews_topic_input")
    limit = st.number_input(
        "Story limit", min_value=1, value=HACKERNEWS_DEFAULT_LIMIT, key="hackernews_limit"
    )

    if st.button("Ingest Hacker News", key="hackernews_ingest"):
        with st.spinner("Searching, fetching discussions, and storing Hacker News content..."):
            try:
                report = ingest_hackernews_topic(topic, limit=int(limit))
            except HackerNewsIngestionError as error:
                st.error(str(error))
            except Exception:
                st.error("Hacker News ingestion could not complete. Check your connection and try again.")
            else:
                st.session_state["hackernews_ingested"] = True
                st.session_state["hackernews_scope"] = [story.id for story in report.stories]
                st.session_state["hackernews_chat_history"] = []
                st.success(
                    f"Stored {report.stored_chunks} chunks from {len(report.stories)} Hacker News stories."
                )

    if st.session_state["hackernews_ingested"]:
        _render_chat(
            source,
            lambda question: LocalChromaStore(collection_name=HACKERNEWS_COLLECTION).similarity_search(
                question,
                metadata_filter={"story_id": {"$in": st.session_state["hackernews_scope"]}},
            ),
        )


def _wikipedia_tab() -> None:
    source = "wikipedia"
    _initialize_source_state(source)
    st.caption("Ask questions grounded in English Wikipedia articles for a topic.")
    topic = st.text_input("Topic or search term", key="wikipedia_topic_input")
    limit = st.number_input(
        "Article limit", min_value=1, value=WIKIPEDIA_DEFAULT_LIMIT, key="wikipedia_limit"
    )

    if st.button("Ingest Wikipedia", key="wikipedia_ingest"):
        with st.spinner("Searching, chunking sections, and storing Wikipedia content..."):
            try:
                report = ingest_wikipedia_topic(topic, limit=int(limit))
            except WikipediaIngestionError as error:
                st.error(str(error))
            except Exception:
                st.error("Wikipedia ingestion could not complete. Check your connection and try again.")
            else:
                st.session_state["wikipedia_ingested"] = True
                st.session_state["wikipedia_scope"] = {
                    "topic": report.topic,
                    "articles": report.articles,
                }
                st.session_state["wikipedia_chat_history"] = []
                st.success(
                    f"Stored {report.stored_chunks} chunks from {len(report.articles)} Wikipedia articles."
                )

    if st.session_state["wikipedia_ingested"]:
        _render_chat(
            source,
            lambda question: retrieve_wikipedia_documents(
                LocalChromaStore(collection_name=WIKIPEDIA_COLLECTION),
                st.session_state["wikipedia_scope"]["topic"],
                st.session_state["wikipedia_scope"]["articles"],
                question,
            ),
        )


def main() -> None:
    st.set_page_config(page_title="CiteWise", page_icon="🔎", layout="wide")
    st.title("CiteWise")
    st.write("Ingest public source content, then ask citation-grounded questions in each independent tab.")

    github_tab, youtube_tab, hackernews_tab, wikipedia_tab = st.tabs(
        ["GitHub", "YouTube", "Hacker News", "Wikipedia"]
    )
    with github_tab:
        _github_tab()
    with youtube_tab:
        _youtube_tab()
    with hackernews_tab:
        _hackernews_tab()
    with wikipedia_tab:
        _wikipedia_tab()


if __name__ == "__main__":
    main()

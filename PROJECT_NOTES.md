# CiteWise — Project Notes

## What this project is

A learning RAG application that lets a user ask grounded questions about public content from
several sources. Wikipedia, Hacker News, YouTube, and GitHub are supported; podcasts may be added later.

## Built so far

- A `src/citewise` package split by configuration, ingestion, processing, storage,
  retrieval, and UI responsibilities.
- A project-local Python virtual environment and a dependency list.
- `.env.example`, an ignored local `.env`, and a configuration loader for Groq credentials.
- A Hacker News ingestion client that searches a topic through HN Algolia, then fetches the
  canonical story, top-level comments, and one bounded reply level from HN's public Firebase API.
- A Hacker News pipeline that chunks, embeds, stores, retrieves, and produces grounded answers
  with exact Hacker News citations.
- A Wikipedia pipeline that searches the public Wikipedia Action API, retrieves official article
  HTML, preserves visible section boundaries, and stores section-cited chunks in its own Chroma
  collection.

## Important decisions

- **Reddit replaced by Hacker News:** Reddit application registration is blocked by a persistent
  reCAPTCHA issue, and its unauthenticated JSON fallback was unreliable. Hacker News is now the
  supported discussion source because its public API needs no account or API key. The old
  `reddit_json.py` and `reddit_praw.py` files are retained only as clearly marked deprecated
  historical reference and are not used by any active pipeline.
- **Hacker News ingestion:** Topic searches use the public HN Algolia endpoint because the
  official Firebase API has no full-text search. Each matching story is then fetched from the
  canonical Firebase item endpoint along with up to 10 top-level comments and up to 3 direct
  replies per comment. `--limit` defaults to 10 stories so a broad topic cannot accidentally
  ingest an unbounded result set. Story, comment, and reply chunks use 900 characters with 150
  characters of overlap. Citations read `Hacker News: story title (comment by author) - HN URL`
  (with equivalent story/reply labels), preserving the exact HN item link for verification.
- **Wikipedia ingestion:** Topic searches use the free, keyless English Wikipedia Action API;
  article content comes from Wikipedia's official REST HTML endpoint so visible headings become
  natural section boundaries before long sections are split into 900-character chunks with
  150-character overlap. `--limit` defaults to 4 articles: full articles contain much more
  material than HN discussions, so a small result set is usually sufficient and avoids broad,
  expensive imports. Chunks live in the `wikipedia_content` collection and cite
  `Wikipedia: article title - section name - article URL`. Wikipedia provides broad coverage and
  a useful grounding baseline, though it has less of a "the LLM could not otherwise know this"
  advantage than the other sources because general LLMs have already been extensively trained on
  Wikipedia. A live MS Dhoni retrieval revealed that References and External links could dominate
  results with citation-list text and leave Groq with unusable context. Before chunking, the
  Wikipedia pipeline now excludes standard non-content sections case-insensitively: References,
  External links, See also, Further reading, Bibliography, and Notes. They are never embedded or
  stored.
- **Local embeddings:** Sentence Transformers with `all-MiniLM-L6-v2` will create embeddings
  locally, avoiding embedding API costs.
- **Vector store:** Chroma will run locally under `data/chroma`, so no hosted database is
  needed for this learning project.
- **Answer generation:** Groq will call a Llama-family model for fast, low-cost grounded
  answers after retrieved context is available.
- **GitHub discussions:** The repository Discussions REST request is supported. A `410 Gone`
  response means Discussions are disabled for that specific repository, not that GraphQL is
  required. GitHub ingestion treats this as optional and skips it with the message
  `Discussions are disabled for this repository, skipping.` Repositories with Discussions
  enabled, such as `langchain-ai/langchain`, return discussion data through REST normally.
- **GitHub access:** `GITHUB_TOKEN` is optional. When present in `.env`, it is sent as a
  Bearer token; otherwise, the client continues with GitHub's lower unauthenticated limit and
  reports a warning.
- **GitHub README and code:** In addition to issues and discussions, ingestion fetches the
  README and recursively fetches eligible UTF-8 code files through the Contents API. Lock and
  configuration files, generated/build folders, non-code extensions, and binary files are
  deliberately excluded. Every eligible code file is fetched in full: files at or below 400
  lines become one chunk, while larger files are split through the same chunking pipeline
  (with language-aware separators where available). There are no silent truncations of an
  included file. The tradeoff is that questions about a large whole file may retrieve only its
  relevant fragments instead of one coherent file view. This is not code-aware RAG; AST-based
  parsing is a possible future enhancement and is not implemented now.
- **GitHub chunking and citations:** Issues, discussions, and README use 900-character chunks
  with 150 characters of overlap. All chunks retain source type, repository, URL, chunk index,
  and a readable citation: `GitHub Issue #…`, `GitHub Discussion #…`, `GitHub README: …`, or
  `GitHub File: path (owner/repo)`.
- **Grounded GitHub answers:** After Chroma retrieves the most relevant chunks, the pipeline
  calls Groq through LangChain using `GROQ_API_KEY` from the environment or Streamlit secrets.
  Each prompt chunk is headed by
  its readable citation in brackets (for example, `[GitHub File: src/click/core.py
  (pallets/click)]`), and the model is explicitly told to cite those exact labels rather than
  generic placeholders such as `[Source 1]`. The prompt permits answers only from retrieved
  context, requires inline source labels, and uses a fixed fallback sentence when the context
  does not contain the answer. Groq API, authentication, and rate-limit failures are shown as
  readable messages; the raw retrieved citations remain visible below every generated answer
  for grounding checks.
- **Single-video YouTube transcripts:** A YouTube URL or 11-character video ID is normalized,
  then `youtube-transcript-api` lists caption tracks without an API key. With no `--language`
  override, English (including regional English) is preferred; otherwise the first available
  manual or auto-generated track is used with a visible notice. An explicit language override
  remains strict: if captions exist but not in that language, the error lists every available
  language code and whether each track is manual or auto-generated. No captions in any language,
  unavailable/private videos, and network or YouTube-blocking failures have separate readable
  messages. The public oEmbed endpoint supplies the video title. Caption text is split
  into 1,000-character chunks with 150 characters of overlap; each chunk retains the start time
  of its first caption segment and cites `YouTube: title @ M:SS (timestamp URL)`. YouTube chunks
  live in their own Chroma collection and retrieval is filtered by `video_id`, so captions from
  another video cannot appear in a single-video answer.
- **YouTube channel transcripts:** The YouTube pipeline also accepts a channel handle (such as
  `@GoogleDevelopers`) or a canonical channel URL. It uses `yt-dlp` to discover the channel's
  most recent videos, then runs each one through the existing single-video transcript, chunking,
  embedding, and storage workflow. `--limit` defaults to 10 videos so a channel's full history
  is never ingested accidentally; it can be raised deliberately for larger imports. Videos with
  no available captions are skipped with a readable message while the rest continue. Each chunk
  keeps the existing per-video ID, title, timestamp URL, and citation metadata, so channel-wide
  storage remains attributable to the exact video and moment. This approach does not use the
  YouTube Data API, so it consumes no Data API daily quota; normal YouTube access limits or
  blocking can still prevent discovery or transcript download.
- **Multilingual grounded answers:** Groq is instructed to answer in the question's language
  while using only retrieved source facts, so an English question can be answered from Hindi or
  other non-English captions. A blank provider response is treated as an explicit answer-
  generation error and logged, rather than leaving an empty `Grounded answer` section.
- **Blank Groq answers:** The shared answer prompt explicitly asks for a concise partial answer
  when thin context still has relevant facts and forbids blank output. If a provider still returns
  blank content, every source pipeline now shows the fixed insufficiency message
  `I couldn't find that answer in the retrieved source context.` instead of an empty answer.
- **Claim-level grounding limitation:** The answer prompt requires every distinct sentence detail
  to be explicit in retrieved context and forbids generic advice or labeled inferences. This is a
  prompt-level guardrail rather than a formal citation verifier, so important answers should still
  be checked against the displayed retrieved context.
- **Groq context and fallback key:** Before answer generation, retrieved chunks are kept in
  relevance order but their labeled combined context is capped at 12,000 characters. This keeps
  larger `--k` retrieval requests below Groq's free-tier request budget while preserving the
  best matches first. `GROQ_API_KEY_2` is optional in `.env`; it is tried once only after the
  primary `GROQ_API_KEY` receives an HTTP 429 rate limit. It does not apply to request-size
  errors such as HTTP 413.
- **Deployment secrets:** Credentials are read through a shared helper that checks environment
  variables first (including local `.env` values), then `st.secrets`. This keeps local use and
  Streamlit Cloud deployment compatible without committing a secrets file.
- **GitHub retrieval diversity:** Semantic ranking can favor topically similar short discussion
  posts over a more substantive README section. GitHub retrieval therefore considers the top 20
  candidates, returns up to 16 chunks, and reserves the best README candidate when one appears
  in that pool. This is a lightweight diversity guardrail, not a full re-ranking system; a README
  can still be absent when it is not among the semantic candidates.
- **Citation rendering:** The model is asked to use ordinary parenthesized inline citations. A
  response-boundary normalizer converts full-width or square source-label brackets to that form,
  so Streamlit does not display stray `【` or `】` markers.
- **Non-English retrieval limitation (verified):** Non-English content retrieval is unreliable
  with the current English-optimized `all-MiniLM-L6-v2` embedding model. Direct testing
  confirmed that the Hindi target video has 31+ stored chunks containing relevant terms, yet
  none surfaced even at `k=20` for a semantically matching English query. This rules out the
  retrieval depth as the cause. A multilingual embedding model such as
  `paraphrase-multilingual-MiniLM-L12-v2`, followed by re-embedding the YouTube collection,
  would likely be required for reliable non-English support. This is a future enhancement and
  is not implemented in this version.
- **Streamlit interface:** `app.py` is a single tabbed interface launched with
  `streamlit run app.py`. Its GitHub, YouTube, Hacker News, and Wikipedia tabs each have their
  own ingestion controls, scoped chat retrieval, citation expander, and prefixed session-state
  keys (for example, `github_chat_history` and `wikipedia_scope`). Ingesting or chatting in one
  tab therefore does not clear another tab's source scope or chat history.

## Still to build

1. Add podcasts as another public-content source.
2. Consider a multilingual embedding model and re-embed YouTube content for reliable
   non-English retrieval.

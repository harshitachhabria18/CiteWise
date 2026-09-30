"""Minimal direct Groq connectivity and response-content check.

Run from the repository root with:
    .\\.venv\\Scripts\\python.exe test_groq.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


SRC_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from citewise.config.settings import ENV_FILE, get_secret
from citewise.retrieval.github_answer import GROQ_MODEL


def main() -> None:
    load_dotenv(dotenv_path=ENV_FILE)
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        print("RESULT: GROQ_API_KEY is missing from .env.")
        raise SystemExit(1)

    print(f"Model: {GROQ_MODEL}")
    print("Sending direct Groq test prompt...")
    try:
        response = Groq(api_key=api_key).chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            temperature=0,
        )
    except Exception as error:
        print(f"RESULT: EXCEPTION ({type(error).__name__}): {error}")
        traceback.print_exc()
        raise SystemExit(1) from error

    content = response.choices[0].message.content if response.choices else None
    print("\nRaw response object:")
    print(repr(response))
    print("\nExtracted content:")
    print(repr(content))
    if isinstance(content, str) and content.strip():
        print("\nRESULT: REAL TEXT")
    else:
        print("\nRESULT: EMPTY CONTENT")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

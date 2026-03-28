"""Google Gemini API key resolution and base URL.

Simple API-key auth via GEMINI_API_KEY / GOOGLE_API_KEY env vars.
OAuth can be added later once a GCP Desktop OAuth client is registered.
"""

from __future__ import annotations

import os
from typing import Optional

# Gemini API base URL (OpenAI-compatible)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def resolve_gemini_api_key() -> Optional[str]:
    """Return a Gemini API key from environment variables, or None."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    return None


# Alias used by runtime_provider.py
resolve_gemini_token = resolve_gemini_api_key

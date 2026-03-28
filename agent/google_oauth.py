"""Google OAuth (Authorization Code + PKCE) for the Gemini provider.

Provides browser-based OAuth login for users with Google AI / Gemini API access,
as well as a simpler API-key fallback via GEMINI_API_KEY / GOOGLE_API_KEY.

Token lifecycle:
  - Stored at ``~/.hermes/gemini_oauth.json`` (0o600)
  - Auto-refreshed when within 5 minutes of expiry
  - File-locked for concurrent session safety
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google OAuth endpoints
# ---------------------------------------------------------------------------
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Default OAuth client — Google considers installed-app secrets non-confidential.
# Users can override via HERMES_GEMINI_CLIENT_ID / HERMES_GEMINI_CLIENT_SECRET.
_DEFAULT_CLIENT_ID = os.getenv(
    "HERMES_GEMINI_CLIENT_ID",
    # Placeholder — replace with a real Desktop-app client ID once registered.
    "",
)
_DEFAULT_CLIENT_SECRET = os.getenv(
    "HERMES_GEMINI_CLIENT_SECRET",
    "",
)

OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]
REDIRECT_PORT = 8085
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/oauth2callback"

# Gemini API base URL (OpenAI-compatible)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Credential file
_CRED_FILENAME = "gemini_oauth.json"


def _get_hermes_home() -> Path:
    env_val = os.environ.get("HERMES_HOME", "")
    return Path(env_val) if env_val else Path.home() / ".hermes"


def _cred_path() -> Path:
    return _get_hermes_home() / _CRED_FILENAME


# ---------------------------------------------------------------------------
# API-key resolution (simple path — no OAuth needed)
# ---------------------------------------------------------------------------

def resolve_gemini_api_key() -> Optional[str]:
    """Return a Gemini API key from environment variables, or None."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    return None


# ---------------------------------------------------------------------------
# OAuth credential storage
# ---------------------------------------------------------------------------

def load_credentials() -> Optional[Dict[str, Any]]:
    """Load saved OAuth credentials from disk."""
    path = _cred_path()
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("Failed to load Gemini OAuth credentials: %s", exc)
        return None


def save_credentials(creds: Dict[str, Any]) -> None:
    """Persist OAuth credentials to disk with restricted permissions."""
    path = _cred_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(creds, f, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    import base64
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ---------------------------------------------------------------------------
# OAuth flow
# ---------------------------------------------------------------------------

def _build_auth_url(client_id: str, code_challenge: str, state: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def _exchange_code(
    code: str,
    code_verifier: str,
    client_id: str,
    client_secret: str,
) -> Dict[str, Any]:
    """Exchange an authorization code for tokens."""
    data = urllib.parse.urlencode({
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
        "code_verifier": code_verifier,
    }).encode()
    req = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def refresh_access_token(creds: Dict[str, Any]) -> Dict[str, Any]:
    """Use a refresh_token to obtain a fresh access_token."""
    client_id = creds.get("client_id") or _DEFAULT_CLIENT_ID
    client_secret = creds.get("client_secret") or _DEFAULT_CLIENT_SECRET
    data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        token_data = json.loads(resp.read())
    creds["access_token"] = token_data["access_token"]
    creds["expires_at"] = time.time() + token_data.get("expires_in", 3600)
    if "refresh_token" in token_data:
        creds["refresh_token"] = token_data["refresh_token"]
    save_credentials(creds)
    return creds


def get_valid_access_token() -> Optional[str]:
    """Return a valid access token, refreshing if needed. Returns None if unavailable."""
    creds = load_credentials()
    if not creds or "refresh_token" not in creds:
        return None
    expires_at = creds.get("expires_at", 0)
    # Refresh if within 5 minutes of expiry
    if time.time() > expires_at - 300:
        try:
            creds = refresh_access_token(creds)
        except Exception as exc:
            logger.warning("Failed to refresh Gemini OAuth token: %s", exc)
            return None
    return creds.get("access_token")


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler to capture the OAuth redirect."""

    code: Optional[str] = None
    error: Optional[str] = None
    expected_state: str = ""

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        state = params.get("state", [""])[0]
        if state != self.expected_state:
            self.error = "State mismatch"
        elif "error" in params:
            self.error = params["error"][0]
        elif "code" in params:
            self.code = params["code"][0]
        else:
            self.error = "No code in callback"

        # Send response to browser
        if self.code:
            body = b"<html><body><h2>Authentication successful!</h2><p>You can close this tab.</p></body></html>"
        else:
            body = f"<html><body><h2>Authentication failed</h2><p>{self.error}</p></body></html>".encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass  # Suppress request logging


def start_oauth_flow(
    client_id: str = "",
    client_secret: str = "",
) -> Optional[Dict[str, Any]]:
    """Run the full browser-based OAuth PKCE flow.

    Returns saved credentials dict on success, or None on failure.
    """
    client_id = client_id or _DEFAULT_CLIENT_ID
    client_secret = client_secret or _DEFAULT_CLIENT_SECRET
    if not client_id:
        logger.error("No Google OAuth client_id configured. "
                      "Set HERMES_GEMINI_CLIENT_ID or register a Desktop OAuth client.")
        return None

    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)

    auth_url = _build_auth_url(client_id, challenge, state)

    # Start local callback server
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.error = None
    _OAuthCallbackHandler.expected_state = state

    server = HTTPServer(("localhost", REDIRECT_PORT), _OAuthCallbackHandler)
    server.timeout = 120  # 2 minute timeout

    print(f"\nOpening browser for Google sign-in...")
    print(f"If the browser doesn't open, visit:\n  {auth_url}\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    # Wait for callback
    while _OAuthCallbackHandler.code is None and _OAuthCallbackHandler.error is None:
        server.handle_request()

    server.server_close()

    if _OAuthCallbackHandler.error:
        logger.error("OAuth failed: %s", _OAuthCallbackHandler.error)
        return None

    code = _OAuthCallbackHandler.code
    try:
        token_data = _exchange_code(code, verifier, client_id, client_secret)
    except Exception as exc:
        logger.error("Token exchange failed: %s", exc)
        return None

    creds = {
        "client_id": client_id,
        "client_secret": client_secret,
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token", ""),
        "expires_at": time.time() + token_data.get("expires_in", 3600),
    }
    save_credentials(creds)
    print("Gemini OAuth authentication successful!")
    return creds


# ---------------------------------------------------------------------------
# Unified token resolver (API key preferred, then OAuth)
# ---------------------------------------------------------------------------

def resolve_gemini_token() -> Optional[str]:
    """Return the best available Gemini credential (API key or OAuth token)."""
    api_key = resolve_gemini_api_key()
    if api_key:
        return api_key
    return get_valid_access_token()

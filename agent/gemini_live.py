"""Gemini Live API client — real-time bidirectional voice via WebSocket.

Connects to ``wss://generativelanguage.googleapis.com/ws/...`` and streams
16-bit PCM audio in both directions.  Designed to hook into the existing
AudioRecorder (16 kHz input) and sounddevice (24 kHz output).

Usage::

    session = GeminiLiveSession(api_key="...", model="gemini-3.1-flash-live-preview")
    await session.connect()
    await session.send_audio(pcm_16khz_bytes)
    # responses arrive via callbacks
    await session.disconnect()
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
_WS_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# Audio specs matching the Live API requirements
INPUT_SAMPLE_RATE = 16000   # 16 kHz mono 16-bit PCM (matches AudioRecorder)
OUTPUT_SAMPLE_RATE = 24000  # 24 kHz mono 16-bit PCM (API output)
AUDIO_MIME_INPUT = "audio/pcm;rate=16000"
AUDIO_MIME_OUTPUT = "audio/pcm;rate=24000"

# Chunk size for streaming audio input (~100ms of 16kHz 16-bit mono)
INPUT_CHUNK_BYTES = INPUT_SAMPLE_RATE * 2 // 10  # 3200 bytes = 100ms


class GeminiLiveSession:
    """Manages a single Gemini Live API WebSocket session."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-live-preview",
        system_instruction: str = "",
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,
        on_output_transcript: Optional[Callable[[str], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
        on_turn_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction

        # Callbacks
        self.on_audio = on_audio
        self.on_text = on_text
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.on_interrupted = on_interrupted
        self.on_turn_complete = on_turn_complete
        self.on_error = on_error

        self._ws: Any = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket connection and send the setup message."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required for Gemini Live. "
                "Install with: pip install websockets"
            )

        url = f"{_WS_ENDPOINT}?key={self.api_key}"
        self._ws = await websockets.connect(
            url,
            additional_headers={"Content-Type": "application/json"},
            max_size=None,  # No limit on message size
            ping_interval=20,
            ping_timeout=10,
        )

        # Send setup message
        setup_msg = {
            "setup": {
                "model": f"models/{self.model}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Aoede",
                            }
                        }
                    },
                },
            }
        }

        if self.system_instruction:
            setup_msg["setup"]["systemInstruction"] = {
                "parts": [{"text": self.system_instruction}]
            }

        await self._ws.send(json.dumps(setup_msg))

        # Wait for setup confirmation
        raw = await self._ws.recv()
        resp = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
        if "setupComplete" not in resp:
            error_msg = resp.get("error", {}).get("message", "Setup failed")
            raise ConnectionError(f"Gemini Live setup failed: {error_msg}")

        self._connected = True
        logger.info("Gemini Live session connected (model=%s)", self.model)

        # Start background receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Gemini Live session disconnected")

    # ------------------------------------------------------------------
    # Sending data
    # ------------------------------------------------------------------

    async def send_audio(self, pcm_data: bytes) -> None:
        """Send a chunk of 16 kHz 16-bit PCM audio to the model."""
        if not self._connected or not self._ws:
            return
        encoded = base64.b64encode(pcm_data).decode("ascii")
        msg = {
            "realtimeInput": {
                "mediaChunks": [{
                    "data": encoded,
                    "mimeType": AUDIO_MIME_INPUT,
                }]
            }
        }
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as exc:
            logger.debug("Failed to send audio: %s", exc)

    async def send_text(self, text: str) -> None:
        """Send a text message to the model."""
        if not self._connected or not self._ws:
            return
        msg = {
            "clientContent": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": text}]
                }],
                "turnComplete": True,
            }
        }
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as exc:
            logger.debug("Failed to send text: %s", exc)

    async def send_tool_response(
        self, function_responses: List[Dict[str, Any]]
    ) -> None:
        """Send tool/function call responses back to the model."""
        if not self._connected or not self._ws:
            return
        msg = {
            "toolResponse": {
                "functionResponses": function_responses,
            }
        }
        await self._ws.send(json.dumps(msg))

    # ------------------------------------------------------------------
    # Receiving data
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Background task that reads WebSocket messages and dispatches callbacks."""
        try:
            async for raw in self._ws:
                if not self._connected:
                    break
                try:
                    msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                self._handle_message(msg)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Gemini Live receive error: %s", exc)
            if self.on_error:
                self.on_error(str(exc))
            self._connected = False

    def _handle_message(self, msg: dict) -> None:
        """Dispatch a single server message to the appropriate callback."""
        server_content = msg.get("serverContent")
        if server_content:
            # Model turn — contains audio/text parts
            model_turn = server_content.get("modelTurn")
            if model_turn:
                for part in model_turn.get("parts", []):
                    # Audio data
                    inline_data = part.get("inlineData")
                    if inline_data and self.on_audio:
                        audio_bytes = base64.b64decode(inline_data.get("data", ""))
                        if audio_bytes:
                            self.on_audio(audio_bytes)
                    # Text data
                    text = part.get("text")
                    if text and self.on_text:
                        self.on_text(text)

            # Input transcription (what the user said)
            input_transcript = server_content.get("inputTranscription")
            if input_transcript and self.on_input_transcript:
                text = input_transcript.get("text", "")
                if text:
                    self.on_input_transcript(text)

            # Output transcription (what the model said)
            output_transcript = server_content.get("outputTranscription")
            if output_transcript and self.on_output_transcript:
                text = output_transcript.get("text", "")
                if text:
                    self.on_output_transcript(text)

            # Turn complete
            if server_content.get("turnComplete") and self.on_turn_complete:
                self.on_turn_complete()

            # Interrupted (barge-in)
            if server_content.get("interrupted") and self.on_interrupted:
                self.on_interrupted()

        # Tool calls
        tool_call = msg.get("toolCall")
        if tool_call:
            logger.debug("Gemini Live tool call: %s", tool_call)

        # Error
        error = msg.get("error")
        if error and self.on_error:
            self.on_error(error.get("message", str(error)))


# ---------------------------------------------------------------------------
# Synchronous wrapper for CLI integration
# ---------------------------------------------------------------------------

class GeminiLiveVoiceSession:
    """High-level synchronous wrapper that bridges AudioRecorder ↔ Gemini Live.

    Runs the async WebSocket session in a dedicated event loop thread, and
    exposes simple start/stop/feed_audio methods for the CLI.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-live-preview",
        system_instruction: str = "",
        on_output_audio: Optional[Callable[[bytes], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,
        on_output_transcript: Optional[Callable[[str], None]] = None,
        on_turn_complete: Optional[Callable[[], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction

        # External callbacks (called from the event loop thread)
        self._on_output_audio = on_output_audio
        self._on_input_transcript = on_input_transcript
        self._on_output_transcript = on_output_transcript
        self._on_turn_complete = on_turn_complete
        self._on_interrupted = on_interrupted
        self._on_error = on_error

        self._session: Optional[GeminiLiveSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._active = False

        # Audio output buffer for real-time playback
        self._audio_buffer = bytearray()
        self._audio_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Start the live voice session in a background thread."""
        if self._active:
            return

        self._active = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        """Run the async event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_session())
        except Exception as exc:
            logger.error("Gemini Live session error: %s", exc)
            if self._on_error:
                self._on_error(str(exc))
        finally:
            self._active = False
            self._loop.close()

    async def _async_session(self) -> None:
        """Manage the async session lifecycle."""
        self._session = GeminiLiveSession(
            api_key=self.api_key,
            model=self.model,
            system_instruction=self.system_instruction,
            on_audio=self._handle_audio,
            on_input_transcript=self._on_input_transcript,
            on_output_transcript=self._on_output_transcript,
            on_turn_complete=self._on_turn_complete,
            on_interrupted=self._handle_interrupted,
            on_error=self._on_error,
        )

        await self._session.connect()

        # Keep alive until stopped
        while self._active and self._session.connected:
            await asyncio.sleep(0.1)

        if self._session.connected:
            await self._session.disconnect()

    def _handle_audio(self, pcm_data: bytes) -> None:
        """Buffer incoming audio and forward to callback."""
        if self._on_output_audio:
            self._on_output_audio(pcm_data)
        with self._audio_lock:
            self._audio_buffer.extend(pcm_data)

    def _handle_interrupted(self) -> None:
        """Clear audio buffer on barge-in and forward callback."""
        with self._audio_lock:
            self._audio_buffer.clear()
        if self._on_interrupted:
            self._on_interrupted()

    def feed_audio(self, pcm_data: bytes) -> None:
        """Feed raw 16 kHz 16-bit PCM audio from the microphone.

        Thread-safe — can be called from the AudioRecorder callback.
        """
        if not self._active or not self._loop or not self._session:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._session.send_audio(pcm_data), self._loop
            )
        except Exception:
            pass

    def send_text(self, text: str) -> None:
        """Send a text message to the live session."""
        if not self._active or not self._loop or not self._session:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._session.send_text(text), self._loop
            )
        except Exception:
            pass

    def stop(self) -> None:
        """Stop the live voice session."""
        self._active = False
        if self._loop and self._session:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._session.disconnect(), self._loop
                )
                future.result(timeout=5)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._session = None
        self._loop = None

    def drain_audio(self) -> bytes:
        """Drain and return buffered output audio. Thread-safe."""
        with self._audio_lock:
            data = bytes(self._audio_buffer)
            self._audio_buffer.clear()
        return data

"""Gemini Live API client — real-time bidirectional voice via WebSocket.

Connects to the Gemini Live API and streams 16-bit PCM audio in both
directions.  Input: 16 kHz mono.  Output: 24 kHz mono.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_WS_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
AUDIO_MIME_INPUT = "audio/pcm;rate=16000"


class GeminiLiveSession:
    """Async WebSocket session for the Gemini Live API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-live-preview",
        system_instruction: str = "",
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,
        on_output_transcript: Optional[Callable[[str], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
        on_turn_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction
        self.on_audio = on_audio
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

    async def connect(self) -> None:
        import websockets

        url = f"{_WS_ENDPOINT}?key={self.api_key}"
        self._ws = await websockets.connect(url, max_size=None, ping_interval=20)

        setup_msg: dict = {
            "setup": {
                "model": f"models/{self.model}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": "Aoede"}
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
        raw = await self._ws.recv()
        resp = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
        if "setupComplete" not in resp:
            raise ConnectionError(f"Gemini Live setup failed: {resp}")

        self._connected = True
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        self._connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_audio(self, pcm_data: bytes) -> None:
        if not self._connected or not self._ws:
            return
        msg = {
            "realtimeInput": {
                "mediaChunks": [{
                    "data": base64.b64encode(pcm_data).decode("ascii"),
                    "mimeType": AUDIO_MIME_INPUT,
                }]
            }
        }
        try:
            await self._ws.send(json.dumps(msg))
        except Exception:
            pass

    async def send_text(self, text: str) -> None:
        if not self._connected or not self._ws:
            return
        msg = {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": True,
            }
        }
        try:
            await self._ws.send(json.dumps(msg))
        except Exception:
            pass

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                if not self._connected:
                    break
                try:
                    msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.on_error:
                self.on_error(str(exc))
            self._connected = False

    def _dispatch(self, msg: dict) -> None:
        sc = msg.get("serverContent")
        if sc:
            mt = sc.get("modelTurn")
            if mt:
                for part in mt.get("parts", []):
                    inline = part.get("inlineData")
                    if inline and self.on_audio:
                        data = base64.b64decode(inline.get("data", ""))
                        if data:
                            self.on_audio(data)
            it = sc.get("inputTranscription")
            if it and self.on_input_transcript:
                t = it.get("text", "")
                if t:
                    self.on_input_transcript(t)
            ot = sc.get("outputTranscription")
            if ot and self.on_output_transcript:
                t = ot.get("text", "")
                if t:
                    self.on_output_transcript(t)
            if sc.get("turnComplete") and self.on_turn_complete:
                self.on_turn_complete()
            if sc.get("interrupted") and self.on_interrupted:
                self.on_interrupted()

        err = msg.get("error")
        if err and self.on_error:
            self.on_error(err.get("message", str(err)))


# ---------------------------------------------------------------------------
# Sync wrapper — runs the async session in a background thread
# ---------------------------------------------------------------------------

class GeminiLiveVoiceSession:
    """Synchronous wrapper for CLI use. Call feed_audio() from any thread."""

    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-live-preview",
                 system_instruction: str = "", **callbacks) -> None:
        self._api_key = api_key
        self._model = model
        self._system_instruction = system_instruction
        self._callbacks = callbacks
        self._session: Optional[GeminiLiveSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:
            on_err = self._callbacks.get("on_error")
            if on_err:
                on_err(str(exc))
        finally:
            self._active = False
            self._loop.close()

    async def _main(self) -> None:
        self._session = GeminiLiveSession(
            api_key=self._api_key,
            model=self._model,
            system_instruction=self._system_instruction,
            **self._callbacks,
        )
        await self._session.connect()
        while self._active and self._session.connected:
            await asyncio.sleep(0.1)
        if self._session.connected:
            await self._session.disconnect()

    def feed_audio(self, pcm_data: bytes) -> None:
        if self._active and self._loop and self._session:
            asyncio.run_coroutine_threadsafe(self._session.send_audio(pcm_data), self._loop)

    def send_text(self, text: str) -> None:
        if self._active and self._loop and self._session:
            asyncio.run_coroutine_threadsafe(self._session.send_text(text), self._loop)

    def stop(self) -> None:
        self._active = False
        if self._loop and self._session:
            try:
                asyncio.run_coroutine_threadsafe(self._session.disconnect(), self._loop).result(timeout=5)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        self._session = None
        self._loop = None
        self._thread = None

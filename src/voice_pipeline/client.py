from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import replace

import websockets
from websockets.exceptions import ConnectionClosed

from voice_pipeline.config import PipelineConfig

logger = logging.getLogger("voice_pipeline.client")


class Protocol:
    @staticmethod
    def setup_payload(config: PipelineConfig, resumption_handle: str | None = None) -> str:
        setup_dict: dict = {
            "setup": {
                "model": config.model_id,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": config.voice_name}
                        }
                    },
                },
                "systemInstruction": {
                    "parts": [{"text": config.system_instruction}]
                },
            }
        }
        if resumption_handle:
            setup_dict["setup"]["sessionResumption"] = {"handle": resumption_handle}
        return json.dumps(setup_dict)

    @staticmethod
    def audio_turn_payload(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        b64 = base64.b64encode(pcm_bytes).decode("ascii")
        return json.dumps({
            "clientContent": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": f"audio/pcm;rate={sample_rate}",
                                    "data": b64,
                                }
                            }
                        ],
                    }
                ],
                "turnComplete": True,
            }
        })

    @staticmethod
    def text_turn_payload(text: str) -> str:
        return json.dumps({
            "clientContent": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": text}],
                    }
                ],
                "turnComplete": True,
            }
        })

    @staticmethod
    def parse(raw_text: str) -> tuple[list[bytes], list[str], bool, bool, str | None]:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return [], [], False, False, None

        audio_chunks: list[bytes] = []
        transcripts: list[str] = []
        interrupted = False
        turn_complete = False
        resumption_handle: str | None = None

        if isinstance(data, dict):
            ru = data.get("sessionResumptionUpdate")
            if isinstance(ru, dict) and ru.get("newHandle"):
                resumption_handle = ru["newHandle"]

        sc = data.get("serverContent")
        if isinstance(sc, dict):
            interrupted = bool(sc.get("interrupted", False))
            turn_complete = bool(sc.get("turnComplete", False))

            mt = sc.get("modelTurn")
            if isinstance(mt, dict):
                for part in mt.get("parts", []):
                    if not isinstance(part, dict):
                        continue
                    inline = part.get("inlineData")
                    if isinstance(inline, dict) and inline.get("data"):
                        try:
                            audio_chunks.append(base64.b64decode(inline["data"]))
                        except Exception:
                            pass
                    text = part.get("text")
                    if text:
                        transcripts.append(text)

            ot = sc.get("outputTranscription")
            if isinstance(ot, dict) and ot.get("text"):
                transcripts.append(ot["text"])

        return audio_chunks, transcripts, interrupted, turn_complete, resumption_handle


class LiveClient:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._resumption_handle: str | None = None
        self._running = False

    async def connect(self) -> None:
        self._running = True
        endpoint = f"{self.config.websocket_base_url}?key={self.config.api_key}"
        logger.info("Connecting to Gemini Live (%s)...", self.config.model_id)
        try:
            ws = await websockets.connect(
                endpoint,
                ping_interval=20,
                ping_timeout=20,
                max_size=16 * 1024 * 1024,
            )
            payload = Protocol.setup_payload(self.config, self._resumption_handle)
            await ws.send(payload)
            await ws.recv()
            self.ws = ws
            logger.info("Connected to Gemini Live (Voice: %s)", self.config.voice_name)
        except ConnectionClosed as e:
            if "API key not valid" in str(e.reason) or "API_KEY_INVALID" in str(e.reason):
                raise PermissionError("Invalid GEMINI_API_KEY. Verify your key at https://aistudio.google.com/") from e
            raise

    async def switch_voice(self, voice_name: str, context_prompt: str | None = None) -> None:
        self._resumption_handle = None
        if context_prompt:
            self.config = replace(self.config, voice_name=voice_name, system_instruction=context_prompt)
        else:
            self.config = replace(self.config, voice_name=voice_name)
        old_ws = self.ws
        self.ws = None
        if old_ws is not None:
            try:
                await old_ws.close()
            except Exception:
                pass
        await self.connect()

    async def update_voice(self, voice_name: str, context_prompt: str | None = None) -> None:
        await self.switch_voice(voice_name, context_prompt)

    async def send_turn(self, pcm_bytes: bytes) -> None:
        while self._running and self.ws is None:
            await asyncio.sleep(0.05)
        if self.ws is not None and pcm_bytes:
            try:
                await self.ws.send(Protocol.audio_turn_payload(pcm_bytes, self.config.input_sample_rate))
            except ConnectionClosed:
                pass

    async def send_text_turn(self, text: str) -> None:
        while self._running and self.ws is None:
            await asyncio.sleep(0.05)
        if self.ws is not None and text.strip():
            try:
                await self.ws.send(Protocol.text_turn_payload(text.strip()))
            except ConnectionClosed:
                pass

    async def recv(self) -> tuple[list[bytes], list[str], bool, bool]:
        while self._running and self.ws is None:
            await asyncio.sleep(0.05)
        if self.ws is None:
            return [], [], False, True
        try:
            raw = await self.ws.recv()
            raw_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            audio, texts, interrupted, complete, handle = Protocol.parse(raw_text)
            if handle:
                self._resumption_handle = handle
            return audio, texts, interrupted, complete
        except ConnectionClosed:
            while self._running and self.ws is None:
                await asyncio.sleep(0.05)
            return [], [], False, False

    async def close(self) -> None:
        self._running = False
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            finally:
                self.ws = None

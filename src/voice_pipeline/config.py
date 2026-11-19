from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: str | Path | None = None, override: bool = False) -> bool:
        target = Path(dotenv_path) if dotenv_path else Path(".env")
        if not target.is_file():
            return False
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if override or key not in os.environ:
                        os.environ[key] = val
            return True
        except Exception:
            return False


@dataclass(frozen=True)
class PipelineConfig:
    api_key: str
    model_id: str = "models/gemini-3.8-live"
    voice_name: str = "Kore"
    system_instruction: str = (
        "You are a helpful, concise, and ultra-fast conversational voice assistant. "
        "Speak naturally and conversationally without markdown formatting or bullet points."
    )
    input_sample_rate: int = 16000
    input_channels: int = 1
    input_chunk_size: int = 480
    output_sample_rate: int = 24000
    output_channels: int = 1
    input_device: int | None = None
    output_device: int | None = None
    silence_timeout_ms: int = 900
    text_mode: bool = False
    websocket_base_url: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> PipelineConfig:
        if env_path is not None:
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Missing or invalid GEMINI_API_KEY. Configure your Google AI Studio API key in .env"
            )

        model_id = os.getenv("GEMINI_MODEL", "models/gemini-3.8-live").strip()
        voice_name = os.getenv("VOICE_NAME", "Kore").strip()
        system_instruction = os.getenv(
            "SYSTEM_INSTRUCTION",
            "You are a helpful, concise, and ultra-fast conversational voice assistant. "
            "Speak naturally and conversationally without markdown formatting.",
        ).strip()

        input_dev = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
        output_dev = os.getenv("AUDIO_OUTPUT_DEVICE", "").strip()
        silence_timeout = os.getenv("SILENCE_TIMEOUT_MS", "900").strip()

        return cls(
            api_key=api_key,
            model_id=model_id,
            voice_name=voice_name,
            system_instruction=system_instruction,
            input_device=int(input_dev) if input_dev.isdigit() else None,
            output_device=int(output_dev) if output_dev.isdigit() else None,
            silence_timeout_ms=int(silence_timeout) if silence_timeout.isdigit() else 900,
            text_mode=os.getenv("TEXT_MODE", "false").strip().lower() in ("true", "1", "yes"),
        )

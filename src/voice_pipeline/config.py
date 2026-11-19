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
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
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
    voice_barge_in: bool = True
    silence_timeout_ms: int = 800

    websocket_base_url: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 2.0

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> PipelineConfig:
        if env_path is not None:
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Missing or invalid GEMINI_API_KEY. "
                "Please configure your Google AI Studio API key in .env. "
                "Get a free key at: https://aistudio.google.com/"
            )

        model_id = os.getenv("GEMINI_MODEL", "models/gemini-3.8-live").strip()
        voice_name = os.getenv("VOICE_NAME", "Kore").strip()
        system_instruction = os.getenv(
            "SYSTEM_INSTRUCTION",
            "You are a helpful, concise, and ultra-fast conversational voice assistant. "
            "Speak naturally and conversationally without markdown formatting.",
        ).strip()

        input_device_env = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
        output_device_env = os.getenv("AUDIO_OUTPUT_DEVICE", "").strip()

        input_device = int(input_device_env) if input_device_env.isdigit() else None
        output_device = int(output_device_env) if output_device_env.isdigit() else None
        voice_barge_in = os.getenv("VOICE_BARGE_IN", "true").strip().lower() in ("true", "1", "yes")
        silence_timeout_env = os.getenv("SILENCE_TIMEOUT_MS", "800").strip()
        silence_timeout_ms = int(silence_timeout_env) if silence_timeout_env.isdigit() else 800

        return cls(
            api_key=api_key,
            model_id=model_id,
            voice_name=voice_name,
            system_instruction=system_instruction,
            input_device=input_device,
            output_device=output_device,
            voice_barge_in=voice_barge_in,
            silence_timeout_ms=silence_timeout_ms,
        )

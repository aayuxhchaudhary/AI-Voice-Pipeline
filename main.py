from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path

_root = Path(__file__).resolve().parent
_venv_candidates = [
    _root / ".venv" / "bin" / "python",
    _root / ".venv" / "Scripts" / "python.exe",
    _root / "venv" / "bin" / "python",
    _root / "venv" / "Scripts" / "python.exe",
]
for _cand in _venv_candidates:
    if _cand.is_file() and Path(sys.executable).resolve() != _cand.resolve():
        try:
            os.execv(str(_cand), [str(_cand)] + sys.argv)
        except OSError:
            pass

sys.path.insert(0, str(_root / "src"))

import sounddevice as sd

from voice_pipeline.config import PipelineConfig
from voice_pipeline.pipeline import VoicePipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def list_devices() -> None:
    print("\n--- Audio Devices ---")
    for idx, dev in enumerate(sd.query_devices()):
        in_ch = dev.get("max_input_channels", 0)
        out_ch = dev.get("max_output_channels", 0)
        markers = []
        if in_ch > 0:
            markers.append(f"In: {in_ch}ch")
        if out_ch > 0:
            markers.append(f"Out: {out_ch}ch")
        print(f"[{idx:2d}] {dev.get('name')} ({', '.join(markers)})")
    print("---------------------\n")


def print_banner(config: PipelineConfig) -> None:
    model_name = config.model_id.replace("models/", "")
    barge_mode = "Voice & Enter" if config.voice_barge_in else "Enter key"
    print(
        f"\n┌─ Voice Pipeline\n"
        f"│  model    : {model_name} ({config.voice_name})\n"
        f"│  stream   : {config.input_sample_rate // 1000}kHz in · {config.output_sample_rate // 1000}kHz out\n"
        f"│  barge-in : {barge_mode}\n"
        f"└─ status   : listening (Ctrl+C to exit)\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversational voice pipeline via Gemini Live WebSockets.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env file")
    parser.add_argument("--voice", type=str, default=None, help="Voice name (Kore, Puck, Charon, Aoede, Fenrir)")
    parser.add_argument("--model", type=str, default=None, help="Gemini model ID")
    parser.add_argument("--voice-barge-in", action="store_true", help="Enable acoustic voice barge-in (recommended with earphones)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    setup_logging(args.verbose)

    try:
        config = PipelineConfig.from_env(args.env_file)
    except ValueError as e:
        print(f"\n[Configuration Error] {e}\n", file=sys.stderr)
        sys.exit(1)

    if args.voice:
        config = replace(config, voice_name=args.voice)
    if args.model:
        config = replace(config, model_id=args.model)
    if args.voice_barge_in:
        config = replace(config, voice_barge_in=True)

    print_banner(config)
    pipeline = VoicePipeline(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(pipeline.run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except PermissionError as e:
        print(f"\n[Authentication Error] {e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.error("Fatal pipeline error: %s", e)
        sys.exit(1)
    finally:
        try:
            loop.run_until_complete(pipeline.stop())
        except Exception:
            pass
        loop.close()
        print("\nPipeline shut down successfully.\n")


if __name__ == "__main__":
    main()

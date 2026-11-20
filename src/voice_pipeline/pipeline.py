from __future__ import annotations

import asyncio
import collections
import logging
import sys
import time
from dataclasses import replace

try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

import numpy as np

try:
    import webrtcvad
    _HAS_WEBRTC_VAD = True
except ImportError:
    _HAS_WEBRTC_VAD = False

from voice_pipeline.audio import AudioIn, AudioOut
from voice_pipeline.client import LiveClient
from voice_pipeline.config import PipelineConfig

logger = logging.getLogger("voice_pipeline.pipeline")

USE_COLOR = sys.stdout.isatty()
GREEN = "\033[32m" if USE_COLOR else ""
CYAN = "\033[36m" if USE_COLOR else ""
RED = "\033[31m" if USE_COLOR else ""
DIM = "\033[2m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""
CLEAR_LINE = "\r\033[K" if USE_COLOR else ""

SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
VOICES = ["Kore", "Fenrir", "Aoede", "Puck", "Charon"]


def render_waveform(rms: float) -> str:
    levels = ["⡀", "⣀", "⣄", "⣤", "⣶", "⣿"]
    peak = min(len(levels) - 1, max(0, int(rms / 20.0)))
    c = levels[peak]
    s1 = levels[max(0, peak - 1)]
    s2 = levels[max(0, peak - 2)]
    s3 = levels[max(0, peak - 3)]
    return f"{s3}{s2}{s1}{c}{s1}{s2}{s3}"


class VoicePipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.audio_in = AudioIn(
            sample_rate=config.input_sample_rate,
            channels=config.input_channels,
            chunk_size=config.input_chunk_size,
            device=config.input_device,
        )
        self.audio_out = AudioOut(
            sample_rate=config.output_sample_rate,
            channels=config.output_channels,
            device=config.output_device,
        )
        self.client = LiveClient(config)
        self.vad = webrtcvad.Vad(3) if _HAS_WEBRTC_VAD else None
        self._running = False
        self._interrupted = False
        self._force_commit = False
        self._is_speaking = False
        self._awaiting_response = False
        self._muted = False
        self._voice_idx = VOICES.index(config.voice_name) if config.voice_name in VOICES else 0
        self._last_turn_audio: list[bytes] = []
        self._turn_start_time: float | None = None
        self._old_termios = None
        self._history: list[tuple[str, str]] = []
        self._discard_mic_until: float = 0.0
        self._has_printed_header = False
        self._turn_done_event = asyncio.Event()
        self._turn_done_event.set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._running = True

        if self.config.text_mode:
            self.audio_out.start()
            try:
                await self.client.connect()
                await asyncio.gather(
                    self._text_loop(),
                    self._receiver_loop(),
                )
            finally:
                await self.stop()
        else:
            self.audio_in.start(loop)
            self.audio_out.start()

            if USE_COLOR and _HAS_TERMIOS:
                try:
                    self._old_termios = termios.tcgetattr(sys.stdin.fileno())
                    tty.setcbreak(sys.stdin.fileno())
                except Exception:
                    self._old_termios = None

            try:
                loop.add_reader(sys.stdin.fileno(), self._on_stdin_activity)
            except Exception:
                pass

            try:
                await self.client.connect()
                await asyncio.gather(
                    self._mic_loop(),
                    self._receiver_loop(),
                )
            finally:
                try:
                    loop.remove_reader(sys.stdin.fileno())
                except Exception:
                    pass
                if self._old_termios is not None:
                    try:
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
                    except Exception:
                        pass
                await self.stop()

    def _on_stdin_activity(self) -> None:
        try:
            ch = sys.stdin.read(1)
            if not ch:
                return
        except Exception:
            return

        if ch == " ":
            self._discard_mic_until = time.monotonic() + 0.4
            self._muted = not self._muted
            if self._muted:
                if self.audio_out.is_playing:
                    self.audio_out.flush()
                if USE_COLOR:
                    sys.stdout.write(f"{CLEAR_LINE}{RED}[MUTED]{RESET} {DIM}(Space to unmute){RESET}")
                    sys.stdout.flush()
            return

        if ch in ("v", "V"):
            self._discard_mic_until = time.monotonic() + 1.2
            self._voice_idx = (self._voice_idx + 1) % len(VOICES)
            next_voice = VOICES[self._voice_idx]
            asyncio.create_task(self._switch_voice(next_voice))
            return

        if ch in ("c", "C"):
            self._discard_mic_until = time.monotonic() + 0.4
            if USE_COLOR:
                sys.stdout.write("\033[2J\033[H")
                mode = "Text mode" if self.config.text_mode else "Ready"
                print(f"{GREEN}●{RESET} Gemini Live ({self.config.voice_name}) · {DIM}{mode} (Ctrl+C to exit){RESET}\n")
                sys.stdout.flush()
            return

        if ch in ("r", "R"):
            self._discard_mic_until = time.monotonic() + 0.4
            if self._last_turn_audio:
                if USE_COLOR:
                    sys.stdout.write(f"{CLEAR_LINE}{DIM}[replaying audio]{RESET}\n")
                    sys.stdout.flush()
                for c in self._last_turn_audio:
                    self.audio_out.play(c)
            return

        if ch in ("\r", "\n"):
            self._discard_mic_until = time.monotonic() + 0.4
            if self.audio_out.is_playing and not self._interrupted:
                self._interrupted = True
                if USE_COLOR:
                    sys.stdout.write(f" {DIM}[interrupted]{RESET}\n")
                    sys.stdout.flush()
                else:
                    print(" [interrupted]")
                self.audio_out.flush()
                self._awaiting_response = False
            elif self._is_speaking:
                self._force_commit = True

    async def _switch_voice(self, next_voice: str) -> None:
        try:
            self._discard_mic_until = time.monotonic() + 1.2
            if self.audio_out.is_playing:
                self.audio_out.flush()
            if self._has_printed_header:
                if USE_COLOR:
                    sys.stdout.write(f" {DIM}[switched]{RESET}\n")
                else:
                    print(" [switched]")
                self._has_printed_header = False
            else:
                if USE_COLOR and not self.config.text_mode:
                    sys.stdout.write(CLEAR_LINE)
            if USE_COLOR:
                sys.stdout.write(f"● Voice: {next_voice}\n")
                sys.stdout.flush()
            else:
                print(f"● Voice: {next_voice}")
            context_prompt = self._build_context_prompt()
            await self.client.switch_voice(next_voice, context_prompt)
            self.config = replace(self.config, voice_name=next_voice)
            self._turn_done_event.set()
        except Exception as e:
            logger.debug("Voice switch error: %s", e)
            self._turn_done_event.set()

    def _build_context_prompt(self) -> str | None:
        if not self._history:
            return None
        recent = self._history[-6:]
        lines = [f"{speaker}: {text}" for speaker, text in recent]
        summary = "\n".join(lines)
        return f"{self.config.system_instruction}\n\nRecent conversation context:\n{summary}"

    async def _text_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                await asyncio.wait_for(self._turn_done_event.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                self._turn_done_event.set()
            if not self._running:
                break
            try:
                user_input = await loop.run_in_executor(None, input, f"{GREEN}❯ You:{RESET} ")
            except (EOFError, asyncio.CancelledError):
                break
            except Exception:
                break

            user_text = user_input.strip()
            if not user_text:
                if self.audio_out.is_playing and not self._interrupted:
                    self._interrupted = True
                    if USE_COLOR:
                        sys.stdout.write(f" {DIM}[interrupted]{RESET}\n")
                        sys.stdout.flush()
                    else:
                        print(" [interrupted]")
                    self.audio_out.flush()
                continue

            if user_text.lower() in ("/v", "/voice"):
                self._voice_idx = (self._voice_idx + 1) % len(VOICES)
                next_voice = VOICES[self._voice_idx]
                await self._switch_voice(next_voice)
                continue
            elif user_text.lower().startswith(("/v ", "/voice ")):
                target = user_text.split(None, 1)[1].strip().capitalize()
                if target in VOICES:
                    self._voice_idx = VOICES.index(target)
                    await self._switch_voice(target)
                else:
                    print(f"Available voices: {', '.join(VOICES)}")
                continue

            if self.audio_out.is_playing:
                self._interrupted = True
                self.audio_out.flush()

            self._turn_done_event.clear()
            self._awaiting_response = True
            self._turn_start_time = time.monotonic()
            self._history.append(("User", user_text))
            await self.client.send_text_turn(user_text)

    async def _receiver_loop(self) -> None:
        self._has_printed_header = False
        current_audio: list[bytes] = []
        accumulated_text: list[str] = []
        ttfb_ms: int | None = None
        turn_audio_bytes = 0

        while self._running:
            try:
                audio_chunks, transcripts, interrupted, turn_complete = await self.client.recv()

                if audio_chunks or transcripts or interrupted or turn_complete:
                    self._awaiting_response = False
                    if ttfb_ms is None and self._turn_start_time is not None:
                        ttfb_ms = int((time.monotonic() - self._turn_start_time) * 1000)

                if interrupted:
                    self._interrupted = False
                    self.audio_out.flush()
                    self._has_printed_header = False
                    current_audio.clear()
                    accumulated_text.clear()
                    ttfb_ms = None
                    turn_audio_bytes = 0
                    self._turn_done_event.set()
                    continue

                if self._interrupted:
                    if turn_complete:
                        self._interrupted = False
                        self._has_printed_header = False
                        current_audio.clear()
                        accumulated_text.clear()
                        ttfb_ms = None
                        turn_audio_bytes = 0
                        self._turn_done_event.set()
                    continue

                for chunk in audio_chunks:
                    self.audio_out.play(chunk)
                    current_audio.append(chunk)
                    turn_audio_bytes += len(chunk)

                for text in transcripts:
                    if not self._has_printed_header:
                        if USE_COLOR and not self.config.text_mode:
                            sys.stdout.write(CLEAR_LINE)
                        print(f"{CYAN}◆ Gemini:{RESET} ", end="", flush=True)
                        self._has_printed_header = True
                    print(text, end="", flush=True)
                    accumulated_text.append(text)

                if turn_complete:
                    if current_audio:
                        self._last_turn_audio = list(current_audio)
                        current_audio.clear()

                    if self._has_printed_header:
                        if ttfb_ms is not None and turn_audio_bytes > 0:
                            sec = turn_audio_bytes / (self.config.output_sample_rate * 2)
                            print(f" {DIM}[{ttfb_ms}ms · {sec:.1f}s audio]{RESET}\n")
                        else:
                            print("\n")
                        if accumulated_text:
                            self._history.append(("Gemini", "".join(accumulated_text)))
                    elif turn_audio_bytes > 0:
                        if USE_COLOR and not self.config.text_mode:
                            sys.stdout.write(CLEAR_LINE)
                        sec = turn_audio_bytes / (self.config.output_sample_rate * 2)
                        lat = f" [{ttfb_ms}ms · {sec:.1f}s audio]" if ttfb_ms else f" [{sec:.1f}s audio]"
                        print(f"{CYAN}◆ Gemini:{RESET} {DIM}[voice]{lat}{RESET}\n")
                        self._history.append(("Gemini", "[voice response]"))

                    accumulated_text.clear()
                    self._has_printed_header = False
                    ttfb_ms = None
                    turn_audio_bytes = 0
                    self._turn_start_time = None
                    self._turn_done_event.set()

            except asyncio.CancelledError:
                self._turn_done_event.set()
                break
            except Exception as e:
                self._turn_done_event.set()
                if not self._running:
                    break
                logger.error("Receiver error: %s", e)
                await asyncio.sleep(0.1)

    async def _mic_loop(self) -> None:
        preroll = collections.deque(maxlen=6)
        buffer = bytearray()
        is_speaking = False
        silence_chunks = 0
        speech_chunks = 0
        total_speech_chunks = 0
        barge_in_chunks = 0
        min_speech_chunks = 4
        spin_idx = 0

        chunk_ms = (self.config.input_chunk_size / self.config.input_sample_rate) * 1000.0
        silence_limit_chunks = max(10, int(round(self.config.silence_timeout_ms / chunk_ms)))

        noise_floor = 30.0
        calibration_samples = 0

        while self._running:
            chunk = await self.audio_in.read()
            if not chunk:
                continue

            if time.monotonic() < self._discard_mic_until:
                buffer.clear()
                preroll.clear()
                is_speaking = False
                self._is_speaking = False
                speech_chunks = 0
                silence_chunks = 0
                total_speech_chunks = 0
                continue

            spin_idx += 1
            currently_playing = self.audio_out.is_playing

            if self._muted:
                chunk = bytes(len(chunk))
                rms = 0.0
                if USE_COLOR and (spin_idx % 6 == 0):
                    sys.stdout.write(f"{CLEAR_LINE}{RED}[MUTED]{RESET} {DIM}(Space to unmute){RESET}")
                    sys.stdout.flush()
                continue

            s = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            s -= s.mean()
            rms = float(np.sqrt(np.mean(s * s)))

            if calibration_samples < 15:
                noise_floor = (noise_floor * calibration_samples + rms) / (calibration_samples + 1)
                calibration_samples += 1
                preroll.append(chunk)
                continue

            if not is_speaking and not currently_playing and not self._awaiting_response:
                if rms < noise_floor * 1.3:
                    noise_floor = 0.95 * noise_floor + 0.05 * rms
                elif rms < noise_floor * 1.8:
                    noise_floor = 0.998 * noise_floor + 0.002 * rms

                if USE_COLOR and (spin_idx % 3 == 0):
                    spin = SPINNERS[(spin_idx // 3) % len(SPINNERS)]
                    wave = render_waveform(rms)
                    sys.stdout.write(f"{CLEAR_LINE}{DIM}{spin} Listening... {wave}{RESET}")
                    sys.stdout.flush()

            is_speech_frame = False
            if self.vad is not None and len(chunk) == 960:
                try:
                    is_speech_frame = self.vad.is_speech(chunk, self.config.input_sample_rate)
                except Exception:
                    is_speech_frame = False

            speech_energy_thresh = max(40.0, noise_floor * 1.25 + 15.0)
            has_speech = is_speech_frame and (rms > speech_energy_thresh)

            if self._awaiting_response:
                if has_speech and rms > (noise_floor * 2.0 + 30.0):
                    speech_chunks += 1
                    if speech_chunks >= 5:
                        self._awaiting_response = False
                        is_speaking = True
                        self._is_speaking = True
                        total_speech_chunks = speech_chunks
                        buffer.clear()
                        for p in preroll:
                            buffer.extend(p)
                        buffer.extend(chunk)
                        preroll.clear()
                else:
                    speech_chunks = 0
                preroll.append(chunk)
                continue

            if currently_playing:
                if self.audio_in._is_echo_cancel and self.audio_out._is_echo_cancel:
                    barge_thresh = max(50.0, noise_floor * 1.35 + 20.0)
                else:
                    barge_thresh = max(350.0, noise_floor * 2.5)

                if has_speech and rms > barge_thresh:
                    barge_in_chunks += 1
                    if barge_in_chunks >= 3 and not self._interrupted:
                        self._interrupted = True
                        if USE_COLOR:
                            sys.stdout.write(f" {DIM}[interrupted]{RESET}\n")
                            sys.stdout.flush()
                        else:
                            print(" [interrupted]")
                        self.audio_out.flush()

                        buffer.clear()
                        for p in preroll:
                            buffer.extend(p)
                        buffer.extend(chunk)
                        preroll.clear()

                        is_speaking = True
                        self._is_speaking = True
                        speech_chunks = min_speech_chunks
                        total_speech_chunks = barge_in_chunks
                        silence_chunks = 0
                        barge_in_chunks = 0
                else:
                    barge_in_chunks = 0
                continue

            barge_in_chunks = 0

            if not is_speaking:
                preroll.append(chunk)

            if has_speech:
                speech_chunks += 1
                if speech_chunks >= min_speech_chunks and not is_speaking:
                    is_speaking = True
                    self._is_speaking = True
                    buffer.clear()
                    for p in preroll:
                        buffer.extend(p)
                    preroll.clear()

                if is_speaking:
                    total_speech_chunks += 1
                    buffer.extend(chunk)
                    silence_chunks = 0
                    if USE_COLOR and (spin_idx % 2 == 0):
                        wave = render_waveform(rms)
                        sys.stdout.write(f"{CLEAR_LINE}{GREEN}❯ You:{RESET} {wave}")
                        sys.stdout.flush()
            else:
                if not is_speaking:
                    speech_chunks = 0
                else:
                    buffer.extend(chunk)
                    silence_chunks += 1

                    max_duration_reached = len(buffer) >= (self.config.input_sample_rate * 2 * 20)
                    force_commit = self._force_commit and is_speaking and len(buffer) >= 3200
                    self._force_commit = False

                    if silence_chunks >= silence_limit_chunks or max_duration_reached or force_commit:
                        is_speaking = False
                        self._is_speaking = False
                        saved_silence = silence_chunks
                        saved_speech = total_speech_chunks
                        silence_chunks = 0
                        speech_chunks = 0
                        total_speech_chunks = 0

                        if saved_speech < 6 and not force_commit:
                            buffer.clear()
                            continue

                        duration_sec = len(buffer) / (self.config.input_sample_rate * 2)
                        if duration_sec < 0.35 and not force_commit:
                            buffer.clear()
                            continue

                        trim_chunks = max(0, saved_silence - 4)
                        trim_bytes = trim_chunks * len(chunk)
                        if len(buffer) > trim_bytes + 3200:
                            raw_pcm = bytes(buffer[:-trim_bytes])
                        else:
                            raw_pcm = bytes(buffer)
                        buffer.clear()

                        if len(raw_pcm) == 0:
                            continue

                        samples_out = np.frombuffer(raw_pcm, dtype=np.int16)
                        if len(samples_out) == 0:
                            continue

                        if USE_COLOR:
                            sys.stdout.write(f"{CLEAR_LINE}{GREEN}❯ You:{RESET} {DIM}[voice]{RESET}\n")
                            sys.stdout.flush()
                        else:
                            print("❯ You: [voice]")

                        peak = float(np.max(np.abs(samples_out)))
                        if 0 < peak < 20000:
                            gain = min(24000.0 / peak, 4.0)
                            boosted = np.clip(samples_out.astype(np.float32) * gain, -32767, 32767).astype(np.int16)
                            utterance = boosted.tobytes()
                        else:
                            utterance = raw_pcm

                        self._awaiting_response = True
                        self._turn_start_time = time.monotonic()
                        self._history.append(("User", "[voice]"))
                        await self.client.send_turn(utterance)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._turn_done_event.set()
        self.audio_in.stop()
        self.audio_out.stop()
        await self.client.close()

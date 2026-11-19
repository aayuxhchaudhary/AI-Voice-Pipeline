from __future__ import annotations

import asyncio
import collections
import logging
import sys

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
        self._force_listening = False
        self._force_commit = False
        self._is_speaking = False
        self._awaiting_response = False

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
                await self.stop()

    def _on_stdin_activity(self) -> None:
        try:
            line = sys.stdin.readline()
            if not line:
                return
        except Exception:
            return

        if self.audio_out.is_playing and not self._interrupted:
            self._interrupted = True
            print("\n[Barge-in] Interrupted assistant.")
            print("\n[You]: Listening...", end="", flush=True)
            self.audio_out.flush()
            self._force_listening = True
            self._awaiting_response = False
        elif self._is_speaking:
            self._force_commit = True

    async def _text_loop(self) -> None:
        loop = asyncio.get_running_loop()
        print("\nType your message below (or press Enter while assistant speaks to barge in):")
        while self._running:
            try:
                user_input = await loop.run_in_executor(None, input, "\n[You]: ")
            except (EOFError, asyncio.CancelledError):
                break
            except Exception:
                break

            user_text = user_input.strip()
            if not user_text:
                if self.audio_out.is_playing and not self._interrupted:
                    self._interrupted = True
                    print("\n[Barge-in] Interrupted assistant.")
                    self.audio_out.flush()
                continue

            if self.audio_out.is_playing:
                self._interrupted = True
                self.audio_out.flush()

            self._awaiting_response = True
            await self.client.send_text_turn(user_text)

    async def _receiver_loop(self) -> None:
        has_printed_header = False
        while self._running:
            try:
                audio_chunks, transcripts, interrupted, turn_complete = await self.client.recv()

                if audio_chunks or transcripts or interrupted or turn_complete:
                    self._awaiting_response = False

                if interrupted:
                    self._interrupted = False
                    self.audio_out.flush()
                    has_printed_header = False
                    continue

                if self._interrupted:
                    if turn_complete:
                        self._interrupted = False
                        has_printed_header = False
                    continue

                for chunk in audio_chunks:
                    self.audio_out.play(chunk)

                for text in transcripts:
                    if not has_printed_header:
                        print("\n[Gemini]: ", end="", flush=True)
                        has_printed_header = True
                    print(text, end="", flush=True)

                if turn_complete:
                    if has_printed_header:
                        print()
                    has_printed_header = False

            except asyncio.CancelledError:
                break
            except Exception as e:
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
        min_speech_chunks = 2
        was_playing = False

        chunk_ms = (self.config.input_chunk_size / self.config.input_sample_rate) * 1000.0
        silence_limit_chunks = max(10, int(round(self.config.silence_timeout_ms / chunk_ms)))

        noise_floor = 30.0
        calibration_samples = 0

        while self._running:
            chunk = await self.audio_in.read()
            if not chunk:
                continue

            # Detect transition: assistant finished speaking -> prompt user that mic is ready
            currently_playing = self.audio_out.is_playing
            if was_playing and not currently_playing:
                if not is_speaking and not self._awaiting_response:
                    print("\n[You]: Listening...", end="", flush=True)
            was_playing = currently_playing

            samples = np.frombuffer(chunk, dtype=np.int16)
            samples_clean = samples - int(np.mean(samples))
            rms = float(np.sqrt(np.mean(samples_clean.astype(np.float32) ** 2)))

            # Initial calibration of ambient noise floor (first 450ms)
            if calibration_samples < 15:
                noise_floor = (noise_floor * calibration_samples + rms) / (calibration_samples + 1)
                calibration_samples += 1
                preroll.append(chunk)
                continue

            # Adaptive noise floor tracking during quiet periods
            if not is_speaking and not currently_playing and not self._awaiting_response:
                if rms < noise_floor * 1.3:
                    noise_floor = 0.95 * noise_floor + 0.05 * rms
                elif rms < noise_floor * 1.8:
                    noise_floor = 0.998 * noise_floor + 0.002 * rms

            # WebRTC VAD Mode 3 on clean un-boosted samples (filters fan/ambient noise)
            is_speech_frame = False
            if self.vad is not None and len(chunk) == 960:
                try:
                    is_speech_frame = self.vad.is_speech(samples_clean.tobytes(), self.config.input_sample_rate)
                except Exception:
                    is_speech_frame = False

            # Speech requires vocal formants AND energy above ambient noise
            speech_energy_thresh = max(40.0, noise_floor * 1.25 + 15.0)
            has_speech = is_speech_frame and (rms > speech_energy_thresh)

            # While awaiting assistant response, suppress noise; allow fast override only on confident speech
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
                        print("\n[You]: Listening...", end="", flush=True)
                else:
                    speech_chunks = 0
                preroll.append(chunk)
                continue

            # Manual Hotkey Interrupt (<Enter>)
            if self._force_listening:
                self._force_listening = False
                is_speaking = True
                self._is_speaking = True
                speech_chunks = min_speech_chunks
                total_speech_chunks = min_speech_chunks
                silence_chunks = 0
                buffer.clear()
                preroll.clear()

            # --- Playback Active (Assistant Speaking) ---
            if currently_playing:
                # If PipeWire Echo-Cancel Source + Sink are paired, speaker output is cancelled
                if self.audio_in._is_echo_cancel and self.audio_out._is_echo_cancel:
                    barge_thresh = max(50.0, noise_floor * 1.35 + 20.0)
                else:
                    barge_thresh = max(350.0, noise_floor * 2.5)

                if has_speech and rms > barge_thresh:
                    barge_in_chunks += 1
                    if barge_in_chunks >= 3 and not self._interrupted:
                        self._interrupted = True
                        print("\n[Barge-in] Interrupted assistant.")
                        print("\n[You]: Listening...", end="", flush=True)
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

            # Pre-roll buffer during silence
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
                    print("\n[You]: Listening...", end="", flush=True)

                if is_speaking:
                    total_speech_chunks += 1
                    buffer.extend(chunk)
                    silence_chunks = 0
            else:
                if not is_speaking:
                    speech_chunks = 0
                else:
                    buffer.extend(chunk)
                    silence_chunks += 1

                    # Commit condition: silence timeout reached, max duration reached, or user pressed Enter
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

                        # Discard transient noise bursts / fan fluctuations (must contain at least 4 speech frames)
                        if saved_speech < 4 and not force_commit:
                            buffer.clear()
                            continue

                        duration_sec = len(buffer) / (self.config.input_sample_rate * 2)
                        if duration_sec < 0.25 and not force_commit:
                            buffer.clear()
                            continue

                        # Trim trailing silence buffer so Gemini receives utterance without dead silence lag
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

                        peak = float(np.max(np.abs(samples_out)))
                        if 0 < peak < 20000:
                            gain = min(24000.0 / peak, 4.0)
                            boosted = np.clip(samples_out.astype(np.float32) * gain, -32767, 32767).astype(np.int16)
                            utterance = boosted.tobytes()
                        else:
                            utterance = raw_pcm

                        self._awaiting_response = True
                        await self.client.send_turn(utterance)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.audio_in.stop()
        self.audio_out.stop()
        await self.client.close()

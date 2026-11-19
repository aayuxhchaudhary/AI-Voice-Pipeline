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
        self._awaiting_response = False

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self.audio_in.start(loop)
        self.audio_out.start()
        self._running = True

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
            sys.stdin.readline()
        except Exception:
            return
        if self.audio_out.is_playing and not self._interrupted:
            self._interrupted = True
            print("\n[Barge-in] Interrupted assistant.")
            print("\n[You]: Listening...", end="", flush=True)
            self.audio_out.flush()
            self._force_listening = True
            self._awaiting_response = False

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
        barge_in_chunks = 0
        min_speech_chunks = 2
        silence_limit_chunks = 10  # 300ms pause to commit turn

        noise_floor = 30.0
        calibration_samples = 0
        awaiting_response_chunks = 0

        while self._running:
            chunk = await self.audio_in.read()
            if not chunk:
                continue

            if self._awaiting_response:
                awaiting_response_chunks += 1
                if awaiting_response_chunks > 100:
                    self._awaiting_response = False
                    awaiting_response_chunks = 0
                continue
            awaiting_response_chunks = 0

            samples = np.frombuffer(chunk, dtype=np.int16)
            samples_clean = samples - int(np.mean(samples))
            rms = float(np.sqrt(np.mean(samples_clean.astype(np.float32) ** 2)))

            if calibration_samples < 20:
                noise_floor = (noise_floor * calibration_samples + rms) / (calibration_samples + 1)
                calibration_samples += 1
                preroll.append(chunk)
                continue

            # Continuous noise floor adaptation during non-speech:
            if not is_speaking and rms < noise_floor * 1.4:
                noise_floor = 0.96 * noise_floor + 0.04 * rms

            # WebRTC VAD Mode 3 on clean un-boosted samples (filters fan/ambient noise)
            is_speech_frame = False
            if self.vad is not None and len(chunk) == 960:
                try:
                    is_speech_frame = self.vad.is_speech(samples_clean.tobytes(), self.config.input_sample_rate)
                except Exception:
                    is_speech_frame = False

            # Speech requires vocal formants AND energy above ambient noise
            speech_energy_thresh = max(45.0, noise_floor * 1.3)
            has_speech = is_speech_frame and (rms > speech_energy_thresh)

            # Silence condition: energy drops back to ambient noise OR VAD detects no speech
            is_silence = not has_speech or (rms < noise_floor * 1.35 + 10.0)

            # Manual Hotkey Interrupt (<Enter>)
            if self._force_listening:
                self._force_listening = False
                is_speaking = True
                speech_chunks = min_speech_chunks
                silence_chunks = 0
                buffer.clear()
                preroll.clear()

            # --- Playback Active (Assistant Speaking) ---
            if self.audio_out.is_playing:
                # If voice barge-in is enabled (e.g. headphones):
                if self.config.voice_barge_in:
                    barge_thresh = max(600.0, noise_floor * 3.5)
                    if has_speech and rms > barge_thresh:
                        barge_in_chunks += 1
                        if barge_in_chunks >= 2 and not self._interrupted:
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
                            speech_chunks = min_speech_chunks
                            silence_chunks = 0
                            barge_in_chunks = 0
                    else:
                        barge_in_chunks = 0
                else:
                    # Echo-Guard active: mic ignores speaker output to prevent self-interruption.
                    # User presses <Enter> to interrupt instantly!
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
                    buffer.clear()
                    for p in preroll:
                        buffer.extend(p)
                    preroll.clear()
                    print("\n[You]: Listening...", end="", flush=True)

                if is_speaking:
                    buffer.extend(chunk)
                    silence_chunks = 0
            else:
                if is_speaking:
                    buffer.extend(chunk)
                    if is_silence:
                        silence_chunks += 1

                    # Commit condition: 300ms silence OR max duration (10s) reached
                    max_duration_reached = len(buffer) >= (self.config.input_sample_rate * 2 * 10)
                    if silence_chunks >= silence_limit_chunks or max_duration_reached:
                        is_speaking = False
                        speech_chunks = 0
                        silence_chunks = 0

                        duration_sec = len(buffer) / (self.config.input_sample_rate * 2)
                        if duration_sec < 0.30:
                            buffer.clear()
                            continue

                        raw_pcm = bytes(buffer)
                        buffer.clear()

                        samples_out = np.frombuffer(raw_pcm, dtype=np.int16)
                        peak = float(np.max(np.abs(samples_out)))
                        if 0 < peak < 20000:
                            gain = min(24000.0 / peak, 5.0)
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

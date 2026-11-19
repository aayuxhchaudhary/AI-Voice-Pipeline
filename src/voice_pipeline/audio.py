from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger("voice_pipeline.audio")


def find_echo_cancel_devices() -> tuple[int | None, int | None]:
    """Find PipeWire/PulseAudio Echo-Cancel Source (input) and Sink (output) device indices."""
    in_idx: int | None = None
    out_idx: int | None = None
    try:
        for idx, dev in enumerate(sd.query_devices()):
            name = dev.get("name", "").lower()
            if "echo-cancel" in name or "echo_cancel" in name or "echo cancel" in name:
                if in_idx is None and "source" in name and dev.get("max_input_channels", 0) > 0:
                    in_idx = idx
                elif out_idx is None and "sink" in name and dev.get("max_output_channels", 0) > 0:
                    out_idx = idx
    except Exception as e:
        logger.debug("Failed querying devices for echo-cancel: %s", e)
    return in_idx, out_idx


class AudioIn:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 480,
        device: int | None = None,
        max_queue: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue)
        self._stream: sd.RawInputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._is_echo_cancel = False
        self._hw_channels = channels
        self._downsample_ratio = 1

    def _callback(self, indata: bytes, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Audio input status: %s", status)
        if not self._running or self._loop is None:
            return

        if self._is_echo_cancel and self._downsample_ratio > 1:
            samples = np.frombuffer(indata, dtype=np.int16)
            ch0 = samples[0::self._hw_channels]
            ch0_16k = ch0[::self._downsample_ratio]
            chunk = ch0_16k.tobytes()
        else:
            chunk = bytes(indata)

        try:
            self._loop.call_soon_threadsafe(self._enqueue, chunk)
        except RuntimeError:
            pass

    def _enqueue(self, chunk: bytes) -> None:
        try:
            self.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(chunk)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            return
        self._loop = loop
        self._running = True

        hw_device = self.device
        hw_rate = self.sample_rate
        hw_channels = self.channels
        hw_blocksize = self.chunk_size

        if hw_device is None:
            in_aec, _ = find_echo_cancel_devices()
            if in_aec is not None:
                try:
                    dev = sd.query_devices(in_aec)
                    hw_device = in_aec
                    self._is_echo_cancel = True
                    hw_rate = int(dev.get("default_samplerate", 48000))
                    hw_channels = int(dev.get("max_input_channels", 2))
                    self._hw_channels = hw_channels
                    self._downsample_ratio = hw_rate // self.sample_rate
                    hw_blocksize = self.chunk_size * self._downsample_ratio
                    logger.info("Auto-selected PipeWire Echo-Cancel Source (Device [%d] %s)", in_aec, dev.get("name"))
                except Exception as e:
                    logger.debug("Could not initialize echo-cancel source: %s", e)

        self._stream = sd.RawInputStream(
            samplerate=hw_rate,
            channels=hw_channels,
            dtype="int16",
            blocksize=hw_blocksize,
            device=hw_device,
            callback=self._callback,
        )
        self._stream.start()
        dev_label = "echo-cancel" if self._is_echo_cancel else (hw_device or "default")
        logger.info("Microphone initialized (%dHz, ch=%d, dev=%s)", self.sample_rate, self.channels, dev_label)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error stopping input stream: %s", e)
            finally:
                self._stream = None

    async def read(self) -> bytes:
        return await self.queue.get()

    def drain(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class AudioOut:
    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        chunk_size: int = 2048,
        device: int | None = None,
        prebuffer_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self._prebuffer_ms = prebuffer_ms
        self._prebuffer_bytes = int(sample_rate * channels * 2 * (prebuffer_ms / 1000.0))
        self._stream: sd.RawOutputStream | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._running = False
        self._is_buffering = True
        self._last_playback_time: float = 0.0
        self._is_echo_cancel = False
        self._hw_rate = sample_rate
        self._hw_channels = channels
        self._upsample_factor = 1

    def _callback(self, outdata: bytearray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Audio output status: %s", status)

        bytes_needed = len(outdata)

        with self._lock:
            if self._is_buffering:
                if len(self._buffer) >= self._prebuffer_bytes:
                    self._is_buffering = False
                else:
                    outdata[:] = b"\x00" * bytes_needed
                    return

            available = len(self._buffer)
            if available >= bytes_needed:
                outdata[:] = self._buffer[:bytes_needed]
                del self._buffer[:bytes_needed]
                self._last_playback_time = time.monotonic()
            elif available > 0:
                outdata[:available] = self._buffer[:available]
                outdata[available:bytes_needed] = b"\x00" * (bytes_needed - available)
                del self._buffer[:available]
                self._last_playback_time = time.monotonic()
                self._is_buffering = True
            else:
                outdata[:] = b"\x00" * bytes_needed
                self._is_buffering = True

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._is_buffering = True
        self._last_playback_time = 0.0

        hw_device = self.device
        hw_rate = self.sample_rate
        hw_channels = self.channels
        hw_blocksize = self.chunk_size

        if hw_device is None:
            _, out_aec = find_echo_cancel_devices()
            if out_aec is not None:
                try:
                    dev = sd.query_devices(out_aec)
                    hw_device = out_aec
                    self._is_echo_cancel = True
                    hw_rate = int(dev.get("default_samplerate", 48000))
                    hw_channels = int(dev.get("max_output_channels", 2))
                    self._hw_rate = hw_rate
                    self._hw_channels = hw_channels
                    rate_mult = hw_rate // self.sample_rate
                    ch_mult = hw_channels // self.channels
                    self._upsample_factor = rate_mult * ch_mult
                    hw_blocksize = self.chunk_size * rate_mult
                    logger.info("Auto-selected PipeWire Echo-Cancel Sink (Device [%d] %s)", out_aec, dev.get("name"))
                except Exception as e:
                    logger.debug("Could not initialize echo-cancel sink: %s", e)

        self._prebuffer_bytes = int(hw_rate * hw_channels * 2 * (self._prebuffer_ms / 1000.0))

        self._stream = sd.RawOutputStream(
            samplerate=hw_rate,
            channels=hw_channels,
            dtype="int16",
            blocksize=hw_blocksize,
            device=hw_device,
            callback=self._callback,
        )
        self._stream.start()
        dev_label = "echo-cancel" if self._is_echo_cancel else (hw_device or "default")
        logger.info("Speaker initialized (%dHz, ch=%d, dev=%s)", hw_rate, hw_channels, dev_label)

    def stop(self) -> None:
        self._running = False
        self.flush()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error stopping output stream: %s", e)
            finally:
                self._stream = None

    def play(self, chunk: bytes) -> None:
        if not self._running or not chunk:
            return
        if self._is_echo_cancel and self._upsample_factor > 1:
            samples = np.frombuffer(chunk, dtype=np.int16)
            hw_chunk = samples.repeat(self._upsample_factor).tobytes()
        else:
            hw_chunk = chunk

        with self._lock:
            self._buffer.extend(hw_chunk)
            if self._is_buffering and len(self._buffer) >= self._prebuffer_bytes:
                self._is_buffering = False

    def flush(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._is_buffering = True
            self._last_playback_time = 0.0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            if len(self._buffer) > 0:
                return True
            return (time.monotonic() - self._last_playback_time) < 0.20

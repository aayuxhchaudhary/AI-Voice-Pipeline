from __future__ import annotations

import asyncio
import logging
import threading
import time

import sounddevice as sd

logger = logging.getLogger("voice_pipeline.audio")


class AudioIn:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
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

    def _callback(self, indata: bytes, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Audio input status: %s", status)
        if not self._running or self._loop is None:
            return

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
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("Microphone initialized (%dHz, ch=%d, dev=%s)", self.sample_rate, self.channels, self.device or "default")

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
        prebuffer_ms: int = 150,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self._prebuffer_bytes = int(sample_rate * channels * 2 * (prebuffer_ms / 1000.0))
        self._stream: sd.RawOutputStream | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._running = False
        self._is_buffering = True
        self._last_playback_time: float = 0.0

    def _callback(self, outdata: bytearray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Audio output status: %s", status)

        bytes_needed = frames * self.channels * 2

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
            else:
                outdata[:] = b"\x00" * bytes_needed

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._is_buffering = True
        self._last_playback_time = 0.0
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("Speaker initialized (%dHz, ch=%d, dev=%s)", self.sample_rate, self.channels, self.device or "default")

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
        with self._lock:
            self._buffer.extend(chunk)
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
            return (time.monotonic() - self._last_playback_time) < 0.3

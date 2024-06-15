import threading
from typing import List
import pyaudio

from configs import logger
from whisper.audio_params import AudioParams

class AudioRecorder():
    def __init__(
        self, 
        device_id: int = -1
    ):
        def stream_callback(in_data, frame_count, time_info, status):
            self.on_data_received(in_data)
            return (in_data, pyaudio.paContinue)

        self._pa = pyaudio.PyAudio()

        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            print((i, dev['name'], dev['maxInputChannels']))

        self._stream = self._pa.open(
            rate=AudioParams.RATE,
            channels=AudioParams.CHANNELS,
            format=self._pa.get_format_from_width(AudioParams.WIDTH),
            input=True,
            input_device_index=(device_id if device_id >= 0 else None),
            frames_per_buffer=AudioParams.CHUNK_SIZE,
            stream_callback=stream_callback
        )

        self._chunks_lock = threading.Lock()
        self._running = False
        self._remote_allow = True
        self._local_allow = True

        self.reset()

    def reset(self):
        self._chunks = []

    def on_data_received(self, in_data: bytes):
        if not self._running:
            return
        
        self._chunks.append(in_data)
        
    def fetch_chunks(self) -> List[bytes]:
        with self._chunks_lock:
            result = self._chunks
            self._chunks = []

            return result

    def start(self):
        self._running = True

    def pause(self, from_remote=False):
        if from_remote:
            self._remote_allow = False
        else:
            self._local_allow = False

        if not (self._remote_allow and self._local_allow):
            self._running = False
            logger.info(f'audio record paused by {"remote" if from_remote else "local"}.')

    def resume(self, from_remote=False):
        if from_remote:
            self._remote_allow = True
        else:
            self._local_allow = True
        
        if self._remote_allow and self._local_allow:
            self.reset()
            self._running = True
            logger.info('audio record resumed.')

    def is_active(self) -> bool:
        return self._stream.is_active()

    def __del__(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

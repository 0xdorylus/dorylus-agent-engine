import os
from array import array
import struct
from io import BytesIO
import time
import datetime as dt
from typing import List
import wave
import webrtcvad
import collections
import torch
import numpy as np

from configs import logger, SPEECH_DIR
from whisper.audio_params import AudioParams
import whisper.utils as utils

def pack_audio_chunks(chunks: List[bytes]) -> bytes:
    raw_data = array('h')
    for chunk in chunks:
        raw_data.extend(array('h', chunk))
    
    raw_data = normalize(raw_data)
    with BytesIO() as buf:
        for val in raw_data:
            buf.write(struct.pack('<h', val))

        return buf.getvalue()

# normalize the captured samples
def normalize(snd_data: array):
    MAXIMUM = 32767
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r

def dump_wave_file(raw_data: bytes, sub_dir=''):
    file_path = dt.datetime.now().strftime(f'{SPEECH_DIR}/{sub_dir}%Y%m%d_%H%M%S.wav')
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(AudioParams.CHANNELS)
        wf.setsampwidth(AudioParams.WIDTH)
        wf.setframerate(AudioParams.RATE)
        wf.writeframes(raw_data)

def is_speech_complete(pcmf32: List[float]) -> bool:
    return utils.vad_simple(
        pcmf32=pcmf32, 
        sample_rate=AudioParams.RATE, 
        last_ms=AudioParams.SIMPLE_VAD_AUDIO_REAR_MS, 
        vad_thold=AudioParams.SIMPLE_VAD_THOLD, 
        freq_thold=AudioParams.SIMPLE_VAD_FREQ_THOLD, 
    )

class AudioProcessor():
    def __init__(self) -> None:
        self._processing = True

        self._vad_pcmf32_buffer = collections.deque(maxlen=AudioParams.SIMPLE_VAD_AUDIO_CHUNK_SIZE)
        self._stream_pcmf32_buffer = collections.deque(
            maxlen=AudioParams.STREAM_SAMPLES_KEEP + AudioParams.STREAM_SAMPLES_LEN
        )

        self.reset()

        VadUtil.init()

        logger.info("Start audio processing: ")

    def reset(self):
        self._triggered = False

        self._ring_buffer_flags = [0] * AudioParams.NUM_WINDOW_CHUNKS
        self._ring_buffer_index = 0

        self._ring_buffer_flags_end = [0] * AudioParams.NUM_WINDOW_CHUNKS_END
        self._ring_buffer_index_end = 0

        self._chunks = []
        self._start_index = 0
        self._start_time = 0.0

        self._vad_pcmf32_buffer.clear()
        self._stream_pcmf32_buffer.clear()
        self._stream_samples_cur_step = 0
        self._stream_steps_cur_line = 0

    def recv_data(self, in_data: bytes):
        if not self._processing:
            return None
        
        self._chunks.append(in_data)
        int16_arr = array('h', in_data)
        for x in int16_arr:
            val = float(x) / 32768.0
            self._vad_pcmf32_buffer.append(val)
            self._stream_pcmf32_buffer.append(val)

        active = VadUtil.vad.is_speech(in_data, AudioParams.RATE)
        # sys.stdout.write('1' if active else '_')

        self._ring_buffer_flags[self._ring_buffer_index] = 1 if active else 0
        self._ring_buffer_index += 1
        if self._ring_buffer_index >= AudioParams.NUM_WINDOW_CHUNKS:
            self._ring_buffer_index = 0

        self._ring_buffer_flags_end[self._ring_buffer_index_end] = 1 if active else 0
        self._ring_buffer_index_end += 1
        if self._ring_buffer_index_end >= AudioParams.NUM_WINDOW_CHUNKS_END:
            self._ring_buffer_index_end = 0

        # voice starting detection
        got_a_sentence = False
        if not self._triggered:
            self._start_index += 1
            num_voiced = sum(self._ring_buffer_flags)
            if num_voiced > 0.8 * AudioParams.NUM_WINDOW_CHUNKS:
                # sys.stdout.write(' Open ')
                self._triggered = True
                self._start_index -= AudioParams.START_OFFSET
                if self._start_index < 0:
                    self._start_index = 0
                self._chunks = self._chunks[self._start_index : ]
                self._start_time = time.time()

                stream_pcmf32_buffer_len = len(self._stream_pcmf32_buffer)
                simples_keeps = min(stream_pcmf32_buffer_len, AudioParams.STREAM_SAMPLES_STEP)
                for _ in range(stream_pcmf32_buffer_len - simples_keeps):
                    self._stream_pcmf32_buffer.popleft()

        # voice ending detection
        else:
            simple_vad_complete = is_speech_complete(list(self._vad_pcmf32_buffer))

            elasped_secs = time.time() - self._start_time
            num_unvoiced = AudioParams.NUM_WINDOW_CHUNKS_END - sum(self._ring_buffer_flags_end)
            webrtcvad_complete = (num_unvoiced > 0.90 * AudioParams.NUM_WINDOW_CHUNKS_END
                or elasped_secs > AudioParams.MAX_RECORDING_SECONDS)
            if (simple_vad_complete or webrtcvad_complete):
                # got a sentence
                # sys.stdout.write(f' Close[{0 if simple_vad_complete else 1}] \n')
                got_a_sentence = True

        stream_pcmf32 = None
        stream_end_of_line = False
        if self._triggered:
            cur_samples = len(int16_arr)
            self._stream_samples_cur_step += cur_samples

            # transcribe when step samples reached
            if self._stream_samples_cur_step >= AudioParams.STREAM_SAMPLES_STEP:
                stream_pcmf32 = list(self._stream_pcmf32_buffer)
                self._stream_samples_cur_step = 0
                self._stream_steps_cur_line += 1
            elif got_a_sentence:
                stream_pcmf32 = list(self._stream_pcmf32_buffer)
                stream_end_of_line = True

            # left some samples for next line
            if self._stream_steps_cur_line >= AudioParams.STREAM_STEPS_PER_LINE:
                self._stream_steps_cur_line = 0
                stream_end_of_line = True

                stream_pcmf32_buffer_len = len(self._stream_pcmf32_buffer)
                simples_keeps = min(stream_pcmf32_buffer_len, AudioParams.STREAM_SAMPLES_KEEP)
                for _ in range(stream_pcmf32_buffer_len - simples_keeps):
                    self._stream_pcmf32_buffer.popleft()

        # sys.stdout.flush()
        sentence = None
        if got_a_sentence:
            sentence = VadUtil.silero_vad.detect(self._chunks)
            self.reset()
        
        return (sentence, stream_pcmf32, stream_end_of_line)
    
    def stop(self):
        if not self._processing:
            return

        self._processing = False

        logger.info('audio processing stopped.')
    
    def resume(self):
        if self._processing:
            return
        
        self.reset()
        self._processing = True

        logger.info('audio processing resumed.')

def int_to_float(sound):
    _sound = np.copy(sound)
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1 / abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32

class SileroVAD():
    def __init__(self) -> None:
        silero_vad_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "silero-vad")
        self._model, utils = torch.hub.load(
            repo_or_dir=silero_vad_dir,
            model='silero_vad',
            source='local',
            force_reload=True,
            onnx=False
        )

        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks
        ) = utils

        self._get_speech_timestamps = get_speech_timestamps
        logger.info('SileroVAD init done.')

    def detect(self, chunks):
        raw_data = pack_audio_chunks(chunks)

        start_tm = time.time()
        newsound = np.frombuffer(raw_data, np.int16)
        audio_float32 = int_to_float(newsound)
        time_stamps = self._get_speech_timestamps(
            audio_float32, 
            self._model,
            min_speech_duration_ms=300,  # min speech duration in ms
            min_silence_duration_ms=600,  # min silence duration
            speech_pad_ms=200,  # spech pad ms
        )

        elasped = time.time() - start_tm
        if len(time_stamps) > 0:
            # logger.info(f"silero VAD has detected a possible speech, elasped: {elasped:.02f}s")
            # dump_wave_file(raw_data, sub_dir='record/')
            return raw_data
        else:
            # logger.info(f"silero VAD has detected a noise, elasped: {elasped:.02f}s")
            return None
            
class VadUtil():
    vad = None
    silero_vad = None

    @classmethod
    def init(cls):
        if cls.vad is None:
            cls.vad = webrtcvad.Vad(3)
        if cls.silero_vad is None:
            cls.silero_vad = SileroVAD()

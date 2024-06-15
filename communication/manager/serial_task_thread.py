import os
import threading
import queue
import time
import inspect
from typing import List

from configs import logger, CONST_SPEECH_MAP, SPEECH_DELAY_SECONDS
from communication.manager.local_manager import g_local_manager, NodeStatus
from model_api.tts import play_const_audio
from communication.models import *

class SerialTaskThread(threading.Thread):
    def __init__(self, task_queue: queue.Queue, response_callback):
        super().__init__()
        self._running: bool = True
        self._task_queue = task_queue
        self.response_callback = response_callback
        self._booted_time = time.time()

    def run(self):
        while self._running:
            if g_local_manager.status == NodeStatus.BOOTED:
                speech = CONST_SPEECH_MAP['BOOTED']
                if not os.path.exists(speech['file']):
                    time.sleep(0.1)
                    continue
                else:
                    play_const_audio('BOOTED')
                    self._booted_time = time.time()
                    g_local_manager.set_status(NodeStatus.NETWORK_CONNECTING)
                    continue
            elif g_local_manager.status == NodeStatus.NETWORK_CONNECTED:
                now = time.time()
                speech = CONST_SPEECH_MAP['CONNECTED']
                if ((not os.path.exists(speech['file'])) or 
                    now < self._booted_time + SPEECH_DELAY_SECONDS):
                    
                    time.sleep(0.1)
                    continue
                else:
                    play_const_audio('CONNECTED', block=False)
                    g_local_manager.set_status(NodeStatus.WORKING)
                    continue

            if g_local_manager.status != NodeStatus.WORKING or self._task_queue.empty():
                time.sleep(0.1)
                continue

            func, kwargs, ctx = self._task_queue.get()
            if inspect.isgeneratorfunction(func):
                for result in func(**kwargs):
                    self.response_callback(result, ctx)
            else:
                result = func(**kwargs)
                if result:
                    self.response_callback(result, ctx)
        
        logger.info('SerialTaskThread exit.')

    def stop(self):
        self._running = False

class WhisperStreamThread(threading.Thread):
    def __init__(self, task_queue: queue.Queue, response_callback):
        super().__init__()
        self._running: bool = True
        self._task_queue = task_queue
        self.response_callback = response_callback
    
    def run(self):
        while self._running:
            if self._task_queue.empty():
                time.sleep(0.1)
                continue
            
            stream_pcmf32: List[float]
            stream_end_of_line: bool
            context: ChatAgentConnContext

            stream_pcmf32, stream_end_of_line, context = self._task_queue.get()
            stream_txt = g_local_manager.process_speech_stream(stream_pcmf32)
            self.response_callback(WsBase[AsrStreamResult](
                msgType='asr_stream',
                data=AsrStreamResult(
                    txt=stream_txt,
                    end_of_line=stream_end_of_line,
                ),
            ), context)

    def stop(self):
        self._running = False

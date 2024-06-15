import uvicorn
import signal
import queue
import asyncio

from communication.manager.serial_task_thread import SerialTaskThread, WhisperStreamThread
from communication.models import *

# FastAPI Uvicorn override
class RestfulServer(uvicorn.Server):

    # Override
    def install_signal_handlers(self) -> None:
        # Do nothing
        pass

class ServerManager():
    def __init__(self):
        self._restful_servers = []
        self._ws_servers = []
        self._is_running = True
        self._serial_task_queue = queue.Queue()
        self._ws_response_queue = queue.Queue()
        self._event_queue = queue.Queue()
        self._whisper_stream_queue = queue.Queue()
        self._speech_queue = asyncio.Queue()
        signal.signal(signal.SIGINT, lambda _, __: self.terminate_all())

    def start_serial_worker_thread(self):
        self._serial_worker = SerialTaskThread(
            task_queue=self._serial_task_queue, 
            response_callback=lambda rsp, ctx: self.put_ws_response(rsp, ctx),
        )
        self._serial_worker.start()

        self._whisper_stream_worker = WhisperStreamThread(
            task_queue=self._whisper_stream_queue, 
            response_callback=lambda rsp, ctx: self.put_ws_response(rsp, ctx),
        )
        self._whisper_stream_worker.start()
    
    def add_serial_task(self, func, kwargs, ctx):
        self._serial_task_queue.put((func, kwargs, ctx))

    def put_whisper_stream(self, stream_pcmf32, stream_end_of_line, context):
        self._whisper_stream_queue.put((stream_pcmf32, stream_end_of_line, context))
    
    def put_ws_response(self, response, context: WsConnContext):
        self._ws_response_queue.put((response, context))

    def get_ws_response(self):
        if self._ws_response_queue.empty():
            return None
        return self._ws_response_queue.get()
    
    def put_event(self, kind, content):
        self._event_queue.put((kind, content))
    
    def get_event(self):
        if self._event_queue.empty():
            return None
        return self._event_queue.get()
    
    async def put_speech(
        self, 
        speech_data: bytes, 
        context
    ):
        await self._speech_queue.put((speech_data, context))

    async def get_speech(self):
        if self._speech_queue.empty():
            return None
        return await self._speech_queue.get()

    def reg_ws_server(self, server):
        self._ws_servers.append(server)

    def create_restful_server(self, config: uvicorn.Config):
        server = RestfulServer(config)
        self._restful_servers.append(server)
        return server

    def terminate_all(self):
        for svr in self._ws_servers:
            svr.close()
        for svr in self._restful_servers:
            svr.should_exit = True

        self._serial_worker.stop()
        self._serial_worker.join()

        self._whisper_stream_worker.stop()
        self._whisper_stream_worker.join()

        self._is_running = False
    
    def is_running(self):
        return self._is_running

g_server_manager = ServerManager()

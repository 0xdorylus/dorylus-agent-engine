import sys
import time
import json
import websockets
import asyncio
import subprocess
import shlex

from configs import logger, WS_MAX_SIZE
from whisper.audio_recorder import AudioRecorder
from communication.models import *

class AudioPlayer():
    def __init__(self, audio_recorder: AudioRecorder) -> None:
        self._player = None
        self._audio_recorder = audio_recorder

    def recv_audio_bytes(self, audio_bytes):
        if self._player is None:
            ffplay_process = "ffplay -autoexit -nodisp -hide_banner -loglevel error -i pipe:0"
            self._player = subprocess.Popen(shlex.split(ffplay_process), stdin=subprocess.PIPE)
            logger.info('Audio player opened.')

            # Stop record when player start
            self._audio_recorder.pause()
        
        self._player.stdin.write(audio_bytes)

    def close(self):
        if self._player is None:
            return
        
        self._player.stdin.close()
        self._player.wait()
        self._player = None
        logger.info('Audio player closed.')

        # Resume record when player closed
        self._audio_recorder.resume()

_local_ws = None
async def send_message_to_local(message):
    global _local_ws
    if not _local_ws:
        return
    
    await _local_ws.send(message)

async def record_pause(_: WsBase[object], audio_recorder: AudioRecorder, __: AudioPlayer):    
    audio_recorder.pause(from_remote=True)

async def record_resume(_: WsBase[object], audio_recorder: AudioRecorder, __: AudioPlayer):    
    audio_recorder.resume(from_remote=True)

async def audio_end(_: WsBase[object], __: AudioRecorder, audio_player: AudioPlayer):    
    audio_player.close()

action_list = [
    (record_pause, WsBase[object]),
    (record_resume, WsBase[object]),
    (audio_end, WsBase[object]),
]
actions_map = {}
for x in action_list:
    actions_map[x[0].__name__] = x

async def start_local_ws_connection(audio_recorder: AudioRecorder, audio_player: AudioPlayer):
    async for websocket in websockets.connect(
        "ws://127.0.0.1:5001",
        max_size=WS_MAX_SIZE,
    ):
        logger.info('websocket connected.')
        global _local_ws
        _local_ws = websocket
        try:
            async for message in websocket:
                if type(message) == bytes:
                    # receive audio bytes
                    audio_player.recv_audio_bytes(message)
                    continue

                logger.info(f"Received local ws message: {message}")
                event = json.loads(message)
                reqBase = WsBase[object].model_validate(event)
                if reqBase.msgType in actions_map:
                    func, req = actions_map[reqBase.msgType]
                    await func(req, audio_recorder, audio_player)
                
        except websockets.ConnectionClosed as e:
            logger.info(f'websocket error: {e}, reconnect...')
            audio_recorder.resume()
            continue

async def capture_audio_data(audio_recorder: AudioRecorder):
    time.sleep(2)
    while audio_recorder.is_active():
        chunks = audio_recorder.fetch_chunks()
        if len(chunks) == 0:
            await asyncio.sleep(0.1)
            continue
        
        for chunk in chunks:
            await send_message_to_local(chunk)

async def main(audio_recorder: AudioRecorder, audio_player: AudioPlayer):
    tasks = [
        start_local_ws_connection(audio_recorder, audio_player),
        capture_audio_data(audio_recorder),
    ]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    audio_recorder = AudioRecorder()

    # wait for 1 second to avoid any buffered noise
    time.sleep(1)
    audio_recorder.start()

    audio_player = AudioPlayer(audio_recorder)

    try:
        asyncio.run(main(audio_recorder, audio_player))
    except KeyboardInterrupt:
        sys.stdout.write('\n')
        sys.stdout.flush()
        logger.info('>>> Stop audio recording')

        audio_player.close()

    audio_recorder.pause()
    del audio_recorder

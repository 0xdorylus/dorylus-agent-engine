import asyncio

from configs import logger
from communication.models import *
import communication.api as api
from communication.server_manager import g_server_manager
from communication.manager.local_manager import g_local_manager

async def agent_speech_processing_loop():
    while g_server_manager.is_running():
        data = await g_server_manager.get_speech()
        if not data:
            await asyncio.sleep(0.1)
            continue

        speech_data: bytes
        context: ChatAgentConnContext
        speech_data, context = data
        try:
            await api.process_speech(speech_data, context)
        except Exception as e:
            logger.error(f'process_speech: {e}', exc_info=e)
        finally:
            context.audio_processor.resume()
            # require audio record resuming
            try:
                await context.websocket.send(
                    WsBase[object](msgType='record_resume').model_dump_json(exclude_none=True)
                )
            except:
                pass

async def ws_responsing_loop():
    while g_server_manager.is_running():
        working = False
        result = g_server_manager.get_ws_response()
        if result:
            context: WsConnContext
            response, context = result
            if type(response) == bytes:
                await context.websocket.send(response)
            else:
                msg = response.model_dump_json(exclude_none=True)
                logger.info(f'send response: {msg}')
                await context.websocket.send(msg)

        ev = g_server_manager.get_event()
        if ev:
            kind, content = ev
            await g_local_manager.send_event(kind, content)
            working = True

        if not working:
            await asyncio.sleep(0.1)

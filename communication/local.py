import websockets
import json
import asyncio
from fastapi import FastAPI
import uvicorn
from typing import Annotated
from fastapi import File, Header

from configs import logger, NLTK_DATA_PATH, WS_MAX_SIZE
from communication.models import *
import communication.api as api
from communication.manager.local_manager import g_local_manager
from communication.server_manager import g_server_manager
from whisper.audio_processor import AudioProcessor
from model_api.tts import async_fetch_const_audio_stream

import nltk
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

async def search_docs_from_kb(req: WsBase[SearchDocsFromKBRequest], _: WsConnContext):
    return await api.search_docs_from_kb(req)

async def list_kbs(req: WsBase[object], _: WsConnContext):
    return await asyncio.to_thread(api.list_kbs, req)

async def create_kb(req: WsBase[KBCreateRequest], ctx: WsConnContext):
    g_server_manager.add_serial_task(api.create_kb, {'req': req}, ctx)

async def delete_kb(req: WsBase[KBDeleteRequest], ctx: WsConnContext):
    g_server_manager.add_serial_task(api.delete_kb, {'req': req}, ctx)

async def update_info(req: WsBase[KBUpdateInfoRequest], ctx: WsConnContext):
    g_server_manager.add_serial_task(api.update_info, {'req': req}, ctx)

async def list_files(req: WsBase[KBListFilesRequest], _: WsConnContext):
    return await asyncio.to_thread(api.list_files, req)

async def update_docs(req: WsBase[KBUpdateDocsRequest], ctx: WsConnContext):
    g_server_manager.add_serial_task(api.update_docs, {'req': req}, ctx)

async def delete_docs(req: WsBase[KBDeleteDocsRequest], ctx: WsConnContext):
    g_server_manager.add_serial_task(api.delete_docs, {'req': req}, ctx)

async def recreate_vector_store(req: WsBase[KBRecreateVectorStoreRequest], ctx: WsConnContext):
    g_server_manager.add_serial_task(api.recreate_vector_store, {'req': req}, ctx)     

async def search_docs(req: WsBase[KBSearchDocsRequest], _: WsConnContext):
    return await asyncio.to_thread(api.search_docs, req)

action_list = [
    (search_docs_from_kb, WsBase[SearchDocsFromKBRequest]),
    (list_kbs, WsBase[object]),
    (create_kb, WsBase[KBCreateRequest]),
    (delete_kb, WsBase[KBDeleteRequest]),
    (update_info, WsBase[KBUpdateInfoRequest]),
    (list_files, WsBase[KBListFilesRequest]),
    (update_docs, WsBase[KBUpdateDocsRequest]),
    (delete_docs, WsBase[KBDeleteDocsRequest]),
    (recreate_vector_store, WsBase[KBRecreateVectorStoreRequest]),
    (search_docs, WsBase[KBSearchDocsRequest]),
]
actions_map = {}
for x in action_list:
    actions_map[x[0].__name__] = x

async def on_connected(websocket: websockets.WebSocketServerProtocol):
    try:
        # Register new ws connection
        g_local_manager.add_ws(websocket)

        # Init audio_processor for current local connection
        context = ChatAgentConnContext()
        context.websocket = websocket
        context.audio_processor = AudioProcessor()

        access_token = await g_local_manager.get_node_api_access_token()
        context.conversation = CachedConversation(access_token=access_token)

        async for audio_chunks in async_fetch_const_audio_stream('WS_CONNECTED'):
            await websocket.send(audio_chunks)
        await websocket.send(
            WsBase[object](msgType='audio_end').model_dump_json(exclude_none=True)
        )

        # Manage state changes
        async for message in websocket:
            try:
                if type(message) == bytes:
                    try:
                        await api.chat_agent_recv_audio_chunks(message, context)
                    except Exception as e:
                        logger.error(f'audio_process: {e}', exc_info=e)
                        # require audio record resuming
                        await websocket.send(
                            WsBase[object](msgType='record_resume').model_dump_json(exclude_none=True)
                        )
                    continue

                event = json.loads(message)
                reqBase = WsBase[object].model_validate(event)
                if reqBase.msgType not in actions_map:
                    logger.error(f'Invalid action: {reqBase.msgType}')
                    continue

                func, req = actions_map[reqBase.msgType]
                result = await func(req.model_validate(event), context)
                if result:
                    await websocket.send(
                        result.model_dump_json(exclude_none=True)
                    )
                    
            except Exception as error:
                logger.error(f"Message handle error: {error}",  exc_info=error)
    finally:
        # Remove connection
        g_local_manager.rmv_ws(websocket)
        
        # Broacast to other connections

async def local_ws_server():
    g_local_manager.load_models()

    server = await websockets.serve(
        on_connected, 
        "127.0.0.1", 5001, 
        max_size=WS_MAX_SIZE, 
        start_serving=True,
    )
    g_server_manager.reg_ws_server(server)
    # await server.start_serving()
    await server.wait_closed()
    # await asyncio.Future()  # run forever
    logger.info('local_ws_server exit')

app = FastAPI()

@app.post("/api/download_doc")
async def api_download_doc(body: KBDownloadDocRequest): 
    return api.download_doc(body)

@app.post("/api/dorylus_agent_clear")
async def api_dorylus_agent_clear(
    x_agent_user_id: Annotated[str, Header()], 
):
    return await api.dorylus_agent_clear(x_agent_user_id)

@app.post("/api/dorylus_agent_text")
async def api_dorylus_agent_text(
    body: DorylusAgentTextRequest,
    x_agent_user_id: Annotated[str, Header()], 
):
    return await api.dorylus_agent_text(body, x_agent_user_id)

@app.post("/api/dorylus_agent_audio")
async def api_dorylus_agent_audio(
    file: Annotated[bytes, File()],
    x_agent_user_id: Annotated[str, Header()], 
):
    return await api.dorylus_agent_audio(file, x_agent_user_id)

async def local_restful_server():
    config = uvicorn.Config("communication.local:app", host='127.0.0.1', port=5002, log_level="info")
    server = g_server_manager.create_restful_server(config)
    await server.serve()
    logger.info('local_restful_server exit')

from fastapi import FastAPI, File, Header
import uvicorn
from typing import Annotated

from configs import logger, WS_MAX_SIZE
from communication.api import *
from communication.models import *
from communication.server_manager import g_server_manager
from communication.manager.local_manager import g_local_manager
import communication.api as api

app = FastAPI()

@app.post("/api/search_docs/{imei}")
async def api_search_docs(imei: str, body: ApiSearchDocsRequest) -> ApiSearchDocsResponse:
    logger.info(f'search_docs(from {imei}) starting ...')
    rsp = search_docs(WsBase[KBSearchDocsRequest](
        msgType='search_docs',
        data=KBSearchDocsRequest(
            query=body.query,
            kb_name=body.kb_name,
        ),
    ))

    chunks = []
    if type(rsp.data) == list:
        chunks = rsp.data
        log_content = f'search_docs(from {imei}) success: {body}'
    else:
        log_content = f'search_docs(from {imei}) failed: {rsp.data.msg}, {body}'
    logger.info(log_content)
    await g_local_manager.send_log(log_content)
    g_server_manager.put_event('vector_db_response', {
        'from_node': imei,
        'to_node': g_local_manager.get_node_info().imei,
        'text': body.query,
    })

    return ApiSearchDocsResponse(
        chunks=chunks,
    )

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

async def neighbor_restful_server():
    config = uvicorn.Config("communication.neighbor:app", host='127.0.0.1', port=5101, log_level="info")
    server = g_server_manager.create_restful_server(config)
    await server.serve()
    logger.info('neighbor_restful_server exit')

async def on_connected(websocket: websockets.WebSocketServerProtocol):
    try:
        # Init audio_processor for current local connection
        context = ChatAgentConnContext()
        context.websocket = websocket
        context.audio_processor = AudioProcessor()

        access_token = await g_local_manager.get_node_api_access_token()
        context.conversation = CachedConversation(access_token=access_token)

        # Manage state changes
        async for message in websocket:
            try:
                if type(message) == bytes:
                    try:
                        if websocket.path == '/chat/agent':
                            await api.chat_agent_recv_audio_chunks(message, context)
                        elif websocket.path == '/asr/stream':
                            await api.asr_stream_recv_audio_chunks(message, context)

                    except Exception as e:
                        logger.error(f'audio_process: {e}', exc_info=e)
                        # require audio record resuming
                        await websocket.send(
                            WsBase[object](msgType='record_resume').model_dump_json(exclude_none=True)
                        )
                    continue
                    
            except Exception as error:
                logger.error(f"Message handle error: {error}",  exc_info=error)
    finally:
        pass
        
        # Broacast to other connections

async def neighbor_ws_server():
    server = await websockets.serve(
        on_connected, 
        "127.0.0.1", 5102, 
        max_size=WS_MAX_SIZE, 
        start_serving=True,
    )
    g_server_manager.reg_ws_server(server)
    await server.wait_closed()
    logger.info('neighbor_ws_server exit')

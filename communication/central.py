import os
import json
import time
import random
import websockets
import asyncio
from urllib.parse import urlencode

from configs import (
    logger, CENTRAL_SERVER_WS_URL, EMBEDDING_MODEL_PATH, WS_MAX_SIZE
)
from communication.manager.local_manager import g_local_manager, NodeStatus
from communication.server_manager import g_server_manager
import communication.api as api
from communication.models import *
from model_api.tts import create_const_speech


async def handle_login(req: WsBase[LoginInfo], _: WsConnContext):
    logger.info(f'login: {req}')

    await g_local_manager.set_login_info(req.data)

    if os.path.exists(EMBEDDING_MODEL_PATH):
        kb_name = g_local_manager.get_kb_name()
        rsp = await asyncio.to_thread(api.create_kb, WsBase[KBCreateRequest](
            msgType='create_kb',
            data=KBCreateRequest(
                kb_name=kb_name,
            ),
        ))
        if rsp.data.code == 0:
            await g_local_manager.send_log(f'Knowledge base creation success: {kb_name}')
        elif rsp.data.code > 0:
            logger.error(f'create_kb error: {rsp.data.msg}')
    
    if g_local_manager.status == NodeStatus.NETWORK_CONNECTING:
        g_local_manager.set_status(NodeStatus.NETWORK_CONNECTED)
    return g_local_manager.collect_node_detail()

async def handle_query_indexed(req: WsBase[List[QueryIndexed]], _: WsConnContext):
    logger.info(f'handle_query_indexed: {req}')
    g_local_manager.set_index_types(req.data)

async def handle_ping(req: WsBase[HeartBeat], _: WsConnContext):
    req.msgType = 'pong'
    return req

async def handle_file(req: WsBase[FileTransfer], ctx: WsConnContext):
    def _inner_func(req: WsBase[FileTransfer]) -> WsBase[WsRspBase]:
        data = req.data
        file_name = data.fileName
        kb_name = g_local_manager.get_kb_name()

        try:
            rsp = api.upload_file(file_name, kb_name)
            if rsp.data.code != 0:
                logger.error(f'handle_file({kb_name}, {file_name}) error: {rsp.data.msg}')
            else:
                logger.info(f'handle_file({kb_name}, {file_name}) success.')
            
            return WsBase[WsRspBase](
                msgType=req.msgType,
                data=rsp.data,
            )
            
        except Exception as e:
            msg = f'handle_file: {e}'
            logger.error(msg, exc_info=e)
            return WsBase[WsRspBase](
                msgType=req.msgType,
                data=WsRspBase(
                    code=500,
                    msg=msg,
                ),
            )
    
    g_server_manager.add_serial_task(_inner_func, {'req': req}, ctx)

async def handle_file_del(req: WsBase[FileDel], ctx: WsConnContext):
    def _inner_func(req: WsBase[FileDel]) -> WsBase[WsRspBase]:
        data = req.data
        file_name = data.fileName
        kb_name = g_local_manager.get_kb_name()

        try:
            rsp = api.delete_docs(WsBase[KBDeleteDocsRequest](
                msgType='delete_docs',
                data=KBDeleteDocsRequest(
                    kb_name=kb_name,
                    file_names=[file_name],
                    delete_content=True,
                ),
            ))
            if rsp.data.code != 0:
                logger.error(f'handle_file_del({kb_name}, {file_name}) error: {rsp.data.msg}')
            else:
                logger.info(f'handle_file_del({kb_name}, {file_name}) success.')
                g_server_manager.put_event('file_embedding', {
                    'text': file_name,
                })
            
            return WsBase[WsRspBase](
                msgType=req.msgType,
                data=rsp.data,
            )
            
        except Exception as e:
            msg = f'handle_file_del: {e}'
            logger.error(msg, exc_info=e)
            return WsBase[WsRspBase](
                msgType=req.msgType,
                data=WsRspBase(
                    code=500,
                    msg=msg,
                ),
            )
    
    g_server_manager.add_serial_task(_inner_func, {'req': req}, ctx)

async def handle_query(req: WsBase[List[QueryResponse]], _: WsConnContext):
    logger.info(f'handle_query: {req}')
    if len(req.data) == 0:
        await g_local_manager.put_query_node(None)
    else:
        await g_local_manager.put_query_node(random.choice(req.data))

###########################################################

action_list = [
    (handle_login, WsBase[LoginInfo]),
    (handle_query_indexed, WsBase[QueryIndexed]),
    (handle_ping, WsBase[HeartBeat]),
    (handle_file, WsBase[FileTransfer]),
    (handle_file_del, WsBase[FileDel]),
    (handle_query, WsBase[QueryResponse]),
]
actions_map = {}
for x in action_list:
    actions_map[x[0].__name__[len('handle_'):]] = x

###########################################################

async def start_central_connection():
    g_server_manager.start_serial_worker_thread()
    await create_const_speech()
    g_local_manager.set_status(NodeStatus.BOOTED)

    node_info = g_local_manager.load_node_info()
    params = {
        'imei': node_info.imei,
    }
    url = CENTRAL_SERVER_WS_URL + '?' + urlencode(params)

    while True:
        if g_local_manager.status == NodeStatus.BOOTED:
            await asyncio.sleep(0.1)
            continue
        
        try:
            logger.info(f'connecting central ws: {url}')
            websocket = await websockets.connect(url, max_size=WS_MAX_SIZE)
            g_local_manager.set_central_ws(websocket)
            
            context = WsConnContext()
            context.websocket = websocket

            try:
                async for message in websocket:
                    # print(f'recv message: {message}')
                    event = json.loads(message)
                    reqBase = WsBase[object].model_validate(event)
                    if (not reqBase.msgType) or (reqBase.msgType not in actions_map):
                        logger.error(f'Invalid message type: {reqBase.msgType}')
                        continue

                    func, req = actions_map[reqBase.msgType]
                    result = await func(req.model_validate(event), context)
                    if result:
                        await websocket.send(
                            result.model_dump_json(exclude_none=True)
                        )
                    if not g_server_manager.is_running():
                        break
                
                logger.info(f'central websocket client is closing ...')
                await websocket.close()
                logger.info(f'central websocket client closed.')
                if not g_server_manager.is_running():
                    break

            except websockets.ConnectionClosed as e:
                logger.error(f'websocket error: {e}, reconnect...')
                continue
        except Exception as e:
            logger.error(f'websocket connect error: {e}, reconnect...', exc_info=e)
            time.sleep(3)
            continue


import asyncio
from typing import List, Dict

from communication.models import *
from communication.manager.local_manager import g_local_manager
from communication.server_manager import g_server_manager
from communication.api.kb_doc_api import search_docs
from configs import (
    logger,
)

async def search_docs_from_kb(req: WsBase[SearchDocsFromKBRequest]) -> WsBase[List[Dict]]:
    node_info = g_local_manager.get_node_info()
    
    if node_info.kb_type == req.data.kb_type:
        g_server_manager.put_event('query_vector_db', {
            'from_node': node_info.imei,
            'to_node': node_info.imei,
            'text': req.data.query,
        })

        # match local kb type
        kb_name = g_local_manager.get_kb_name()
        rsp = await asyncio.to_thread(search_docs, WsBase[KBSearchDocsRequest](
            msgType='search_docs',
            data=KBSearchDocsRequest(
                query=req.data.query,
                kb_name=kb_name,
            ),
        ))

        chunks = []
        if type(rsp.data) == list:
            chunks = rsp.data
            log_content = f'search_docs local success: type({req.data.kb_type}), query({req.data.query})'
        else:
            log_content = f'search_docs local failed: {rsp.data.msg}, type({req.data.kb_type}), query({req.data.query})'
        logger.info(log_content)
        await g_local_manager.send_log(log_content)

        return WsBase[List[Dict]](
            msgType=req.msgType,
            data=chunks,
        )
    else:
        # search from remote node
        await g_local_manager.send_query(req.data.kb_type)

        query_node = await g_local_manager.wait_for_query_node(req.data.kb_type)
        if query_node is None:
            logger.info(f'No query_node for {req.data.kb_type}')
            return WsBase[List[Dict]](
                msgType=req.msgType,
                data=[],
            )
        
        logger.info(f'Got remote query_node: {query_node}')
        g_server_manager.put_event('query_vector_db', {
            'from_node': node_info.imei,
            'to_node': query_node.imei,
            'text': req.data.query,
        })

        chunks = await g_local_manager.search_docs_from_neighbor(
            kb_type=req.data.kb_type,
            query=req.data.query,
            query_node=query_node,
        )

        await g_local_manager.send_log(f'search_docs from remote({query_node}) success: type({req.data.kb_type}), query({req.data.query})')

        return WsBase[List[Dict]](
            msgType=req.msgType,
            data=chunks,
        )
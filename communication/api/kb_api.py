import urllib

from configs import logger, log_verbose
from communication.models import *
from communication.manager.local_manager import g_local_manager
from db.repository.knowledge_base_repository import list_kbs_from_db
from knowledge_base.utils import validate_kb_name
from knowledge_base.kb_service.base import KBServiceFactory

def list_kbs(req: WsBase[object]) -> WsBase[List[str]]:
    # Get List of Knowledge Base
    return WsBase[List[str]](
        msgType=req.msgType,
        data=list_kbs_from_db()
    )

def create_kb(req: WsBase[KBCreateRequest]) -> WsBase[WsRspBase]:
    # Create selected knowledge base
    if not validate_kb_name(req.data.kb_name):
        return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase(code=403, msg="Don't attack me"))
    if req.data.kb_name.strip() == "":
        return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase(code=404, msg="Knowledge base name is empty"))

    kb = KBServiceFactory.get_service_by_name(req.data.kb_name)
    if kb is not None:
        return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase(code=-1))

    kb = KBServiceFactory.get_service(req.data.kb_name, req.data.vs_type, req.data.model)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"Knowledge base creation failed: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e if log_verbose else None)
        return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase(code=500, msg=msg))

    return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase())

def delete_kb(req: WsBase[KBDeleteRequest]) -> WsBase[WsRspBase]:
    # Delete selected knowledge base
    if not validate_kb_name(req.data.kb_name):
        return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase(code=403, msg="Don't attack me"))
    req.data.kb_name = urllib.parse.unquote(req.data.kb_name)

    kb = KBServiceFactory.get_service_by_name(req.data.kb_name)
    if kb is None:
        return WsBase[WsRspBase](
            msgType=req.msgType, 
            data=WsRspBase(code=404, msg=f"Knowledge base '{req.data.kb_name}' not found.")
        )

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return WsBase[WsRspBase](
                msgType=req.msgType, data=WsRspBase() 
            )
    except Exception as e:
        msg = f"Knowledge base deletion failed: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e if log_verbose else None)
        return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase(code=500, msg=msg))

    return WsBase[WsRspBase](
        msgType=req.msgType, 
        data=WsRspBase(
            code=500, 
            msg=f"Knowledge base {req.data.kb_name} deletion failed."
        )
    )

def update_info(req: WsBase[KBUpdateInfoRequest]) -> WsBase[WsRspBase]:
    if not validate_kb_name(req.data.kb_name):
        return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase(code=403, msg="Don't attack me"))

    kb = KBServiceFactory.get_service_by_name(req.data.kb_name)
    if kb is None:
        return WsBase[WsRspBase](
            msgType=req.msgType, 
            data=WsRspBase(code=404, msg=f"Knowledge base '{req.data.kb_name}' not found.")
        )
    kb.update_info(req.data.kb_info)

    return WsBase[WsRspBase](msgType=req.msgType, data=WsRspBase())

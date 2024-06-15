from typing import Dict, Union
from pydantic import BaseModel
from communication.models.common import WsBase

class LoginInfo(BaseModel):
    id: int
    type: int
    cmd: str
    ip: str

class QueryIndexed(BaseModel):
    id: int
    name: str

class HeartBeat(BaseModel):
    imei: str

class FileTransfer(BaseModel):
    fileName: str

class FileDel(BaseModel):
    fileName: str

class QueryRequest(WsBase):
    imei: str
    type: int

class QueryResponse(BaseModel):
    imei: str
    ip: str
    type: int

class LogRequest(BaseModel):
    imei: str
    content: str

class NodeDetailRequest(BaseModel):
    imei: str
    hw_info: Dict
    compute_cap_level: int
    compute_cap: Dict

class EventRequest(BaseModel):
    kind: str
    node_id: str
    content: Union[Dict, str]

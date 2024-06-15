from pydantic import BaseModel
from typing import Optional, TypeVar, Generic

T = TypeVar("T")

class WsBase(BaseModel, Generic[T]):
    msgType: Optional[str] = None
    data: Optional[T] = None

class WsRspBase(BaseModel):
    code: int = 0
    msg: Optional[str] = 'success'

class AsrStreamResult(BaseModel):
    txt: str
    end_of_line: bool

class DorylusAgentTextRequest(BaseModel):
    question: str

class DorylusAgentTextResponse(BaseModel):
    answer: str

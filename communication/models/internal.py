import websockets
from enum import Enum
import collections
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple, Any

from whisper.audio_processor import AudioProcessor
from model_api.cached_conversation import CachedConversation

class PromptStatus(Enum):
    NO_PROMPT = 0,
    HAVE_PROMPT = 1,

class WsConnContext():
    websocket: websockets.WebSocketServerProtocol

class ChatAgentConnContext(WsConnContext):
    audio_processor: AudioProcessor
    status: PromptStatus = PromptStatus.NO_PROMPT
    conversation: CachedConversation
    history = collections.deque(maxlen=3)

class DorylusAgentContext():
    last_access: int
    intention: Optional[str] = None
    state: Dict[str, Any] = {}
    conversation: CachedConversation
    history = collections.deque(maxlen=10)

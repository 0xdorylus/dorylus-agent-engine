import os
from enum import Enum
from .kb import MODEL_ROOT_PATH, KB_ROOT_PATH

NODE_COMPUTE_CAP_LEVEL = 2
NODE_COMPUTE_CAP_INFO = {
    'level_1': {
        'asr': True,
        'tts': False,
        'llm': False,
        'kb': True,
        'sd': False,
        'blip': False,
        'clip': False,
        'ai_modules': ['Whisper', 'FAISS', ],
    },
    'level_2': {
        'asr': True,
        'tts': False,
        'llm': True,
        'kb': True,
        'sd': False,
        'blip': False,
        'clip': False,
        'ai_modules': ['Whisper', 'FAISS', 'OLLAMA', ],
    },
    'level_3': {
        'asr': True,
        'tts': True,
        'llm': True,
        'kb': True,
        'sd': True,
        'blip': True,
        'clip': True,
        'ai_modules': ['Whisper', 'FAISS', 'OLLAMA', 'GPT-soVITS', 'Stable diffusion', 'BLIP', 'CLIP'],
    },
}

# Whisper model
WHISPER_MODEL = 'ggml-medium.bin'
WHISPER_MODEL_DIR = os.path.join(MODEL_ROOT_PATH, "whisper")
WHISPER_MODEL_PATH = os.path.join(WHISPER_MODEL_DIR, WHISPER_MODEL)

# Node info
NODE_INFO_PATH = os.path.join(KB_ROOT_PATH, 'node_info.pickle')
NODE_INFO_JSON_PATH = os.path.join(KB_ROOT_PATH, 'node_info.json')

# Central server URL
CENTRAL_SERVER_HOST = '18.181.169.40'
CENTRAL_SERVER_WS_URL = f'ws://{CENTRAL_SERVER_HOST}/ws/instance'
CENTRAL_SERVER_DOWNLOAD_URL = f'http://{CENTRAL_SERVER_HOST}/download'
# CENTRAL_SERVER_PROXIES = {
#     "http": "http://127.0.0.1:8118",
#     "https": "https://127.0.0.1:8118",
# }
CENTRAL_SERVER_PROXIES = None
# OPENAI_PROXY = 'http://127.0.0.1:8118'
OPENAI_PROXY = None
WS_MAX_SIZE = 10 * 1024 * 1024

# tts
SPEECH_DIR = os.path.join(KB_ROOT_PATH, "speech")

# local tts
USING_LOCAL_TTS = False
LOCAL_TTS_SERVER = 'http://127.0.0.1:5000/tts'
DEFAULT_VOICE_LOCAL_MAP = {
    'default': 'Emma',
    'en': 'Emma',
    'zh': 'Emma_zh',
    'ja': 'Emma_ja',
}

# const speech
SPEECH_DELAY_SECONDS = 5
DEFAULT_VOICE = 'en-US-EmmaMultilingualNeural'
# DEFAULT_VOICE = 'zh-CN-XiaoxiaoNeural'
AUDIO_FILE_SUFFIX = 'wav' if USING_LOCAL_TTS else 'mp3'

CONST_SPEECH_MAP = {
    'BOOTED': {
        'txt': 'Dorylus Agent start suceessfully.',
        'file': os.path.join(SPEECH_DIR, f"booted_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'CONNECTED': {
        'txt': 'Successfully accessed the AI computing power network.',
        'file': os.path.join(SPEECH_DIR, f"connected_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'WS_CONNECTED': {
        'txt': 'Greeting. Dorylus Agent\'s here.',
        'file': os.path.join(SPEECH_DIR, f"ws_connected_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'PROMPT_RECOGNIZED': {
        'txt': 'Glad to hear your voice.',
        'file': os.path.join(SPEECH_DIR, f"prompt_recognized_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'PROMPT_NOT_RECOGNIZED': {
        'txt': 'Please provide correct prompt.',
        'file': os.path.join(SPEECH_DIR, f"prompt_not_recognized_speech.{AUDIO_FILE_SUFFIX}"),
    },
    
}

# Knowledge base searching selection
INDEX_TYPE_SELECTION = 'solana'

NODE_API_GATEWAY = 'http://127.0.0.1:8080'

NODE_API_AUTH = f'{NODE_API_GATEWAY}/auth'

# local llm
LOCAL_LLM_SERVER = f'{NODE_API_GATEWAY}/llm'
# LOCAL_LLM_SERVER = f'http://127.0.0.1:11434'
LOCAL_LLM_MODEL = 'llama3' # llama3, llama2:13b, gemma:7b

class ChatModels(Enum):
    OPENAI = 0,
    OLLAMA = 1,

DEFAULT_CHAT_MODEL = ChatModels.OLLAMA

TAVILY_API_KEY = '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'
SERP_API_KEY = 'e53ba6dbd390cadf9510af1c64ff08236f7cbb9a25b9053ffb8528e046d078e6'
LOCAL_SEARCH_ENGINE = 'http://127.0.0.1:7000/{engine}/search'

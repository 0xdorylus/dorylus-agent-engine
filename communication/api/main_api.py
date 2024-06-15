import asyncio
import time
import base64
from fastapi.responses import StreamingResponse

from configs import logger, USING_LOCAL_TTS
from communication.manager.local_manager import g_local_manager
from communication.server_manager import g_server_manager
from communication.agents.chat.workflow import ChatAgentWorkflow
from communication.agents.dorylus.workflow import DorylusAgentWorkflow
from communication.models import *
from model_api.tts import fetch_audio_chunks_from_stream
from model_api.lang_detect import get_voice_by_lang


async def chat_agent_recv_audio_chunks(chunk: bytes, context: ChatAgentConnContext):
    result = context.audio_processor.recv_data(chunk)
    if not result:
        return
    speech_data, stream_pcmf32, stream_end_of_line = result
    if stream_pcmf32:
        g_server_manager.put_whisper_stream(stream_pcmf32, stream_end_of_line, context)

    if not speech_data:
        return
    
    # send speech_data to handle
    context.audio_processor.stop()
    await context.websocket.send(
        WsBase[object](msgType='record_pause').model_dump_json(exclude_none=True)
    )
    await g_server_manager.put_speech(speech_data, context)


async def asr_stream_recv_audio_chunks(chunk: bytes, context: ChatAgentConnContext):
    result = context.audio_processor.recv_data(chunk)
    if not result:
        return
    speech_data, stream_pcmf32, stream_end_of_line = result
    if stream_pcmf32:
        g_server_manager.put_whisper_stream(stream_pcmf32, stream_end_of_line, context)

    if not speech_data:
        return
    
    lang, transcribe_result = await asyncio.to_thread(g_local_manager.process_speech, speech_data)

    available_langs = {'en', 'ja', 'zh'}
    if lang not in available_langs:
        return
    if not transcribe_result:
        return
    g_server_manager.put_ws_response(WsBase[str](
        msgType='asr_transcribe',
        data=transcribe_result,
    ), context)


async def process_speech(speech_data: bytes, context: ChatAgentConnContext):
    start_tm = time.perf_counter()
    
    logger.info(f'speech data len: {len(speech_data)}')

    lang, transcribe_result = await asyncio.to_thread(g_local_manager.process_speech, speech_data)

    available_langs = {'en', 'ja', 'zh'}
    if lang not in available_langs:
        logger.info(f'lang({lang}) not in {available_langs}, ignored.')
        return
    if not transcribe_result:
        logger.info('Got blank audio, ignored.')
        return
    
    g_server_manager.put_ws_response(WsBase[str](
        msgType='asr_transcribe',
        data=transcribe_result,
    ), context)
    
    g_server_manager.put_event('asr', {
        'text': transcribe_result,
    })

    workflow = ChatAgentWorkflow()
    workflow_result = await workflow.ainvoke({
        'context': context, 
        'question': transcribe_result
    })
    logger.info(f'workflow_result: [{workflow_result}]')

    # kb_type = g_local_manager.get_index_type_sel()
    # kb_rsp = await search_docs_from_kb(WsBase[SearchDocsFromKBRequest](
    #     msgType="search_docs_from_kb",
    #     data=SearchDocsFromKBRequest(
    #         kb_type=kb_type,
    #         query=transcribe_result,
    #     ),
    # ))
    # kb_chunks = kb_rsp.data
    # logger.info(f'kb_chunks: [{kb_chunks}]')
    # if not kb_chunks:
    #     llm_result = await asyncio.to_thread(g_local_manager.simple_chat, transcribe_result)
    # else:
    #     llm_result = await asyncio.to_thread(g_local_manager.kb_chat, transcribe_result, kb_chunks)
    # logger.info(f'llm_result: [{llm_result}]')

    # g_server_manager.put_ws_response(WsBase[str](
    #     msgType='llm_result',
    #     data=llm_result,
    # ), context)
    # g_server_manager.put_event('llm_service', {
    #     'text': llm_result,
    # })

    elapsed = time.perf_counter() - start_tm
    logger.info(f'Before audio playing, elapsed: {elapsed}')

    # llm_result_one_line = llm_result.strip().replace("\n", " ")
    # voice = get_voice_by_lang(llm_result_one_line)
    # async for audio_chunks in fetch_audio_chunks_from_stream(llm_result, voice):
    #     g_server_manager.put_ws_response(audio_chunks, context)
    # g_server_manager.put_ws_response(WsBase[object](msgType='audio_end'), context)

    # g_server_manager.put_event('tts', {
    #     'text': 'tts success',
    # })

async def dorylus_agent_clear(x_agent_user_id: str):
    g_local_manager.rmv_dorylus_agent_context(x_agent_user_id)
    return {
        'result': 'ok',
    }

async def dorylus_agent_text(body: DorylusAgentTextRequest, x_agent_user_id: str):
    context = await g_local_manager.get_dorylus_agent_context(x_agent_user_id)
    if context is None:
        return {
            'result': 'Failed to get agent context.',
        }

    workflow = DorylusAgentWorkflow()
    workflow_result = await workflow.ainvoke({
        'context': context, 
        'question': body.question,
    })
    logger.info(f'dorylus_agent_text workflow_result: [{workflow_result}]')

    return DorylusAgentTextResponse(
        answer=workflow_result.get('generation', ''),
    )

async def dorylus_agent_audio(file: bytes, x_agent_user_id: str):
    if len(file) <= 44:
        return {
            'result': 'Malformed audio',
        }

    _, transcribe_result = await asyncio.to_thread(g_local_manager.process_speech, file[44:])
    if not transcribe_result:
        logger.info('Got blank audio, ignored.')
        return {
            'result': 'Got blank audio',
        }
    logger.info(f'dorylus_agent_audio transcribe_result: [{transcribe_result}]')

    context = await g_local_manager.get_dorylus_agent_context(x_agent_user_id)
    if context is None:
        return {
            'result': 'Failed to get agent context.',
        }

    workflow = DorylusAgentWorkflow()
    workflow_result = await workflow.ainvoke({
        'context': context, 
        'question': transcribe_result,
    })
    logger.info(f'dorylus_agent_audio workflow_result: [{workflow_result}]')

    generation = workflow_result.get('generation', None)
    if generation is None:
        return {
            'result': 'Generation error',
        }

    generation_one_line = generation.strip().replace("\n", " ")
    voice = get_voice_by_lang(generation_one_line)

    async def _audio_streamer():
        async for audio_chunks in fetch_audio_chunks_from_stream(generation, voice):
            yield audio_chunks
    
    return StreamingResponse(
        content=_audio_streamer(),
        media_type='audio/wav' if USING_LOCAL_TTS else 'audio/mpeg',
    )

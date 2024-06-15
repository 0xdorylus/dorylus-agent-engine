import asyncio
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from typing import List, Dict

from communication.models import *
from communication.manager.local_manager import g_local_manager
from communication.server_manager import g_server_manager
import communication.agents.utils as agent_utils
from model_api.tts import async_fetch_const_audio_stream
from configs import (
    logger,
)
from .prompts import PROMPT_TEMPLATES

class GraphState(TypedDict):
    """
    Represents the state of lang graph.
    """
    context: ChatAgentConnContext
    question : str
    kb_type: int
    search_keywords: str
    documents : List[str]
    generation : str

async def route_prompt_recognize(state):
    context: ChatAgentConnContext = state['context']
    if context.status == PromptStatus.NO_PROMPT:
        return 'no_prompt'
    else:
        return 'have_prompt'

async def prompt_recognize(state):
    context: ChatAgentConnContext = state['context']
    question = state['question']
    
    answer = await asyncio.to_thread(
        context.conversation.judge, 
        PROMPT_TEMPLATES['prompt_recognize'], 
        {'question': question},
    )
    if answer['recognized']:
        context.status = PromptStatus.HAVE_PROMPT
        audio_key = 'PROMPT_RECOGNIZED'
    else:
        audio_key = 'PROMPT_NOT_RECOGNIZED'
    async for audio_chunks in async_fetch_const_audio_stream(audio_key):
        g_server_manager.put_ws_response(audio_chunks, context)

    return {'context': context, 'question': question}

async def kb_select(state):
    context: ChatAgentConnContext = state['context']
    question = state['question']

    index_types = g_local_manager.get_all_index_types()
    kb_type = -1
    search_keywords = ''
    if len(index_types) > 0:
        index_types_str = ','.join(map(lambda x: x.name, index_types))
        answer = await asyncio.to_thread(
            context.conversation.judge,
            PROMPT_TEMPLATES['kb_select'], 
            {'index_types': index_types_str, 'question': question},
        )
        index_type_name = answer['selection'].lower()
        search_keywords = answer['keywords']
        for x in index_types:
            if x.lower() == index_type_name:
                kb_type = x.id
                break
    else:
        answer = await asyncio.to_thread(
            context.conversation.judge,
            PROMPT_TEMPLATES['extract_keywords'], 
            {'question': question},
        )
        search_keywords = answer['keywords']

    return {'context': context, 'question': question, 'kb_type': kb_type, 'search_keywords': search_keywords}

async def decide_search_path(state):
    kb_type = state["kb_type"]
    if kb_type > 0:
        return 'kb_search'
    else:
        return 'web_search'

async def kb_search(state):
    context: ChatAgentConnContext = state['context']
    question = state['question']
    kb_type = state['kb_type']
    search_keywords = state['search_keywords']

    kb_rsp = await agent_utils.search_docs_from_kb(WsBase[SearchDocsFromKBRequest](
        msgType="search_docs_from_kb",
        data=SearchDocsFromKBRequest(
            kb_type=kb_type,
            query=search_keywords,
        ),
    ))
    kb_chunks = kb_rsp.data
    documents = list(map(lambda x: x['content'].strip(), kb_chunks))
    return {
        'context': context, 
        'question': question, 
        'kb_type': kb_type, 
        'search_keywords': search_keywords,
        'documents': documents,
    }

async def web_search(state):
    context: ChatAgentConnContext = state['context']
    question = state['question']
    kb_type = state['kb_type']
    search_keywords = state['search_keywords']

    documents = []

    return {
        'context': context, 
        'question': question, 
        'kb_type': kb_type, 
        'search_keywords': search_keywords,
        'documents': documents,
    }

async def grade_documents(state):
    context: ChatAgentConnContext = state['context']
    question = state["question"]
    kb_type = state['kb_type']
    search_keywords = state['search_keywords']
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    for doc in documents:
        answer = await asyncio.to_thread(
            context.conversation.judge,
            PROMPT_TEMPLATES['grade_documents'], 
            {'question': question, 'document': doc},
        )
        grade = answer['score']
        # Document relevant
        if grade.lower() == "yes":
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        kb_type = -1

    return {
        'context': context, 
        'question': question, 
        'kb_type': kb_type, 
        'search_keywords': search_keywords,
        'documents': documents,
    }

async def decide_to_generate(state):
    documents = state['documents']
    if len(documents) == 0:
        return 'web_search'
    else:
        return 'generate'

async def generate(state):
    context: ChatAgentConnContext = state['context']
    question = state['question']
    kb_type = state['kb_type']
    search_keywords = state['search_keywords']
    documents = state["documents"]

    if not documents:
        generation = await asyncio.to_thread(context.conversation.simple_chat, question, context.history)
    else:
        reference = '\n'.join(documents)
        generation = await asyncio.to_thread(context.conversation.kb_chat, question, reference, context.history)

    return {
        'context': context, 
        'question': question, 
        'kb_type': kb_type, 
        'search_keywords': search_keywords,
        'documents': documents,
        'generation': generation,
    }

async def grade_generation_v_documents_and_question(state):
    context: ChatAgentConnContext = state['context']
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    if not documents:
        return await grade_generation_v_question(context, question, generation)

    # Check hallucination
    reference = '\n'.join(documents)
    answer = await asyncio.to_thread(
        context.conversation.judge,
        PROMPT_TEMPLATES['check_hallucination'], 
        {'reference': reference, 'generation': generation},
    )
    grade = answer['score']

    if grade == "yes":
        return await grade_generation_v_question(context, question, generation)
    else:
        return "not supported"
    
async def grade_generation_v_question(context: ChatAgentConnContext, question: str, generation: str):
    # Check question-answering
    answer = await asyncio.to_thread(
        context.conversation.judge,
        PROMPT_TEMPLATES['check_question_answering'], 
        {'generation': generation, 'question': question},
    )
    grade = answer['score']
    if grade == "yes":
        context.history.append((question, generation))
        return "useful"
    else:
        return "not useful"

class ChatAgentWorkflow():
    def __init__(self):
        workflow = StateGraph(GraphState)
        workflow.add_node('prompt_recognize', prompt_recognize)
        workflow.add_node('kb_select', kb_select)
        workflow.add_node('kb_search', kb_search)
        workflow.add_node('web_search', web_search)
        workflow.add_node('grade_documents', grade_documents)
        workflow.add_node('generate', generate)

        workflow.set_conditional_entry_point(
            route_prompt_recognize,
            {
                'no_prompt': 'prompt_recognize',
                'have_prompt': 'kb_select',
            },
        )
        workflow.add_edge('prompt_recognize', END)
        workflow.add_conditional_edges(
            'kb_select',
            decide_search_path,
            {
                'web_search': 'web_search',
                'kb_search': 'kb_search',
            },
        )
        workflow.add_edge('web_search', 'generate')
        workflow.add_edge('kb_search', 'grade_documents')
        workflow.add_conditional_edges(
            'grade_documents',
            decide_to_generate,
            {
                'web_search': 'web_search',
                'generate': 'generate',
            },
        )
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "web_search",
            },
        )

        self._app = workflow.compile()
    
    async def ainvoke(self, input):
        return await self._app.ainvoke(input)



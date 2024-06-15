import asyncio
import json
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from typing import List, Dict

from configs import (
    logger,
)
from .prompts import PROMPT_TEMPLATES
from communication.models import *
from model_api.web_search import *

class GraphState(TypedDict):
    """
    Represents the state of lang graph.
    """
    context: DorylusAgentContext
    question: str
    intention: str
    search_keywords: str
    documents : List[str]
    generation : str

async def intention_recognize(state):
    context: DorylusAgentContext = state['context']
    question = state['question']

    cryptocurrency = ''
    if context.intention is None:
        answer = await asyncio.to_thread(
            context.conversation.judge, 
            PROMPT_TEMPLATES['intent_recognition'], 
            {'question': question},
        )

        intention = answer['intention']
        cryptocurrency = answer['coin'] if 'coin' in answer else ''
        if intention == 'nft_contract':
            context.intention = intention
            context.state['nft'] = {
                'query': '',
                'all_info': {},
                'confirm_prompt': False,
            }
    else:
        intention = context.intention

    context.state['cryptocurrency'] = cryptocurrency
    return {
        'context': context,
        'question': question,  
        'intention': intention,
    }

async def decide_opt_path(state):
    intention = state['intention']
    intention_map = {
        'crypto_price': 'query_crypto_price',
        'nft_contract': 'build_nft_contract',
        'gossip': 'simple_chat',
        'explore': 'extract_keywords',
    }
    
    return intention_map[intention]

async def query_crypto_price(state):
    context: DorylusAgentContext = state['context']
    question = state['question']
    intention = state['intention']
    
    cryptocurrency = context.state['cryptocurrency']

    url = f'https://p2p.hajime.ai/binance-api/api/v3/ticker/price'
    params = {
        'symbol': f'{cryptocurrency}USDT'.upper()
    }
    documents = []
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                content = await resp.text()
                documents = [
                    f'Successfully fetching realtime price from market: {content}'
                ]

            else:
                content = await resp.text()
                logger.error(f'query_crypto_price {url} fail: {content}')
                documents = [
                    f'Error occured when fetching realtime price from market: {content}'
                ]

    return {
        'context': context,
        'question': question,  
        'intention': intention,
        'documents': documents,
    }

async def generate_crypto_price_answer(state):
    context: DorylusAgentContext = state['context']
    question = state['question']
    intention = state['intention']
    documents = state['documents']

    cryptocurrency = context.state['cryptocurrency']

    generation = await asyncio.to_thread(
        context.conversation.generate,
        PROMPT_TEMPLATES['crypto_price_answer'], 
        {
            'reference': '\n'.join(documents),
            'cryptocurrency': cryptocurrency,
        },
    )

    context.history.append((question, generation))
    return {
        'context': context,
        'question': question,  
        'intention': intention,
        'documents': documents,
        'generation': generation,
    }

nft_info_collect_steps = [
    {
        'field': 'name',
        'query_item': 'Name of NFT',
        'value_type': 'string',
    },
    {
        'field': 'quantity',
        'query_item': 'Quantity of NFT demand',
        'value_type': 'digits without comma',
    },
    {
        'field': 'chain',
        'query_item': 'On which blockchain is NFT issued',
        'value_type': 'string',
    },
]

async def build_nft_contract(state):
    context: DorylusAgentContext = state['context']
    question = state['question']
    intention = state['intention']
    
    nft_state = context.state['nft']
    query = nft_state['query']
    all_info = nft_state['all_info']
    confirm_prompt = nft_state['confirm_prompt']

    for step in nft_info_collect_steps:
        field = step['field']
        query_item = step['query_item']
        value_type = step['value_type']

        if field in all_info: # got this field
            continue

        if not query:
            result = context.conversation.judge(
                PROMPT_TEMPLATES['nft_generate_query'],
                {
                    'query_item': query_item,
                }
            )

            generation = result['query']
            nft_state['query'] = generation
            break
        else:
            result = context.conversation.judge(
                PROMPT_TEMPLATES['nft_reply_parser'],
                {
                    'value_type': value_type,
                    'query': query,
                    'reply': question,
                }
            )
            if ('info' in result) and (result['info'] is not None):
                all_info[field] = result['info']
                query = ''
                nft_state['query'] = ''
                continue
            else:
                result = context.conversation.judge(
                    PROMPT_TEMPLATES['nft_generate_query_again'],
                    {
                        'query': query,
                    }
                )
                generation = result['query']
                break
    
    if len(nft_info_collect_steps) == len(all_info):
        if not confirm_prompt:
            info = ', '.join(map(lambda x: f'{x["query_item"]}: {all_info[x["field"]]}', nft_info_collect_steps))
            result = context.conversation.judge(
                PROMPT_TEMPLATES['nft_generate_confirm'],
                {
                    'info': info,
                }
            )
            generation = result['query']
            nft_state['confirm_prompt'] = True
        else:
            result = context.conversation.judge(
                PROMPT_TEMPLATES['nft_confirm_parser'],
                {
                    'reply': question,
                }
            )
            generation = result['statement']
            confirmed = result['confirmed']
            if confirmed == 'yes':
                os.system(f'echo "{json.dumps(all_info)}"')
                pass
            
            context.intention = None
            del context.state['nft']

    context.history.append((question, generation))
    return {
        'context': context,
        'question': question,  
        'intention': intention,
        'generation': generation,
    }

async def simple_chat(state):
    context: DorylusAgentContext = state['context']
    question = state['question']
    intention = state['intention']
    documents = state['documents']

    generation = await asyncio.to_thread(context.conversation.simple_chat, question, context.history)
    context.history.append((question, generation))
    return {
        'context': context,
        'question': question,  
        'intention': intention,
        'documents': documents,
        'generation': generation,
    }

async def extract_keywords(state):
    context: DorylusAgentContext = state['context']
    question = state['question']

    answer = await asyncio.to_thread(
        context.conversation.judge,
        PROMPT_TEMPLATES['extract_keywords'], 
        {'question': question},
    )
    search_keywords = answer['keywords']

    return {
        'context': context,
        'question': question,  
        'search_keywords': search_keywords,
    }

async def web_search(state):
    context: DorylusAgentContext = state['context']
    question = state['question']
    search_keywords = state['search_keywords']

    documents = await serp_api_search(search_keywords)

    return {
        'context': context,
        'question': question,  
        'search_keywords': search_keywords,
        'documents': documents,
    }

async def generate(state):
    context: DorylusAgentContext = state['context']
    question = state['question']
    search_keywords = state['search_keywords']
    documents = state["documents"]

    if not documents:
        generation = await asyncio.to_thread(context.conversation.simple_chat, question, context.history)
    else:
        reference = '\n'.join(documents)
        generation = await asyncio.to_thread(context.conversation.kb_chat, question, reference, context.history)

    context.history.append((question, generation))
    return {
        'context': context,
        'question': question,  
        'search_keywords': search_keywords,
        'documents': documents,
        'generation': generation,
    }

class DorylusAgentWorkflow():
    def __init__(self):
        workflow = StateGraph(GraphState)
        workflow.add_node('intention_recognize', intention_recognize)
        workflow.add_node('query_crypto_price', query_crypto_price)
        workflow.add_node('generate_crypto_price_answer', generate_crypto_price_answer)
        workflow.add_node('build_nft_contract', build_nft_contract)
        workflow.add_node('simple_chat', simple_chat)
        workflow.add_node('extract_keywords', extract_keywords)
        workflow.add_node('web_search', web_search)
        workflow.add_node('generate', generate)

        workflow.set_entry_point('intention_recognize')
        workflow.add_conditional_edges(
            'intention_recognize',
            decide_opt_path,
            {
                'query_crypto_price': 'query_crypto_price',
                'build_nft_contract': 'build_nft_contract',
                'simple_chat': 'simple_chat',
                'extract_keywords': 'extract_keywords',
            },
        )

        workflow.add_edge('query_crypto_price', 'generate_crypto_price_answer')
        workflow.add_edge('generate_crypto_price_answer', END)

        workflow.add_edge('build_nft_contract', END)

        workflow.add_edge('simple_chat', END)
        
        workflow.add_edge('extract_keywords', 'web_search')
        workflow.add_edge('web_search', 'generate')
        workflow.add_edge('generate', END)

        self._app = workflow.compile()
    
    async def ainvoke(self, input):
        return await self._app.ainvoke(input)

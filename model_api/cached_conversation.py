import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import (
    PromptTemplate,
)
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import Callable, List, Dict, Tuple, Iterable, Any

from configs import (
    logger, COMMON_PROMPT_TEMPLATES, OPENAI_PROXY,
    LOCAL_LLM_SERVER, LOCAL_LLM_MODEL, ChatModels, DEFAULT_CHAT_MODEL
)

class UnsupportedChatModelException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f'UnsupportedChatModelException: {self.message}'

class TokenUsageCallbackHandler(BaseCallbackHandler):
    def __init__(self, token_usage_handler: Callable = None):
        super().__init__()
        self.token_usage_handler = token_usage_handler

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # print('Response in callback')
        # print(response)
        # print()

        generation = response.generations[0][0]
        gen_info = generation.generation_info

        # get token usage
        token_usage = gen_info.get('prompt_eval_count', 0) + gen_info.get('eval_count', 0)
        # get time costed (local machine)
        # instead of getting total duration, we get the prompt_eval_duration and eval_duration to exclude the load duration (e.g. to load the model to the gpu, etc.)
        time_costed = gen_info.get('prompt_eval_duration', 1e-10) + gen_info.get('eval_duration', 1e-10)     # in ns, a small value to indicate a inf time when it fails


        # create an object to store the token usage and time costed
        token_usage_obj = {
            'token_usage': token_usage,
            'time_costed': time_costed
        }
        if self.token_usage_handler:
            self.token_usage_handler(token_usage_obj)

class CachedConversation():
    def __init__(self, model: ChatModels=None, access_token: str=None):
        temperature = 0
        if model is None:
            model = DEFAULT_CHAT_MODEL
        self._model = model
        if model == ChatModels.OPENAI:
            self._llm = ChatOpenAI(
                temperature=temperature,
                api_key=os.environ['CHAT_OPENAI_KEY'],
                base_url='https://api.aiproxy.io/v1',
                # base_url='https://p2p.hajime.ai/openai-api/v1',
                model='gpt-4',
                # model='gpt-3.5-turbo',
                openai_proxy=OPENAI_PROXY,
            )
        elif model == ChatModels.OLLAMA:
            self._llm = ChatOllama(
                temperature=temperature,
                base_url=LOCAL_LLM_SERVER, 
                model=LOCAL_LLM_MODEL,
                headers={
                    'Authorization': f'bearer {access_token}',
                },
            )
        else:
            raise UnsupportedChatModelException(str(model))

        self._verbose = True
    
    def set_access_token(self, access_token):
        self._llm.headers = {
            'Authorization': f'bearer {access_token}',
        }

    def simple_chat(self, question: str, history: Iterable[Tuple[str, str]], token_usage_handler: Callable=None):
        history_prompt = ''
        for h in history:
            history_prompt += COMMON_PROMPT_TEMPLATES['history'].format(question=h[0], answer=h[1])

        prompt = PromptTemplate.from_template(
            COMMON_PROMPT_TEMPLATES['chat']
        )

        conversation = LLMChain(
            llm=self._llm, 
            prompt=prompt, 
            verbose=self._verbose,
        )
        
        callbacks = None
        if self._model == ChatModels.OLLAMA and token_usage_handler is not None:
            callbacks = [
                TokenUsageCallbackHandler(token_usage_handler=token_usage_handler)
            ]
        return conversation.predict(callbacks=callbacks, question=question, history=history_prompt)

    def kb_chat(self, question: str, reference: str, history: Iterable[Tuple[str, str]], token_usage_handler: Callable=None):
        history_prompt = ''
        for h in history:
            history_prompt += COMMON_PROMPT_TEMPLATES['history'].format(question=h[0], answer=h[1])

        prompt = PromptTemplate.from_template(
            COMMON_PROMPT_TEMPLATES['kb_chat']
        )
        conversation = LLMChain(
            llm=self._llm, 
            prompt=prompt, 
            verbose=self._verbose
        )

        callbacks = None
        if self._model == ChatModels.OLLAMA and token_usage_handler is not None:
            callbacks = [
                TokenUsageCallbackHandler(token_usage_handler=token_usage_handler)
            ]
        return conversation.predict(callbacks=callbacks, question=question, context=reference, history=history_prompt)

    def judge(self, template: str, input: Dict[str, Any]):
        prompt = PromptTemplate.from_template(template)
        
        chain = prompt | self._llm | JsonOutputParser()
        return chain.invoke(input)
        # chain = LLMChain(
        #     llm=self._llm, 
        #     prompt=prompt, 
        #     verbose=self._verbose
        # )
        
        # answer = chain.predict(**input)
        # print(answer)
        # return JsonOutputParser().parse(answer)
    
    def generate(self, template: str, input: Dict[str, Any]):
        prompt = PromptTemplate.from_template(template)
        
        chain = prompt | self._llm | StrOutputParser()
        
        return chain.invoke(input)

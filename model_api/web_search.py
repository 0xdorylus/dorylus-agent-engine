import os
import aiohttp
import asyncio
from langchain_community.tools.tavily_search import TavilySearchResults
import serpapi

from configs import logger, TAVILY_API_KEY, SERP_API_KEY, LOCAL_SEARCH_ENGINE

os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

async def tavily_search(question: str):
    web_search_tool = TavilySearchResults(k=3)
    docs = await web_search_tool.ainvoke({"query": question})
    return [d["content"] for d in docs]

async def baidu_search(question: str):
    url = LOCAL_SEARCH_ENGINE.format(engine='baidu')
    params = {
        'lang': 'EN',
        'limit': 5,
        'text': question,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                content = await resp.json()
                return list(map(lambda x: x['description'], content))

            else:
                content = await resp.text()
                logger.error(f'baidu_search {url} fail: {content}')
                return []

async def serp_api_search(question: str):
    params = {
        "engine": "google",
        "q": question,
        "location": "Seattle-Tacoma, WA, Washington, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "num": "10",
        "safe": "active",
    }


    client = serpapi.Client(api_key=SERP_API_KEY)
    results = await asyncio.to_thread(client.search, params)

    return list(map(lambda x: x['snippet'], results['organic_results']))

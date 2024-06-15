import asyncio

from communication.local import local_ws_server, local_restful_server
from communication.central import start_central_connection
from communication.neighbor import neighbor_restful_server, neighbor_ws_server
from communication.working_loop import agent_speech_processing_loop, ws_responsing_loop

async def main():
    tasks = [
        local_ws_server(),
        local_restful_server(),
        start_central_connection(),
        neighbor_restful_server(),
        neighbor_ws_server(),

        agent_speech_processing_loop(),
        ws_responsing_loop(),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import websockets
import json
from ragfile import get_answer


async def handle_query(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        query = data.get("query")
        collection_name = data.get("collection_name")
        document_ids = data.get("document_ids")

        if query and collection_name and document_ids:
            response = get_answer(query, collection_name, document_ids)
            await websocket.send(json.dumps(response))


async def main():
    async with websockets.serve(handle_query, "localhost", 8080):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

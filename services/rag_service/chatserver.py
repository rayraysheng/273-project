# services/rag_service/chatserver.py
import asyncio
import websockets
import json
from ragfile import get_answer


async def handle_query(websocket, path):
    chat_history = []

    async for message in websocket:
        data = json.loads(message)
        manual = data.get("manual")
        role = data.get("role")
        content = data.get("content")

        if manual and role == "user" and content:
            # Save the user's message to the chat history
            chat_history.append({"role": "user", "content": content})

            # Get the response from the model, including the chat history
            response = await get_answer(manual, role, content, chat_history)
            chat_history.append(
                {"role": "system", "content": response["data"]["output_text"]}
            )

            # Send the response back to the client
            await websocket.send(json.dumps(response))


async def main():
    async with websockets.serve(handle_query, "localhost", 8080):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())

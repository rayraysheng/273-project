import asyncio
import websockets
import json
import logging
from ragfile import get_answer

logging.basicConfig(level=logging.INFO)


async def handle_query(websocket, path):
    logging.info("New connection established.")
    chat_history = []
    try:
        async for message in websocket:
            logging.info(f"Received message: {message}")
            if not message:
                logging.error("Received empty message")
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON: {e}")
                continue

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
                logging.info(f"Sent response: {response}")
    except websockets.exceptions.ConnectionClosed:
        logging.info("Client Disconnected. History Wiped.")


async def main():
    async with websockets.serve(handle_query, "0.0.0.0", 8081):
        logging.info("WebSocket server started on port 8081")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())

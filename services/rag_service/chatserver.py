import json
import logging
from aiohttp import web
import aiohttp_cors
from ragfile import get_answer

logging.basicConfig(level=logging.INFO)


async def handle_query(request):
    websocket = web.WebSocketResponse()
    await websocket.prepare(request)

    logging.info("New connection established.")
    chat_history = []
    try:
        async for message in websocket:
            logging.info(f"Received message: {message}")
            if not message:
                logging.error("Received empty message")
                continue
            try:
                data = json.loads(message.data)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON: {e}")
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            manual = data.get("manual")
            role = data.get("role")
            content = data.get("content")

            if manual and role == "user" and content:
                # Save the user's message to the chat history
                chat_history.append({"role": "user", "content": content})

                try:
                    # Get the response from the model, including the chat history
                    response = await get_answer(manual, role, content, chat_history)
                    chat_history.append(
                        {"role": "system", "content": response["data"]["output_text"]}
                    )

                    # Send the response back to the client
                    await websocket.send_json(response)
                    logging.info(f"Sent response: {response}")
                except Exception as e:
                    logging.error(f"Error processing request: {e}")
                    await websocket.send_json({"error": "Internal server error"})
    except web.WebSocketDisconnect:
        logging.info("Client Disconnected. History Wiped.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    return websocket


app = web.Application()
cors = aiohttp_cors.setup(
    app,
    defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    },
)

cors.add(app.router.add_get("/ws", handle_query))

if __name__ == "__main__":
    web.run_app(app, port=8081)

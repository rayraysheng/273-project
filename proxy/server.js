const express = require("express");
const { Server } = require("socket.io");
const WebSocket = require("ws");
const http = require("http");

const app = express();
const server = http.createServer(app);

// Socket.IO server (for client)
const io = new Server(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"],
    credentials: true,
  },
});

// Connect to Python WebSocket server
const ws = new WebSocket("ws://localhost:8081/ws");

// Bridge Socket.IO and WebSocket
io.on("connection", (socket) => {
  console.log("Client connected to Socket.IO");

  // Forward messages from Socket.IO to WebSocket
  socket.on("message", (data) => {
    console.log("Received message from client:", data);
    ws.send(JSON.stringify(data));
  });

  // Forward messages from WebSocket to Socket.IO
  ws.on("message", (data) => {
    console.log("Received message from server:", JSON.parse(data));
    const message = JSON.parse(data);
    if (message.error) {
      socket.emit("message", { text: "Error: " + message.error });
    }
    const response = {
      text: message.data.output_text["output_text"],
    };
    console.log("response", response);
    socket.emit("message", response);
  });
});

io.on("disconnect", () => {
  console.log("Client disconnected from Socket.IO");
});

server.listen(8082, () => {
  console.log("Proxy running on port 8082");
});

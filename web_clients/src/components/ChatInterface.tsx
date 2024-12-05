"use client";

import {
  ChatBubble,
  ChatBubbleAction,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "@/components/ui/chat/chat-bubble";
import { AnimatePresence, motion } from "framer-motion";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ChatInput } from "@/components/ui/chat/chat-input";
import { ChatMessageList } from "@/components/ui/chat/chat-message-list";

import { SetStateAction, useEffect, useRef, useState } from "react";

import { GitHubLogoIcon } from "@radix-ui/react-icons";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CodeDisplayBlock from "@/components/code-display-block";
import io, { Socket } from "socket.io-client";

import UploadPage from "./Upload";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface Message {
  id: string;
  manual: string;
  role: "user" | "assistant";
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [messageStatuses, setMessageStatuses] = useState(new Map());
  const [manual, setManual] = useState<string>("");
  const [isConnected, setIsConnected] = useState(false);

  const messagesRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const socketIo = io("http://localhost:8080", {
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketIo.on("connect", () => {
      console.log("Connected to Socket.IO");
      setIsConnected(true);
    });

    socketIo.on("disconnect", () => {
      console.log("Disconnected from Socket.IO");
      setIsConnected(false);
    });

    socketIo.on("connect_error", (error) => {
      console.error("Socket.IO connection error:", error);
      setIsConnected(false);
    });

    socketIo.on("message", (receivedMessage) => {
      try {
        const newMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          manual: manual,
          content: receivedMessage.text,
        };
        setMessages((prev) => [...prev, newMessage]);
        setIsGenerating(false);

        if (messagesRef.current) {
          messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
        }
      } catch (error) {
        console.error("Error processing message:", error);
      }
    });

    setSocket(socketIo);

    return () => {
      socketIo.disconnect();
    };
  }, []);

  const handleInputChange = (e: {
    target: { value: SetStateAction<string> };
  }) => {
    setInput(e.target.value);
  };

  const onKeyDown = (e: {
    key: string;
    shiftKey: any;
    preventDefault: () => void;
  }) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formRef.current?.requestSubmit();
    }
  };

  const onSubmit = async (e: { preventDefault: () => void }) => {
    e.preventDefault();
    if (!input.trim() || !socket || !isConnected) return;

    const messageId = Date.now().toString();
    const newMessage: Message = {
      manual: manual,
      id: messageId,
      role: "user",
      content: input.trim(),
    };

    // Update message status
    setMessageStatuses((prev) =>
      new Map(prev).set(messageId, { status: "Sending..." })
    );

    // Add message to chat
    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setIsGenerating(true);

    // Send message through WebSocket
    try {
      const messageData = {
        text: input.trim(),
        timestamp: new Date().toISOString(),
        manual: manual, 
        type: "outgoing",
      };
      socket.send(messageData);

      // Update message status to sent
      setMessageStatuses((prev) =>
        new Map(prev).set(messageId, { status: "Sent ✓" })
      );
    } catch (error) {
      console.error("Error sending message:", error);
      setMessageStatuses((prev) =>
        new Map(prev).set(messageId, { status: "Failed to send ✗" })
      );
      setIsGenerating(false);
    }

    // Scroll to bottom
    // if (messagesRef.current) {
    //   messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    // }
  };

  return (
    <main className="flex h-screen bg-[#121212] w-full flex-col">
      <div className="flex flex-col h-full w-full">
        <header className="flex items-center justify-between p-4">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              Manual Reader
            </span>
            <span className="text-zinc-500">CS-273</span>
            <span
              className={`text-xs ${
                isConnected ? "text-green-500" : "text-red-500"
              }`}
            >
              {isConnected ? "● Connected" : "● Disconnected"}
            </span>
          </div>
          <div className="flex items-center">
            <GitHubLogoIcon className="size-6 text-zinc-500" />
          </div>
        </header>
        <UploadPage />

        <div className="flex h-4/5 w-full max-w-3xl flex-col items-center mx-auto py-6">
          <Select onValueChange={setManual} value={manual}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder={"Select manual"} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="option1">Option 1</SelectItem>
              <SelectItem value="option2">Option 2</SelectItem>
              <SelectItem value="option3">Option 3</SelectItem>
              <SelectItem value="option4">Option 4</SelectItem>
            </SelectContent>
          </Select>

          <ChatMessageList ref={messagesRef} className="gap-6 p-4">
            <AnimatePresence>
              {messages.length === 0 && (
                <div className="w-full p-8">
                  <h1 className="font-bold text-zinc-100">
                    Welcome to Manual Reader Bot
                  </h1>

                  <p className="text-zinc-500 text-sm pt-2">
                    This is a real-time chat application using WebSocket for
                    communication.
                  </p>
                </div>
              )}

              {messages &&
                messages.map((message, index) => (
                  <motion.div
                    key={index}
                    layout
                    initial={{ opacity: 0, scale: 1, y: 50, x: 0 }}
                    animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
                    exit={{ opacity: 0, scale: 1, y: 1, x: 0 }}
                    transition={{
                      opacity: { duration: 0.1 },
                      layout: {
                        type: "spring",
                        bounce: 0.3,
                        duration: index * 0.05 + 0.2,
                      },
                    }}
                    style={{ originX: 0.5, originY: 0.5 }}
                    className="flex flex-col gap-2 p-4"
                  >
                    <ChatBubble
                      key={index}
                      variant={message.role === "user" ? "sent" : "received"}
                      className={`rounded-lg ${
                        message.role === "assistant" ? "" : "" // Light background for user messages
                      }`}
                    >
                      {/* 👨🏽 */}
                      <Avatar>
                        <AvatarImage
                          src={""}
                          alt="Avatar"
                          className={
                            message.role === "assistant" ? "dark:invert" : ""
                          }
                        />
                        <AvatarFallback>
                          {message.role === "assistant" ? "🤖" : "👨🏽"}
                        </AvatarFallback>
                      </Avatar>
                      <ChatBubbleMessage
                        className={
                          message.role === "user"
                            ? "text-zinc-900"
                            : "text-zinc-200"
                        }
                      >
                        {message.content
                          .split("```")
                          .map((part: string, index: number) => {
                            if (index % 2 === 0) {
                              return (
                                <Markdown
                                  key={index}
                                  remarkPlugins={[remarkGfm]}
                                >
                                  {part}
                                </Markdown>
                              );
                            } else {
                              return (
                                <pre
                                  className="whitespace-pre-wrap pt-2"
                                  key={index}
                                >
                                  <CodeDisplayBlock code={part} lang="" />
                                </pre>
                              );
                            }
                          })}
                      </ChatBubbleMessage>
                    </ChatBubble>
                  </motion.div>
                ))}

              {isGenerating && (
                <ChatBubble variant="received" className="rounded-lg">
                  <ChatBubbleAvatar
                    src=""
                    fallback="🤖"
                    className="bg-[#1a1a1a]"
                  />
                  <ChatBubbleMessage isLoading className="text-zinc-200" />
                </ChatBubble>
              )}
            </AnimatePresence>
          </ChatMessageList>
        </div>

        <div className="flex-1 p-4">
          <form
            ref={formRef}
            onSubmit={onSubmit}
            className="relative flex flex-col items-center"
          >
            <ChatInput
              value={input}
              onKeyDown={onKeyDown}
              onChange={handleInputChange}
              placeholder="Type your message here..."
              className="w-1/3 resize-none rounded-lg bg-[#1a1a1a] border-0 p-4 text-zinc-200 placeholder:text-zinc-500 focus-visible:ring-0"
            />
          </form>
        </div>
      </div>
    </main>
  );
}

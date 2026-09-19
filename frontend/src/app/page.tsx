"use client";

import React, { useState, useEffect } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useParticipants,
  useRoomContext,
  useDataChannel,
  useLocalParticipant,
} from "@livekit/components-react";
import { Mic, MicOff, PhoneOff, MessageSquare, Send, Bot, User, Radio } from "lucide-react";

interface ChatMessage {
  id: string;
  sender: string;
  text: string;
  timestamp: string;
  isBot?: boolean;
  botType?: "dost" | "sathi";
}

export default function Home() {
  const [token, setToken] = useState<string>("");
  const [roomName, setRoomName] = useState<string>("roxstar-voice-room");
  const [username, setUsername] = useState<string>("");
  const [joined, setJoined] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const wsUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL || "wss://demo.livekit.cloud";

  const handleJoin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) {
      setError("Please enter your name");
      return;
    }
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`/api/token?room=${encodeURIComponent(roomName)}&username=${encodeURIComponent(username)}`);
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || "Failed to fetch access token");
      }

      setToken(data.token);
      setJoined(true);
    } catch (err: any) {
      setError(err.message || "Failed to join room");
    } finally {
      setLoading(false);
    }
  };

  const handleLeave = () => {
    setToken("");
    setJoined(false);
  };

  if (!joined || !token) {
    return (
      <main className="join-container">
        <div className="join-card glass-panel">
          <div className="logo-badge">
            <Radio size={28} /> Roxstar AI Voice Room
          </div>
          <p className="sub-title">Real-Time Indian AI Assistants: Dost & Sathi</p>

          <form onSubmit={handleJoin}>
            <div className="input-group">
              <label className="input-label" htmlFor="username-input">Your Name</label>
              <input
                id="username-input"
                type="text"
                className="styled-input"
                placeholder="e.g. Rahul or Priya"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>

            <div className="input-group">
              <label className="input-label" htmlFor="room-input">Room Name</label>
              <input
                id="room-input"
                type="text"
                className="styled-input"
                placeholder="roxstar-voice-room"
                value={roomName}
                onChange={(e) => setRoomName(e.target.value)}
                required
              />
            </div>

            {error && <div style={{ color: "#ef4444", fontSize: "0.85rem", marginBottom: "12px" }}>{error}</div>}

            <button id="join-room-btn" type="submit" className="join-btn" disabled={loading}>
              {loading ? "Connecting..." : "Join Voice Room"}
            </button>
          </form>
        </div>
      </main>
    );
  }

  return (
    <LiveKitRoom
      video={false}
      audio={true}
      token={token}
      serverUrl={wsUrl}
      data-lk-theme="default"
      onDisconnected={handleLeave}
      style={{ height: "100vh" }}
    >
      <RoomContent roomName={roomName} username={username} onLeave={handleLeave} />
      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}

function RoomContent({ roomName, username, onLeave }: { roomName: string; username: string; onLeave: () => void }) {
  const room = useRoomContext();
  const participants = useParticipants();
  const { localParticipant } = useLocalParticipant();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMsg, setInputMsg] = useState("");

  const { send } = useDataChannel("chat", (msg) => {
    try {
      const decoded = new TextDecoder().decode(msg.payload);
      const data = JSON.parse(decoded);
      setMessages((prev) => [...prev, data]);
    } catch (e) {
      console.error("Failed to decode chat message:", e);
    }
  });

  const isMuted = !localParticipant.isMicrophoneEnabled;

  const toggleMic = async () => {
    await localParticipant.setMicrophoneEnabled(isMuted);
  };

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMsg.trim()) return;

    const chatMsg: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      sender: username,
      text: inputMsg,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const encoder = new TextEncoder();
    send(encoder.encode(JSON.stringify(chatMsg)), { reliable: true });
    setMessages((prev) => [...prev, chatMsg]);
    setInputMsg("");
  };

  return (
    <div className="room-layout">
      {/* Left Stage */}
      <div className="main-stage">
        {/* Header */}
        <div className="header-bar glass-panel">
          <div className="room-info">
            <h2 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Roxstar Voice Assistant Room</h2>
            <span className="room-badge">Room: {roomName}</span>
          </div>
          <div className="status-indicator">
            <span className="pulse-dot"></span> Live ({participants.length} Active)
          </div>
        </div>

        {/* Participants Grid */}
        <div className="participant-grid">
          {participants.map((p) => {
            const name = p.name || p.identity;
            const isDost = name.toLowerCase().includes("dost");
            const isSathi = name.toLowerCase().includes("sathi");
            const isBot = isDost || isSathi;
            const isSpeaking = p.isSpeaking;

            let cardClass = "participant-card glass-panel human";
            if (isDost) cardClass = "participant-card glass-panel dost";
            if (isSathi) cardClass = "participant-card glass-panel sathi";
            if (isSpeaking) cardClass += " speaking";

            return (
              <div key={p.sid || p.identity} className={cardClass}>
                <div className="avatar-wrapper">
                  {isBot ? <Bot size={32} /> : <User size={32} />}
                </div>
                <div className="participant-name">{name}</div>
                <div className="participant-tag">
                  {isDost ? "Roxstar AI Dost (Male Bot)" : isSathi ? "Roxstar AI Sathi (Female Bot)" : "Human Participant"}
                </div>
                <div className="mic-badge">
                  {p.isMicrophoneEnabled ? (
                    <Mic size={14} color={isSpeaking ? "#4ade80" : "#94a3b8"} />
                  ) : (
                    <MicOff size={14} color="#ef4444" />
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Controls Bar */}
        <div className="controls-bar glass-panel">
          <button
            id="mic-toggle-btn"
            className={`control-btn ${isMuted ? "" : "active"}`}
            onClick={toggleMic}
          >
            {isMuted ? <MicOff size={18} /> : <Mic size={18} />}
            {isMuted ? "Unmute Mic" : "Mute Mic"}
          </button>

          <button id="leave-room-btn" className="control-btn danger" onClick={onLeave}>
            <PhoneOff size={18} /> Leave Room
          </button>
        </div>
      </div>

      {/* Right Chat Sidebar */}
      <div className="chat-sidebar glass-panel">
        <div className="chat-header">
          <MessageSquare size={18} /> Room Text Chat
        </div>

        <div className="chat-messages">
          {messages.length === 0 ? (
            <div style={{ color: "var(--text-secondary)", fontSize: "0.85rem", textAlign: "center", marginTop: "40px" }}>
              No chat messages yet. Ask a question!
            </div>
          ) : (
            messages.map((msg) => {
              const isSelf = msg.sender === username;
              let bubbleClass = isSelf ? "message-bubble outgoing" : "message-bubble incoming";
              if (msg.botType === "dost") bubbleClass = "message-bubble dost-msg";
              if (msg.botType === "sathi") bubbleClass = "message-bubble sathi-msg";

              return (
                <div key={msg.id} className={bubbleClass}>
                  <div className="msg-author">{msg.sender} • {msg.timestamp}</div>
                  <div>{msg.text}</div>
                </div>
              );
            })
          )}
        </div>

        <form onSubmit={handleSendMessage} className="chat-input-area">
          <input
            id="chat-input"
            type="text"
            className="chat-input"
            placeholder="Type a message or question..."
            value={inputMsg}
            onChange={(e) => setInputMsg(e.target.value)}
          />
          <button id="send-chat-btn" type="submit" className="send-btn">
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  );
}

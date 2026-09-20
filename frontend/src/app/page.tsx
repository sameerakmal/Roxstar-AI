"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useParticipants,
  useRoomContext,
  useDataChannel,
  useLocalParticipant,
} from "@livekit/components-react";
import {
  Mic,
  MicOff,
  PhoneOff,
  MessageSquare,
  Send,
  Radio,
  AlertCircle,
  MessageCircle,
} from "lucide-react";

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
  const [serverUrl, setServerUrl] = useState<string>(
    process.env.NEXT_PUBLIC_LIVEKIT_URL || ""
  );

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

      const activeUrl = data.serverUrl || process.env.NEXT_PUBLIC_LIVEKIT_URL;
      if (!activeUrl) {
        throw new Error("LiveKit WebSocket URL is missing. Set NEXT_PUBLIC_LIVEKIT_URL in your .env file.");
      }

      setServerUrl(activeUrl);
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

  /* ──────────────────────────────────────
     Join Screen
     ────────────────────────────────────── */
  if (!joined || !token || !serverUrl) {
    return (
      <main className="join-container">
        <div className="join-card">
          {/* Brand */}
          <div className="join-brand">
            <div className="join-brand-icon">
              <Radio size={18} />
            </div>
            <div className="join-brand-text">
              <span>Rox</span>Star
            </div>
          </div>

          {/* Tagline */}
          <p className="join-tagline">
            Talk naturally.<br />
            Your AI companions are listening.
          </p>
          <p className="join-description">
            Join a live voice room with Dost and Sathi — friendly AI companions who
            understand Hindi and English.
          </p>

          {/* Form */}
          <div className="join-form-section">
            <form onSubmit={handleJoin}>
              <div className="input-group">
                <label className="input-label" htmlFor="username-input">
                  Your name
                </label>
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
                <label className="input-label" htmlFor="room-input">
                  Room name
                </label>
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

              {error && (
                <div className="join-error">
                  <AlertCircle size={16} />
                  <span>{error}</span>
                </div>
              )}

              <button
                id="join-room-btn"
                type="submit"
                className="join-btn"
                disabled={loading}
              >
                {loading ? "Connecting…" : "Join Voice Room"}
              </button>
            </form>
          </div>

          {/* Companion preview */}
          <div className="companion-preview">
            <div className="companion-pill">
              <div className="companion-pill-avatar dost">D</div>
              Dost
            </div>
            <div className="companion-pill">
              <div className="companion-pill-avatar sathi">S</div>
              Sathi
            </div>
          </div>
        </div>
      </main>
    );
  }

  /* ──────────────────────────────────────
     Room
     ────────────────────────────────────── */
  return (
    <LiveKitRoom
      video={false}
      audio={false}
      token={token}
      serverUrl={serverUrl}
      connect={true}
      data-lk-theme="default"
      onError={(err) => {
        console.error("LiveKit Room Error:", err);
        setError(`LiveKit Error: ${err.message || err}`);
        setJoined(false);
        setToken("");
      }}
      onDisconnected={(reason) => {
        console.warn("LiveKit Room Disconnected:", reason);
        if (reason) {
          setError(`Disconnected: ${reason}`);
        } else {
          setError("Disconnected from LiveKit server. Please check your credentials and server URL.");
        }
        setJoined(false);
        setToken("");
      }}
      style={{ height: "100vh" }}
    >
      <RoomContent roomName={roomName} username={username} onLeave={handleLeave} />
      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}


/* ================================================================
   Room Content
   ================================================================ */

function RoomContent({
  roomName,
  username,
  onLeave,
}: {
  roomName: string;
  username: string;
  onLeave: () => void;
}) {
  const room = useRoomContext();
  const participants = useParticipants();
  const { localParticipant } = useLocalParticipant();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMsg, setInputMsg] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const { send } = useDataChannel("chat", (msg) => {
    try {
      const decoded = new TextDecoder().decode(msg.payload);
      const data = JSON.parse(decoded);
      setMessages((prev) => [...prev, data]);
    } catch (e) {
      console.error("Failed to decode chat message:", e);
    }
  });

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const isMuted = !localParticipant.isMicrophoneEnabled;

  const toggleMic = async () => {
    try {
      await localParticipant.setMicrophoneEnabled(isMuted);
    } catch (err) {
      console.error("Failed to toggle microphone:", err);
    }
  };

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMsg.trim()) return;

    const chatMsg: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      sender: username,
      text: inputMsg,
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    const encoder = new TextEncoder();
    if (send) {
      send(encoder.encode(JSON.stringify(chatMsg)), { reliable: true });
    }
    setMessages((prev) => [...prev, chatMsg]);
    setInputMsg("");
  };

  /** Get initial letter for avatar */
  const getInitial = (name: string) => {
    return (name || "?").charAt(0).toUpperCase();
  };

  return (
    <div className="room-layout">
      {/* ── Header ── */}
      <header className="room-header">
        <div className="room-header-brand">
          <Radio size={18} />
          <span className="room-brand-text">
            <span>Rox</span>Star
          </span>
        </div>

        <div className="room-header-center desktop-only">
          <span className="room-name-text">{roomName}</span>
        </div>

        <div className="room-header-right">
          <div className="room-header-live">
            <span className="pulse-dot" />
            Live
          </div>
          <div className="room-header-count">
            {participants.length} participant{participants.length !== 1 ? "s" : ""}
          </div>
        </div>
      </header>

      {/* ── Participant Stage ── */}
      <div className="main-stage">
        <div className="participant-stage">
          <div className="participant-grid">
            {participants.map((p) => {
              const name = p.name || p.identity;
              const isDost = name.toLowerCase().includes("dost");
              const isSathi = name.toLowerCase().includes("sathi");
              const isBot = isDost || isSathi;
              const isSpeaking = p.isSpeaking;

              let avatarClass = "participant-avatar human";
              if (isDost) avatarClass = "participant-avatar dost";
              if (isSathi) avatarClass = "participant-avatar sathi";

              let spotClass = "participant-spot";
              if (isSpeaking) spotClass += " speaking";

              const initial = isDost ? "D" : isSathi ? "S" : getInitial(name);
              const role = isDost
                ? "AI Companion"
                : isSathi
                ? "AI Companion"
                : "Participant";

              return (
                <div key={p.sid || p.identity} className={spotClass}>
                  {/* Avatar */}
                  <div className={avatarClass}>
                    {initial}
                    {!p.isMicrophoneEnabled && (
                      <div className="participant-muted-badge">
                        <MicOff size={11} />
                      </div>
                    )}
                  </div>

                  {/* Name */}
                  <div className="participant-name">{name}</div>

                  {/* Role */}
                  <div className="participant-role">{role}</div>

                  {/* Speaking label */}
                  <div
                    className={`participant-speaking-label${
                      isSathi ? " sathi-label" : ""
                    }`}
                  >
                    {isSpeaking ? "Speaking" : "\u00A0"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Controls ── */}
        <div className="controls-bar">
          <button
            className={`control-btn chat-toggle-btn mobile-only`}
            onClick={() => setChatOpen((v) => !v)}
            aria-label="Toggle chat"
          >
            <MessageSquare size={16} />
          </button>

          <button
            id="mic-toggle-btn"
            className={`mic-btn ${isMuted ? "mic-off" : "mic-on"}`}
            onClick={toggleMic}
            aria-label={isMuted ? "Unmute microphone" : "Mute microphone"}
          >
            {isMuted ? <MicOff size={20} /> : <Mic size={20} />}
          </button>

          <button
            id="leave-room-btn"
            className="control-btn leave-btn"
            onClick={onLeave}
          >
            <PhoneOff size={15} />
            Leave
          </button>
        </div>
      </div>

      {/* ── Chat backdrop (mobile) ── */}
      <div
        className={`chat-overlay-backdrop${chatOpen ? " chat-visible" : ""}`}
        onClick={() => setChatOpen(false)}
      />

      {/* ── Chat Sidebar ── */}
      <aside
        className={`chat-sidebar${chatOpen ? " chat-visible" : ""}`}
      >
        <div className="chat-header">
          <MessageCircle size={16} />
          Chat
        </div>

        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="chat-empty">
              <MessageCircle size={28} />
              <span>No messages yet.</span>
              <span>Say something or ask a question!</span>
            </div>
          ) : (
            messages.map((msg) => {
              const isSelf = msg.sender === username;
              const isDostMsg = msg.botType === "dost";
              const isSathiMsg = msg.botType === "sathi";

              let rowClass = "message-row";
              if (isSelf) rowClass += " outgoing";

              let initialClass = "message-initial human-msg";
              if (isSelf) initialClass = "message-initial self-msg";
              if (isDostMsg) initialClass = "message-initial dost-msg";
              if (isSathiMsg) initialClass = "message-initial sathi-msg";

              let textClass = "message-text";
              if (isDostMsg) textClass += " dost-accent";
              if (isSathiMsg) textClass += " sathi-accent";

              const initial = isDostMsg
                ? "D"
                : isSathiMsg
                ? "S"
                : (msg.sender || "?").charAt(0).toUpperCase();

              return (
                <div key={msg.id} className={rowClass}>
                  <div className={initialClass}>{initial}</div>
                  <div className="message-content">
                    <div className="message-meta">
                      <span className="message-author">{msg.sender}</span>
                      <span className="message-time">{msg.timestamp}</span>
                    </div>
                    <div className={textClass}>{msg.text}</div>
                  </div>
                </div>
              );
            })
          )}
          <div ref={chatEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="chat-input-area">
          <input
            id="chat-input"
            type="text"
            className="chat-input"
            placeholder="Type a message…"
            value={inputMsg}
            onChange={(e) => setInputMsg(e.target.value)}
          />
          <button id="send-chat-btn" type="submit" className="send-btn">
            <Send size={14} />
          </button>
        </form>
      </aside>
    </div>
  );
}

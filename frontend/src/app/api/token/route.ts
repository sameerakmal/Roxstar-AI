import { NextRequest, NextResponse } from "next/server";
import { AccessToken, RoomAgentDispatch, RoomConfiguration } from "livekit-server-sdk";

function sanitizeEnvVar(val?: string): string {
  if (!val) return "";
  return val.trim().replace(/^["']|["']$/g, "");
}

function formatWsUrl(url: string): string {
  let formatted = sanitizeEnvVar(url);
  if (formatted.startsWith("https://")) {
    formatted = formatted.replace("https://", "wss://");
  } else if (formatted.startsWith("http://")) {
    formatted = formatted.replace("http://", "ws://");
  }
  return formatted;
}

export async function GET(req: NextRequest) {
  const room = req.nextUrl.searchParams.get("room") || "default-room";
  const username = req.nextUrl.searchParams.get("username") || `user-${Math.floor(Math.random() * 1000)}`;

  const apiKey = sanitizeEnvVar(process.env.LIVEKIT_API_KEY);
  const apiSecret = sanitizeEnvVar(process.env.LIVEKIT_API_SECRET);
  const rawServerUrl = process.env.LIVEKIT_URL || process.env.NEXT_PUBLIC_LIVEKIT_URL || "";
  const serverUrl = formatWsUrl(rawServerUrl);

  if (!apiKey || !apiSecret || !serverUrl) {
    return NextResponse.json(
      {
        error:
          "LiveKit credentials missing or invalid. Please set LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL (or NEXT_PUBLIC_LIVEKIT_URL) in your .env file.",
      },
      { status: 400 }
    );
  }

  try {
    // Generate participant token with embedded room agent dispatch configuration
    const at = new AccessToken(apiKey, apiSecret, {
      identity: username,
      name: username,
      ttl: "1d",
    });

    at.addGrant({
      roomJoin: true,
      room: room,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });

    // Embed agent dispatch configuration for 'roxstar-ai-dost' and 'roxstar-ai-sathi'
    at.roomConfig = new RoomConfiguration({
      agents: [
        new RoomAgentDispatch({
          agentName: "roxstar-ai-dost",
        }),
        new RoomAgentDispatch({
          agentName: "roxstar-ai-sathi",
        }),
      ],
    });

    console.log(`[Agent Dispatch] Added roxstar-ai-dost and roxstar-ai-sathi to room configuration for room: ${room}`);

    const token = await at.toJwt();
    return NextResponse.json({ token, room, username, serverUrl });
  } catch (err: any) {
    console.error("Token generation error:", err);
    return NextResponse.json({ error: err.message || "Failed to generate token" }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const room = body.room || "default-room";
    const username = body.username || `user-${Math.floor(Math.random() * 1000)}`;

    const apiKey = sanitizeEnvVar(process.env.LIVEKIT_API_KEY);
    const apiSecret = sanitizeEnvVar(process.env.LIVEKIT_API_SECRET);
    const rawServerUrl = process.env.LIVEKIT_URL || process.env.NEXT_PUBLIC_LIVEKIT_URL || "";
    const serverUrl = formatWsUrl(rawServerUrl);

    if (!apiKey || !apiSecret || !serverUrl) {
      return NextResponse.json(
        {
          error:
            "LiveKit credentials missing or invalid. Please set LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL (or NEXT_PUBLIC_LIVEKIT_URL) in your .env file.",
        },
        { status: 400 }
      );
    }

    const at = new AccessToken(apiKey, apiSecret, {
      identity: username,
      name: username,
      ttl: "1d",
    });

    at.addGrant({
      roomJoin: true,
      room: room,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });

    // Embed agent dispatch configuration for 'roxstar-ai-dost' and 'roxstar-ai-sathi'
    at.roomConfig = new RoomConfiguration({
      agents: [
        new RoomAgentDispatch({
          agentName: "roxstar-ai-dost",
        }),
        new RoomAgentDispatch({
          agentName: "roxstar-ai-sathi",
        }),
      ],
    });

    console.log(`[Agent Dispatch] Added roxstar-ai-dost and roxstar-ai-sathi to room configuration for room: ${room}`);

    const token = await at.toJwt();
    return NextResponse.json({ token, room, username, serverUrl });
  } catch (err: any) {
    console.error("Token generation error:", err);
    return NextResponse.json({ error: err.message || "Failed to generate token" }, { status: 500 });
  }
}

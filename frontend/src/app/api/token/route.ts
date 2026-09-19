import { NextRequest, NextResponse } from "next/server";
import { AccessToken } from "livekit-server-sdk";

export async function GET(req: NextRequest) {
  const room = req.nextUrl.searchParams.get("room") || "default-room";
  const username = req.nextUrl.searchParams.get("username") || `user-${Math.floor(Math.random() * 1000)}`;

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;
  const serverUrl = process.env.LIVEKIT_URL || process.env.NEXT_PUBLIC_LIVEKIT_URL;

  if (!apiKey || !apiSecret) {
    return NextResponse.json(
      {
        error:
          "LiveKit credentials missing. Please set LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL in your .env file.",
      },
      { status: 400 }
    );
  }

  try {
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

    const apiKey = process.env.LIVEKIT_API_KEY;
    const apiSecret = process.env.LIVEKIT_API_SECRET;
    const serverUrl = process.env.LIVEKIT_URL || process.env.NEXT_PUBLIC_LIVEKIT_URL;

    if (!apiKey || !apiSecret) {
      return NextResponse.json(
        {
          error:
            "LiveKit credentials missing. Please set LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL in your .env file.",
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

    const token = await at.toJwt();
    return NextResponse.json({ token, room, username, serverUrl });
  } catch (err: any) {
    console.error("Token generation error:", err);
    return NextResponse.json({ error: err.message || "Failed to generate token" }, { status: 500 });
  }
}

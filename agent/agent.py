import asyncio
import logging
import os
import sys
import time
from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import groq, openai, silero
from openai import AsyncOpenAI
from edge_tts_wrapper import EdgeTTS

# Absolute path resolution for dotenv loading
agent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(agent_dir)

load_dotenv(dotenv_path=os.path.join(root_dir, "frontend", ".env"))
load_dotenv(dotenv_path=os.path.join(root_dir, ".env"))
load_dotenv(dotenv_path=os.path.join(agent_dir, ".env"))
load_dotenv()

# Configure structured logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("roxstar-ai-dost")

# Roxstar AI Dost Persona System Prompt
DOST_SYSTEM_PROMPT = """
You are 'Roxstar AI Dost', a friendly, intelligent Indian male AI Voice Assistant participating in a LiveKit room.

COMMUNICATION & LANGUAGE STYLE RULES:
1. Speak in natural, everyday Indian Hinglish (a mixture of simple Hindi and English written in Roman script).
2. Avoid textbook, overly formal, or literal translation-style Hindi. Use modern everyday words.
   - PREFER: "AI ek aisi technology hai jo machine ko samajhne aur decision lene layak banati hai."
   - AVOID: "AI ek aisi takneek hai jo machines ko nirnay lene mein saksham banati hai."
   - PREFER: "Simple words mein bolo to machine ko smart bana deti hai."
   - PREFER: "Ek second ruk jao", "Room join karo", "Mic unmute karo".
3. You must understand spoken & typed English, Hindi, and Roman-script Hinglish questions.
4. Reply mainly in natural Hinglish unless the user explicitly requests an answer in English.
5. Keep your answers concise, engaging, and clear (1-3 sentences max for voice clarity).
6. Naturally mix technical English terms (technology, decision, room, mic, network, server, cloud, example, smart) inside Hindi sentences.
7. Be polite, friendly, and helpful like a great friend (Dost).
"""

async def entrypoint(ctx: JobContext):
    logger.info(f"[Dost] Job received for room: {ctx.room.name}")

    # 1. OpenRouter Credentials Verification (LLM)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key or not openrouter_key.strip():
        logger.error("CRITICAL CONFIGURATION ERROR: OPENROUTER_API_KEY is missing in environment variables.")
        return
    openrouter_key = openrouter_key.strip().strip("'\"")

    # 2. Groq Credentials Verification (STT)
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or not groq_key.strip():
        logger.error("CRITICAL CONFIGURATION ERROR: GROQ_API_KEY is missing in environment variables.")
        return
    groq_key = groq_key.strip().strip("'\"")

    # 3. Connect to LiveKit Room (Must be done FIRST)
    try:
        logger.info(f"[Dost] Connecting to room: {ctx.room.name}...")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[Dost] Successfully connected to room: {ctx.room.name}")

        # Set Participant Identity Name in Room
        try:
            await ctx.room.local_participant.set_name("Roxstar AI Dost")
            logger.info(f"[Dost] Set participant name to 'Roxstar AI Dost' (Identity: {ctx.room.local_participant.identity})")
        except Exception as name_err:
            logger.warning(f"[Dost Warning] Could not set participant name: {name_err}")
    except Exception as conn_err:
        logger.error(f"[Dost Error] Room connection failed: {conn_err}")
        return

    # 4. OpenRouter AsyncOpenAI Client Setup
    try:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            default_headers={
                "HTTP-Referer": "https://roxstar.ai",
                "X-Title": "Roxstar AI Voice Room Assistant",
            },
        )
        logger.info("[Dost] Configured OpenRouter AsyncOpenAI client.")
    except Exception as e:
        logger.error(f"[Dost Error] Failed to initialize OpenRouter client: {e}")
        return

    # 5. Pipeline Components Setup
    try:
        vad_plugin = silero.VAD.load()
        stt_plugin = groq.STT(
            model="whisper-large-v3-turbo",
            api_key=groq_key,
            language="",  # Multilingual auto-detect
        )
        llm_plugin = openai.LLM(
            client=openrouter_client,
            model="openai/gpt-4o-mini",
            api_key=openrouter_key,
            max_tokens=500,
            temperature=0.7,
        )
        tts_plugin = EdgeTTS(voice="hi-IN-MadhurNeural")

        logger.info("[Dost] Pipeline components (Silero VAD, Groq STT, OpenRouter LLM, Edge-TTS) ready.")
    except Exception as e:
        logger.error(f"[Dost Error] Pipeline setup failed: {e}")
        return

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=DOST_SYSTEM_PROMPT,
    )

    agent = VoicePipelineAgent(
        vad=vad_plugin,
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        chat_ctx=initial_ctx,
        allow_interruptions=True,
    )

    @agent.on("user_started_speaking")
    def _on_user_started_speaking():
        logger.info("[Dost VAD] User started speaking...")

    @agent.on("user_stopped_speaking")
    def _on_user_stopped_speaking():
        logger.info("[Dost VAD] User stopped speaking.")

    @agent.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        logger.info(f"[Dost LLM Response]: {msg.content}")

    # 6. Start Voice Pipeline Agent
    logger.info("[Dost] Starting VoicePipelineAgent session on room...")
    agent.start(ctx.room, participant=None)
    logger.info("[Dost] AgentSession started and listening.")

    # 7. Greeting Speech Publication
    try:
        logger.info("[Dost] Publishing greeting speech into room...")
        await agent.say(
            "Namaste! Main hoon Roxstar AI Dost. Aapka swagat hai room mein! Main aapki kya help kar sakta hoon?",
            allow_interruptions=True,
        )
        logger.info("[Dost] Greeting published successfully.")
    except Exception as say_err:
        logger.error(f"[Dost Error] Failed to speak greeting: {say_err}")

if __name__ == "__main__":
    lk_url = os.getenv("LIVEKIT_URL") or os.getenv("NEXT_PUBLIC_LIVEKIT_URL")
    lk_key = os.getenv("LIVEKIT_API_KEY")
    lk_secret = os.getenv("LIVEKIT_API_SECRET")

    opts = WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="roxstar-ai-dost",
    )
    if lk_url:
        opts.ws_url = lk_url.strip().replace("https://", "wss://").replace("http://", "ws://")
    if lk_key:
        opts.api_key = lk_key.strip().strip("'\"")
    if lk_secret:
        opts.api_secret = lk_secret.strip().strip("'\"")

    cli.run_app(opts)

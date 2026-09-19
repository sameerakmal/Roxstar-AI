"""
Roxstar AI Voice Room Assistant - Worker Agent (Phase 3)

DEVELOPMENT & TESTING LIMITATION NOTE:
Phase 3 performs independent STT processing for both AI participants (Dost & Sathi),
which doubles the STT request volume on Groq compared to Phase 2 (2x RPM consumption).
On free-tier Groq keys (20 RPM limit), rapid consecutive speech turns may hit HTTP 429 rate limits.
Graceful STT backoff & non-blocking rate limit recovery is implemented in GracefulGroqSTT to prevent agent crashes.
"""

import asyncio
import logging
import os
import sys
import time
import json
import re
from typing import AsyncIterable
from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    stt,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit import rtc
from livekit.plugins import groq, openai, silero
from openai import AsyncOpenAI
import openai as openai_sdk
from edge_tts_wrapper import EdgeTTS
from router import route_turn

# Absolute path resolution for dotenv loading
agent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(agent_dir)

load_dotenv(dotenv_path=os.path.join(root_dir, "frontend", ".env"))
load_dotenv(dotenv_path=os.path.join(root_dir, ".env"))
load_dotenv(dotenv_path=os.path.join(agent_dir, ".env"))
load_dotenv()

# Determine agent identity
AGENT_NAME = os.getenv("AGENT_NAME", "roxstar-ai-dost").strip().lower()
IS_SATHI = (AGENT_NAME == "roxstar-ai-sathi")
BOT_KEY = "sathi" if IS_SATHI else "dost"
DISPLAY_NAME = "Roxstar AI Sathi" if IS_SATHI else "Roxstar AI Dost"
TTS_VOICE = "hi-IN-SwaraNeural" if IS_SATHI else "hi-IN-MadhurNeural"
STT_MODEL = "whisper-large-v3"
STT_PROMPT = "Roxstar AI Dost, Sathi, Hinglish, Hindi, English conversational assistant room."

# Configure structured logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(AGENT_NAME)

# Graceful Groq STT Subclass for HTTP 429 Rate Limit Handling
class GracefulGroqSTT(groq.STT):
    async def _recognize_impl(self, buffer, *, language, conn_options):
        try:
            return await super()._recognize_impl(buffer, language=language, conn_options=conn_options)
        except (openai_sdk.RateLimitError, openai_sdk.APIStatusError, Exception) as e:
            status_code = getattr(e, "status_code", 429)
            err_str = str(e).lower()
            if status_code == 429 or "rate_limit" in err_str or "429" in err_str:
                retry_delay = 0.0
                response = getattr(e, "response", None)
                if response and hasattr(response, "headers"):
                    h_val = response.headers.get("retry-after") or response.headers.get("Retry-After")
                    if h_val:
                        try:
                            retry_delay = float(h_val)
                        except ValueError:
                            pass
                if retry_delay <= 0.0:
                    match = re.search(r'try\s+again\s+in\s+([\d.]+)\s*s', str(e), re.IGNORECASE)
                    if match:
                        try:
                            retry_delay = float(match.group(1))
                        except ValueError:
                            pass

                logger.warning(
                    f"[{DISPLAY_NAME} Warning] Groq STT HTTP 429 Rate Limit hit (20 RPM limit). "
                    f"Retry-after timing: {retry_delay:.1f}s. Returning empty transcript gracefully."
                )
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language="")],
                )
            raise

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
8. Do not introduce yourself unnecessarily on every response.
"""

# Roxstar AI Sathi Persona System Prompt
SATHI_SYSTEM_PROMPT = """
You are 'Roxstar AI Sathi', a friendly, intelligent, and warm Indian female AI Voice Assistant participating in a LiveKit room.

COMMUNICATION & LANGUAGE STYLE RULES:
1. Speak in natural, everyday Indian Hinglish (a mixture of simple Hindi and English written in Roman script).
2. Avoid textbook, overly formal, or literal translation-style Hindi. Sound warm, conversational, and human.
   - PREFER: "AI ka matlab hai intelligent systems banana jo insano ki tarah soch aur samajh sakein."
   - AVOID: "Kripya dhyan dein ki krutrim buddhimatta ek jatil vishay hai."
   - PREFER: "Aapko main ek easy example se samjhati hoon."
3. You must understand spoken & typed English, Hindi, and Roman-script Hinglish questions.
4. Reply mainly in natural Hinglish unless the user explicitly requests an answer in English.
5. Keep your answers concise, clear, and voice-friendly (approx 1-3 sentences max).
6. Naturally mix technical English terms (computing, cloud, data, algorithm, model, example, machine) inside Hindi sentences.
7. Be encouraging, empathetic, warm, and helpful (Sathi / companion persona), distinct from Dost.
8. Do not introduce yourself unnecessarily on every response.
"""

SYSTEM_PROMPT = SATHI_SYSTEM_PROMPT if IS_SATHI else DOST_SYSTEM_PROMPT

def is_human_participant(p: rtc.RemoteParticipant) -> bool:
    """
    Returns True if participant is a human user, False if it is an AI agent worker.
    LiveKit AI workers have kind = PARTICIPANT_KIND_AGENT and identity starting with "agent-".
    """
    if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
        return False
        
    identity = (p.identity or "").strip().lower()
    name = (p.name or "").strip().lower()
    
    if identity.startswith("agent-") or "roxstar" in identity or "dost" in identity or "sathi" in identity:
        return False
        
    if "roxstar" in name or "dost" in name or "sathi" in name:
        return False
        
    return True

# Speech session turn owner lock to prevent split STT chunks from shifting turn ownership mid-speech
def before_tts_cb(agent_inst: VoicePipelineAgent | None, text: str | AsyncIterable[str], is_sathi: bool | None = None):
    if is_sathi is None:
        is_sathi = IS_SATHI
    bot_name = "Sathi" if is_sathi else "Dost"
    _transcribed_text_val = getattr(agent_inst, "_transcribed_text", "<none>") if agent_inst else "<none>"
    logger.info(f"[TranscriptState] bot={bot_name} before_tts _transcribed_text='{_transcribed_text_val}'")

    # Gate 2: Re-validate latest transcribed text to prevent stale partial chunk speech
    if agent_inst is not None and hasattr(agent_inst, "_transcribed_text"):
        latest_text = (getattr(agent_inst, "_transcribed_text", "") or "").strip()
        if latest_text and latest_text != "<continue>":
            latest_owner = route_turn(latest_text)
            is_still_sel = (latest_owner in ("dost", "both")) if not is_sathi else (latest_owner in ("sathi", "both"))
            if not is_still_sel:
                logger.warning(f"[BotLifecycle] bot={bot_name} event=tts_suppressed latest_transcript='{latest_text}' owner={latest_owner}")
                return ""

    logger.info(f"[BotLifecycle] bot={bot_name} event=llm_finished")
    logger.info(f"[BotLifecycle] bot={bot_name} event=tts_started")
    return text

async def before_llm_cb(agent_inst: VoicePipelineAgent | None, chat_ctx: llm.ChatContext, is_sathi: bool | None = None, room: rtc.Room | None = None):
    if is_sathi is None:
        is_sathi = IS_SATHI
    bot_name = "Sathi" if is_sathi else "Dost"
    task_id = f"{bot_name.lower()}-task-{int(time.time()*1000) % 100000}"
    user_input = ""
    if chat_ctx.messages:
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user" and msg.content:
                if isinstance(msg.content, str):
                    user_input = msg.content
                elif isinstance(msg.content, list):
                    user_input = " ".join([str(item) for item in msg.content])
                else:
                    user_input = str(msg.content)
                break

    user_input_clean = user_input.strip()
    _transcribed_text_val = getattr(agent_inst, "_transcribed_text", "<none>") if agent_inst else "<none>"

    logger.info(f"[TurnDebug] bot={bot_name} task={task_id} transcript='{user_input}'")
    logger.info(f"[STTDebug] bot={bot_name} transcript='{user_input}' final=true")
    logger.info(f"[TranscriptState] bot={bot_name} before_llm _transcribed_text='{_transcribed_text_val}'")
    logger.info(f"[BotLifecycle] bot={bot_name} event=turn_received transcript='{user_input}'")
    logger.info(f"[RoutingDebug] bot={bot_name} input='{user_input}'")

    # Gate 1: Suppress empty or continue prompts
    if not user_input_clean or user_input_clean == "<continue>":
        logger.info(f"[TurnDebug] bot={bot_name} task={task_id} result=suppressed_empty")
        logger.info(f"[BotLifecycle] bot={bot_name} event=route_decision selected=false route_result=empty_input")
        logger.info(f"[BotLifecycle] bot={bot_name} event=before_llm decision=suppressed")
        logger.info(f"[{bot_name}] [Turn Lifecycle] callback return value=False (empty/continue prompt)")
        return False

    # Determine turn routing deterministically from user input
    selected_agent = route_turn(user_input_clean)
    is_sel = (selected_agent in ("dost", "both")) if not is_sathi else (selected_agent in ("sathi", "both"))

    logger.info(f"[RoutingDebug] bot={bot_name} selected_agent={selected_agent}")
    logger.info(f"[BotLifecycle] bot={bot_name} event=route_decision selected={str(is_sel).lower()} route_result={selected_agent}")

    if not is_sel:
        logger.info(f"[TurnDebug] bot={bot_name} task={task_id} result=suppressed_unselected")
        logger.info(f"[BotLifecycle] bot={bot_name} event=before_llm decision=suppressed")
        logger.info(f"[{bot_name}] [Turn Lifecycle] callback return value=False (suppressing synthesis)")
        if agent_inst is not None:
            agent_inst.interrupt(interrupt_all=True)
        return False

    logger.info(f"[TurnDebug] bot={bot_name} task={task_id} result=allowed")
    logger.info(f"[BotLifecycle] bot={bot_name} event=before_llm decision=allowed")

    if is_sathi and selected_agent == "both":
        logger.info(f"[{bot_name}] [Turn Lifecycle] both requested. Sathi waiting for Dost to complete turn...")
        await asyncio.sleep(3.5)

    logger.info(f"[TurnDebug] bot={bot_name} task={task_id} event=llm_started")
    logger.info(f"[BotLifecycle] bot={bot_name} event=llm_started")
    return None

async def entrypoint(ctx: JobContext):
    logger.info(f"[{DISPLAY_NAME}] Job received for room: {ctx.room.name}")

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

    # 3. Connect to LiveKit Room
    try:
        logger.info(f"[{DISPLAY_NAME}] Connecting to room: {ctx.room.name}...")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[{DISPLAY_NAME}] Successfully connected to room: {ctx.room.name}")

        try:
            await ctx.room.local_participant.set_name(DISPLAY_NAME)
            logger.info(f"[{DISPLAY_NAME}] Set participant name to '{DISPLAY_NAME}' (Identity: {ctx.room.local_participant.identity})")
        except Exception as name_err:
            logger.warning(f"[{DISPLAY_NAME} Warning] Could not set participant name: {name_err}")
    except Exception as conn_err:
        logger.error(f"[{DISPLAY_NAME} Error] Room connection failed: {conn_err}")
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
        logger.info(f"[{DISPLAY_NAME}] Configured OpenRouter AsyncOpenAI client.")
    except Exception as e:
        logger.error(f"[{DISPLAY_NAME} Error] Failed to initialize OpenRouter client: {e}")
        return

    # 5. Pipeline Components Setup
    try:
        vad_plugin = silero.VAD.load()
        stt_plugin = GracefulGroqSTT(
            model=STT_MODEL,
            api_key=groq_key,
            prompt=STT_PROMPT,
        )
        llm_plugin = openai.LLM(
            client=openrouter_client,
            model="openai/gpt-4o-mini",
            api_key=openrouter_key,
            max_tokens=500,
            temperature=0.7,
        )
        tts_plugin = EdgeTTS(voice=TTS_VOICE)

        logger.info(f"[{DISPLAY_NAME}] Pipeline components (Silero VAD, Graceful Groq STT '{STT_MODEL}', OpenRouter LLM, Edge-TTS '{TTS_VOICE}') ready.")
    except Exception as e:
        logger.error(f"[{DISPLAY_NAME} Error] Pipeline setup failed: {e}")
        return

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=SYSTEM_PROMPT,
    )

    # Human participant tracking
    human_participant: rtc.RemoteParticipant | None = None
    for p in ctx.room.remote_participants.values():
        if is_human_participant(p):
            human_participant = p
            break

    # 6. Deterministic Voice Turn Routing Callbacks
    async def _before_llm(agent_inst: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        return await before_llm_cb(agent_inst, chat_ctx, is_sathi=IS_SATHI, room=ctx.room)

    def _before_tts(agent_inst: VoicePipelineAgent, text: str | AsyncIterable[str]):
        return before_tts_cb(agent_inst, text, is_sathi=IS_SATHI)

    bot_label = "Sathi" if IS_SATHI else "Dost"
    logger.info(f"[RoutingDebug] bot={bot_label} before_llm_cb={_before_llm}")
    logger.info(f"[RoutingDebug] bot={bot_label} before_tts_cb={_before_tts}")

    agent = VoicePipelineAgent(
        vad=vad_plugin,
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        chat_ctx=initial_ctx,
        allow_interruptions=True,
        before_llm_cb=_before_llm,
        before_tts_cb=_before_tts,
    )

    @agent.on("user_started_speaking")
    def _on_user_started_speaking():
        logger.info(f"[{DISPLAY_NAME} VAD] User started speaking (interrupting any active agent speech)...")
        agent.interrupt(interrupt_all=True)

    @agent.on("user_stopped_speaking")
    def _on_user_stopped_speaking():
        logger.info(f"[{DISPLAY_NAME} VAD] User stopped speaking.")

    @agent.on("agent_started_speaking")
    def _on_agent_started_speaking():
        bot_name = "Sathi" if IS_SATHI else "Dost"
        logger.info(f"[BotLifecycle] bot={bot_name} event=speech_started")

    @agent.on("agent_stopped_speaking")
    def _on_agent_stopped_speaking():
        bot_name = "Sathi" if IS_SATHI else "Dost"
        logger.info(f"[BotLifecycle] bot={bot_name} event=speech_finished")

    @agent.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        bot_name = "Sathi" if IS_SATHI else "Dost"
        logger.info(f"[{AGENT_NAME}] [Turn Lifecycle] LLM generation completed: '{msg.content}'")

    # 7. Text Chat Handling over DataChannel
    async def handle_text_chat_response(text: str, wait_for_dost: bool = False):
        if wait_for_dost:
            await asyncio.sleep(2.5)
        try:
            c_ctx = agent.chat_ctx.copy()
            c_ctx.messages.append(llm.ChatMessage.create(text=text, role="user"))
            stream = agent.llm.chat(chat_ctx=c_ctx)
            reply_text = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    reply_text += chunk.choices[0].delta.content

            if reply_text:
                chat_msg = {
                    "id": str(int(time.time() * 1000)),
                    "sender": DISPLAY_NAME,
                    "text": reply_text,
                    "timestamp": time.strftime("%I:%M %p"),
                    "isBot": True,
                    "botType": BOT_KEY,
                }
                payload = json.dumps(chat_msg).encode("utf-8")
                await ctx.room.local_participant.publish_data(payload, topic="chat", reliable=True)
                await agent.say(reply_text, allow_interruptions=True)
        except Exception as err:
            logger.error(f"[{DISPLAY_NAME}] Text chat response generation error: {err}")

    @ctx.room.on("data_received")
    def on_data_received(dp: rtc.DataPacket):
        try:
            topic = getattr(dp, "topic", "")
            if topic and topic != "chat":
                return
            raw_str = dp.data.decode("utf-8")
            data = json.loads(raw_str)
            sender = data.get("sender", "")
            bot_type = data.get("botType", "")

            # Ignore messages originating from Roxstar AI Dost or Roxstar AI Sathi
            if bot_type in ("dost", "sathi") or "roxstar" in sender.lower() or "dost" in sender.lower() or "sathi" in sender.lower():
                return

            text = data.get("text", "")
            if not text or not text.strip():
                return

            selected_agent = route_turn(text)
            logger.info(f"[{DISPLAY_NAME} DataChannel] Message: '{text}' | Selected agent: {selected_agent}")

            if not IS_SATHI and selected_agent in ("dost", "both"):
                asyncio.create_task(handle_text_chat_response(text))
            elif IS_SATHI and selected_agent in ("sathi", "both"):
                asyncio.create_task(handle_text_chat_response(text, wait_for_dost=(selected_agent == "both")))
        except Exception as e:
            logger.error(f"[{DISPLAY_NAME} DataChannel Error]: {e}")

    # 8. Start VoicePipelineAgent linked exclusively to human participant
    if human_participant is not None:
        logger.info(f"[{DISPLAY_NAME}] LINKING AUDIO TRACK: target_identity='{human_participant.identity}', name='{human_participant.name}', kind='{human_participant.kind}' (Human Participant)")
        agent.start(ctx.room, participant=human_participant)
    else:
        logger.info(f"[{DISPLAY_NAME}] No human in room yet. Standing by for human participant to connect...")
        connected_event = asyncio.Event()

        def _on_participant_connected(p: rtc.RemoteParticipant):
            nonlocal human_participant
            if is_human_participant(p) and not connected_event.is_set():
                logger.info(f"[{DISPLAY_NAME}] LINKING AUDIO TRACK: human participant connected target_identity='{p.identity}', name='{p.name}', kind='{p.kind}'")
                human_participant = p
                connected_event.set()

        ctx.room.on("participant_connected", _on_participant_connected)
        await connected_event.wait()
        agent.start(ctx.room, participant=human_participant)

    logger.info(f"[{DISPLAY_NAME}] AgentSession started and listening exclusively to human participant '{human_participant.identity}'.")

    # 9. Greeting Speech Publication (Dost Only)
    if not IS_SATHI:
        try:
            greeting_text = "Namaste! Main hoon Roxstar AI Dost. Aapka swagat hai room mein! Main aapki kya help kar sakta hoon?"
            logger.info(f"[{DISPLAY_NAME}] Publishing greeting speech into room...")
            await agent.say(
                greeting_text,
                allow_interruptions=True,
            )
            logger.info(f"[{DISPLAY_NAME}] Greeting published successfully.")
        except Exception as say_err:
            logger.error(f"[{DISPLAY_NAME} Error] Failed to speak greeting: {say_err}")
    else:
        logger.info(f"[{DISPLAY_NAME}] Sathi connected. Automatic startup greeting disabled (standing by for user turns).")

def main():
    agent_name = os.getenv("AGENT_NAME", "roxstar-ai-dost").strip().lower()
    lk_url = os.getenv("LIVEKIT_URL") or os.getenv("NEXT_PUBLIC_LIVEKIT_URL")
    lk_key = os.getenv("LIVEKIT_API_KEY")
    lk_secret = os.getenv("LIVEKIT_API_SECRET")

    opts = WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=agent_name,
    )
    if lk_url:
        opts.ws_url = lk_url.strip().replace("https://", "wss://").replace("http://", "ws://")
    if lk_key:
        opts.api_key = lk_key.strip().strip("'\"")
    if lk_secret:
        opts.api_secret = lk_secret.strip().strip("'\"")

    cli.run_app(opts)

if __name__ == "__main__":
    main()

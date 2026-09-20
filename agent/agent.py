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
from typing import AsyncIterable, Optional, List
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
from router import (
    route_turn,
    parse_ordered_plan,
    MultiBotPlan,
    MultiBotTarget,
)
from room_context import (
    RoomHistoryManager,
    RoomTurn,
    SPEAKER_HUMAN,
    SPEAKER_DOST,
    SPEAKER_SATHI,
    strip_speaker_labels,
)

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

def sanitize_log_message(msg: str) -> str:
    """Masks any API keys, tokens, or authorization credentials in error strings."""
    if not msg:
        return ""
    # Mask OpenRouter / OpenAI keys (sk-...)
    msg = re.sub(r'sk-[a-zA-Z0-9_\-]{8,}', '[REDACTED_API_KEY]', msg)
    # Mask Groq keys
    msg = re.sub(r'g[s]k_[a-zA-Z0-9_\-]{8,}', '[REDACTED_API_KEY]', msg)
    # Mask Authorization headers / bearer tokens
    msg = re.sub(r'(?:bearer|token|key|secret)[=:\s]+["\']?[a-zA-Z0-9_\-]{14,}["\']?', '[REDACTED_CREDENTIAL]', msg, flags=re.IGNORECASE)
    return msg

# Phase 7: Multi-Bot Ordered Orchestration State Manager
class OrchestrationManager:
    """Manages multi-bot turn sequencing, pending plans, and completion coordination."""
    def __init__(self):
        self.current_plan: Optional[MultiBotPlan] = None
        self.my_order: Optional[int] = None
        self.pending_order2_plan: Optional[MultiBotPlan] = None
        self.order2_task: Optional[asyncio.Task] = None
        self.is_interrupted: bool = False
        self.is_text_chat: bool = False

    def reset(self, plan: Optional[MultiBotPlan] = None, my_order: Optional[int] = None, is_text_chat: bool = False):
        self.current_plan = plan
        self.my_order = my_order
        self.is_interrupted = False
        self.is_text_chat = is_text_chat
        if self.order2_task and not self.order2_task.done():
            self.order2_task.cancel()
        self.order2_task = None
        self.pending_order2_plan = None

    def mark_order1_finished(self):
        self.current_plan = None
        self.my_order = None

    def cancel_current_plan(self):
        self.current_plan = None
        self.my_order = None
        self.pending_order2_plan = None
        self.is_interrupted = False
        self.is_text_chat = False
        if self.order2_task and not self.order2_task.done():
            self.order2_task.cancel()
        self.order2_task = None

global_orchestrator = OrchestrationManager()

# Graceful Groq STT Subclass for Rate Limit & Network Error Handling
class GracefulGroqSTT(groq.STT):
    async def _recognize_impl(self, buffer, *, language, conn_options):
        try:
            return await super()._recognize_impl(buffer, language=language, conn_options=conn_options)
        except (openai_sdk.RateLimitError, openai_sdk.APIStatusError, openai_sdk.APIConnectionError, openai_sdk.APITimeoutError, Exception) as e:
            status_code = getattr(e, "status_code", 429 if isinstance(e, openai_sdk.RateLimitError) else "error")
            err_str = str(e).lower()
            if status_code == 429 or "rate_limit" in err_str or "429" in err_str or isinstance(e, openai_sdk.RateLimitError):
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
                    f"[STT] bot={DISPLAY_NAME} event=rate_limited retry_after={retry_delay:.1f}s"
                )
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language="")],
                )
            else:
                clean_err = sanitize_log_message(str(e))
                logger.error(
                    f"[STT] bot={DISPLAY_NAME} event=provider_error status={status_code} error='{clean_err}'"
                )
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language="")],
                )

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

ROOM CONTEXT & MULTI-AGENT AWARENESS:
- You are participating in a LiveKit room with human participants and another AI assistant ('Roxstar AI Sathi').
- The conversation history contains previous turns from human participants, Dost, and Sathi labeled as [Human], [Roxstar AI Dost], and [Roxstar AI Sathi].
- Use this shared history to understand references (e.g. "iska", "that", "previous topic") and facts mentioned by any speaker.
- Do not speak or repeat internal speaker labels such as [Human], [Roxstar AI Dost], or [Roxstar AI Sathi]. These labels are metadata used only for conversation context.
- Respond naturally as your own persona without prefixing your output with your name or any speaker tag.
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

ROOM CONTEXT & MULTI-AGENT AWARENESS:
- You are participating in a LiveKit room with human participants and another AI assistant ('Roxstar AI Dost').
- The conversation history contains previous turns from human participants, Dost, and Sathi labeled as [Human], [Roxstar AI Dost], and [Roxstar AI Sathi].
- Use this shared history to understand references (e.g. "iska", "that", "previous topic") and facts mentioned by any speaker.
- Do not speak or repeat internal speaker labels such as [Human], [Roxstar AI Dost], or [Roxstar AI Sathi]. These labels are metadata used only for conversation context.
- Respond naturally as your own persona without prefixing your output with your name or any speaker tag.
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

async def _resilient_tts_stream(
    stream: AsyncIterable[str],
    bot_name: str,
    agent_inst: VoicePipelineAgent | None = None,
    turn_id: str | None = None,
) -> AsyncIterable[str]:
    """
    Consumes streaming LLM chunks safely:
    - Strips internal speaker metadata labels.
    - Yields clean chunks to TTS.
    - If LLM fails or returns empty, provides a natural persona fallback response.
    - If turn is interrupted/cancelled, suppresses late output.
    """
    first_buffer = ""
    stripped = False
    yielded_any = False

    try:
        async for chunk in stream:
            # Stale turn protection: check if speech was interrupted
            if agent_inst is not None and getattr(agent_inst, "_playing_speech", None):
                if getattr(agent_inst._playing_speech, "interrupted", False):
                    logger.info(f"[TURN] turn_id={turn_id} event=stale_output_suppressed")
                    return

            if not stripped:
                first_buffer += chunk
                if ":" in first_buffer or len(first_buffer) > 40:
                    cleaned = strip_speaker_labels(first_buffer)
                    stripped = True
                    if cleaned:
                        yielded_any = True
                        yield cleaned
            else:
                if chunk:
                    yielded_any = True
                    yield chunk

        if not stripped and first_buffer:
            cleaned = strip_speaker_labels(first_buffer)
            if cleaned:
                yielded_any = True
                yield cleaned

        # Handle empty/whitespace-only LLM output
        if not yielded_any:
            is_interrupted = False
            if agent_inst is not None and getattr(agent_inst, "_playing_speech", None):
                is_interrupted = getattr(agent_inst._playing_speech, "interrupted", False)

            if is_interrupted:
                logger.info(f"[TURN] turn_id={turn_id} event=stale_output_suppressed")
                return

            fallback = (
                "Thoda technical issue aa gaya. Ek baar phir poochho."
                if bot_name.lower() == "dost"
                else "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."
            )
            logger.warning(f"[LLM] bot={bot_name} event=empty_response status=fallback")
            yield fallback

    except Exception as e:
        is_interrupted = False
        if agent_inst is not None and getattr(agent_inst, "_playing_speech", None):
            is_interrupted = getattr(agent_inst._playing_speech, "interrupted", False)

        status_code = getattr(e, "status_code", "error")
        clean_err = sanitize_log_message(str(e))
        logger.error(
            f"[LLM] bot={bot_name} event=provider_error status={status_code} error='{clean_err}' interrupted={is_interrupted}"
        )

        if is_interrupted:
            logger.info(f"[TURN] turn_id={turn_id} event=stale_output_suppressed")
            return

        if not yielded_any:
            fallback = (
                "Thoda technical issue aa gaya. Ek baar phir poochho."
                if bot_name.lower() == "dost"
                else "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."
            )
            logger.info(f"[LLM] bot={bot_name} event=yielding_fallback text='{fallback}'")
            yield fallback

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

    if isinstance(text, str):
        cleaned = strip_speaker_labels(text)
        if not cleaned:
            fallback = (
                "Thoda technical issue aa gaya. Ek baar phir poochho."
                if not is_sathi
                else "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."
            )
            logger.warning(f"[LLM] bot={bot_name} event=empty_response status=fallback")
            return fallback
        return cleaned
    elif isinstance(text, AsyncIterable):
        return _resilient_tts_stream(text, bot_name=bot_name, agent_inst=agent_inst)
    return text

# Phase 7: Order 2 Delayed Execution
async def _execute_order2_response(
    agent_inst: VoicePipelineAgent,
    plan: MultiBotPlan,
    room_history: RoomHistoryManager,
    room: rtc.Room,
    first_bot_content: str = "",
    orchestrator: OrchestrationManager | None = None,
    is_sathi: bool | None = None,
    is_text_chat: bool = False,
):
    orch = orchestrator if orchestrator is not None else global_orchestrator
    if is_sathi is None:
        is_sathi = IS_SATHI

    bot_key = "sathi" if is_sathi else "dost"
    bot_display_name = "Roxstar AI Sathi" if is_sathi else "Roxstar AI Dost"
    other_name = "Roxstar AI Dost" if is_sathi else "Roxstar AI Sathi"
    speaker = SPEAKER_SATHI if is_sathi else SPEAKER_DOST
    other_speaker = SPEAKER_DOST if is_sathi else SPEAKER_SATHI

    if orch.is_interrupted:
        logger.info(f"[{bot_display_name}] [Orchestration] Suppressing order 2 execution because plan was interrupted.")
        return

    my_target = next((t for t in plan.targets if t.bot == bot_key), None)
    if not my_target:
        return

    logger.info(
        f"[{bot_display_name}] [Orchestration] Executing order 2 response for plan {plan.turn_id}. "
        f"Instruction: '{my_target.instruction}' | Context length from first bot: {len(first_bot_content)}"
    )

    try:
        # Ensure first bot's content is in room history before building LLM context
        if first_bot_content and room_history:
            recent_turns = getattr(room_history, "turns", [])[-5:]
            has_first_turn = any(
                getattr(t, "speaker", "") == other_speaker and getattr(t, "text", "").strip() == first_bot_content.strip()
                for t in recent_turns
            )
            if not has_first_turn:
                other_turn = room_history.create_bot_turn(other_speaker, first_bot_content)
                room_history.add_turn(other_turn)

        sys_prompt = SATHI_SYSTEM_PROMPT if is_sathi else DOST_SYSTEM_PROMPT

        if first_bot_content:
            enrichment = (
                f"\n\n[Instruction for this turn]: The user asked {other_name} to respond before you.\n"
                f"{other_name} answered: \"{first_bot_content}\"\n"
                f"Now it is your turn. Your specific instruction from the user is: \"{my_target.instruction}\".\n"
                f"Directly build upon {other_name}'s explanation. Provide the requested example or follow-up clearly and concisely in natural Hinglish. Do NOT repeat what {other_name} already said."
            )
        else:
            enrichment = (
                f"\n\n[Instruction for this turn]: The user asked {other_name} to respond before you. "
                f"They have already answered. Now it is your turn. Focus on: {my_target.instruction}. "
                f"Do not repeat what was already said. Keep your response relevant and concise."
            )
        enriched_prompt = sys_prompt + enrichment

        c_ctx = room_history.build_chat_context(enriched_prompt, is_sathi=is_sathi)

        reply_text = ""
        try:
            stream = agent_inst.llm.chat(chat_ctx=c_ctx)
            async for chunk in stream:
                if orch.is_interrupted:
                    logger.info(f"[{bot_display_name}] [Orchestration] Interrupted during order 2 LLM generation.")
                    return
                if chunk.choices and chunk.choices[0].delta.content:
                    reply_text += chunk.choices[0].delta.content
        except Exception as llm_err:
            clean_err = sanitize_log_message(str(llm_err))
            logger.error(f"[LLM] bot={bot_display_name} event=provider_error error='{clean_err}'")
            reply_text = (
                "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."
                if is_sathi
                else "Thoda technical issue aa gaya. Ek baar phir poochho."
            )

        if orch.is_interrupted:
            return

        clean_reply = strip_speaker_labels(reply_text)
        if not clean_reply:
            clean_reply = (
                "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."
                if is_sathi
                else "Thoda technical issue aa gaya. Ek baar phir poochho."
            )

        bot_turn = room_history.create_bot_turn(speaker, clean_reply)
        b_added = room_history.add_turn(bot_turn)
        if b_added and room and getattr(room, "local_participant", None):
            try:
                payload = room_history.serialize_turn_event(bot_turn)
                await room.local_participant.publish_data(payload, topic="room_context", reliable=True)
            except Exception as pub_err:
                clean_err = sanitize_log_message(str(pub_err))
                logger.warning(f"[ROOM] event=context_sync_error error='{clean_err}'")

        if is_text_chat:
            chat_msg = {
                "id": str(int(time.time() * 1000)),
                "sender": bot_display_name,
                "text": clean_reply,
                "timestamp": time.strftime("%I:%M %p"),
                "isBot": True,
                "botType": bot_key,
            }
            try:
                payload = json.dumps(chat_msg).encode("utf-8")
                if room and getattr(room, "local_participant", None):
                    await room.local_participant.publish_data(payload, topic="chat", reliable=True)
            except Exception as pub_err:
                clean_err = sanitize_log_message(str(pub_err))
                logger.warning(f"[ROOM] event=chat_publish_error error='{clean_err}'")

        if not orch.is_interrupted:
            logger.info(f"[{bot_display_name}] [Orchestration] Speaking order 2 response: '{clean_reply[:50]}...'")
            await agent_inst.say(clean_reply, allow_interruptions=True)
    except asyncio.CancelledError:
        logger.info(f"[{bot_display_name}] [Orchestration] Order 2 response task cancelled.")
    except Exception as err:
        clean_err = sanitize_log_message(str(err))
        logger.error(f"[{bot_display_name}] [Orchestration] Order 2 execution error: {clean_err}")

async def before_llm_cb(
    agent_inst: VoicePipelineAgent | None,
    chat_ctx: llm.ChatContext,
    is_sathi: bool | None = None,
    room: rtc.Room | None = None,
    room_history: RoomHistoryManager | None = None,
    orchestrator: OrchestrationManager | None = None,
):
    if is_sathi is None:
        is_sathi = IS_SATHI
    bot_name = "Sathi" if is_sathi else "Dost"
    bot_key = "sathi" if is_sathi else "dost"
    orch = orchestrator if orchestrator is not None else global_orchestrator
    turn_start_time = time.time()
    task_id = f"{bot_name.lower()}-task-{int(turn_start_time * 1000) % 100000}"
    if agent_inst is not None:
        setattr(agent_inst, "_current_turn_start_time", turn_start_time)
        setattr(agent_inst, "_current_task_id", task_id)

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
    logger.info(f"[Latency] bot={bot_name} task={task_id} event=turn_started")

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

    # Phase 7: Multi-bot ordered orchestration handling
    plan: Optional[MultiBotPlan] = None
    if selected_agent == "both":
        plan = parse_ordered_plan(user_input_clean)
        if plan:
            my_target = next((t for t in plan.targets if t.bot == bot_key), None)
            other_target = next((t for t in plan.targets if t.bot != bot_key), None)

            # Announce plan on DataChannel
            if room and hasattr(room, "local_participant") and room.local_participant:
                try:
                    plan_announcement = {
                        "type": "bot_plan_announced",
                        "turn_id": plan.turn_id,
                        "targets": [
                            {"bot": t.bot, "order": t.order, "instruction": t.instruction}
                            for t in plan.targets
                        ],
                        "timestamp": plan.timestamp,
                        "text": plan.original_text,
                    }
                    asyncio.create_task(
                        room.local_participant.publish_data(
                            json.dumps(plan_announcement).encode("utf-8"),
                            topic="bot_orchestration",
                            reliable=True,
                        )
                    )
                except Exception as ann_err:
                    clean_err = sanitize_log_message(str(ann_err))
                    logger.warning(f"[Orchestration] Plan announcement publish error: {clean_err}")

            if my_target and my_target.order == 2:
                # Order 2: Suppress immediate LLM turn, register pending plan and await order 1 completion
                logger.info(f"[{bot_name}] [Orchestration] Registered as order 2 for plan {plan.turn_id}. Suppressing immediate LLM turn, awaiting order 1 completion.")
                orch.reset(plan, my_order=2, is_text_chat=False)
                orch.pending_order2_plan = plan
                return False

            # Order 1: Proceeds immediately
            if my_target and my_target.order == 1:
                logger.info(f"[{bot_name}] [Orchestration] Registered as order 1 for plan {plan.turn_id}. Proceeding immediately.")
                orch.reset(plan, my_order=1, is_text_chat=False)
    else:
        # Single-bot turn: clear any previous multi-bot plan
        orch.cancel_current_plan()

    logger.info(f"[TurnDebug] bot={bot_name} task={task_id} result=allowed")
    logger.info(f"[BotLifecycle] bot={bot_name} event=before_llm decision=allowed")

    # Phase 4: Shared Room History Context Injection for Selected Agent
    if room_history is not None:
        human_turn = room_history.create_human_turn(user_input_clean)
        added = room_history.add_turn(human_turn)
        if added and room and hasattr(room, "local_participant") and room.local_participant:
            async def _safe_publish_human(payload_bytes: bytes):
                try:
                    await room.local_participant.publish_data(payload_bytes, topic="room_context", reliable=True)
                except Exception as pub_err:
                    clean_err = sanitize_log_message(str(pub_err))
                    logger.warning(f"[ROOM] event=context_sync_error error='{clean_err}'")

            try:
                payload = room_history.serialize_turn_event(human_turn)
                asyncio.create_task(_safe_publish_human(payload))
            except Exception as pub_err:
                clean_err = sanitize_log_message(str(pub_err))
                logger.warning(f"[ROOM] event=context_sync_error error='{clean_err}'")

        sys_prompt = SATHI_SYSTEM_PROMPT if is_sathi else DOST_SYSTEM_PROMPT
        new_ctx = room_history.build_chat_context(sys_prompt, is_sathi=is_sathi)
        chat_ctx.messages.clear()
        chat_ctx.messages.extend(new_ctx.messages)

    # Phase 7: Context enrichment for order 1 in multi-bot plan
    if selected_agent == "both" and plan:
        my_target = next((t for t in plan.targets if t.bot == bot_key), None)
        other_target = next((t for t in plan.targets if t.bot != bot_key), None)
        if my_target and my_target.order == 1:
            other_name = "Roxstar AI Sathi" if not is_sathi else "Roxstar AI Dost"
            enrichment = (
                f"\n\n[Instruction for this turn]: The user asked you and {other_name} to respond in sequence. "
                f"You are answering first. Focus on: {my_target.instruction}. "
                f"Be concise and direct so {other_name} can add their part."
            )
            chat_ctx.append(role="system", text=enrichment)

    logger.info(f"[TurnDebug] bot={bot_name} task={task_id} event=llm_started")
    logger.info(f"[BotLifecycle] bot={bot_name} event=llm_started")
    return None

async def entrypoint(ctx: JobContext):
    logger.info(f"[{DISPLAY_NAME}] Job received for room: {ctx.room.name}")
    
    # Phase 7: Clean reset orchestration manager for each new room session
    global_orchestrator.cancel_current_plan()

    # Initialize Phase 4 RoomHistoryManager for this room session
    room_history = RoomHistoryManager(max_turns=20)

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
        return await before_llm_cb(
            agent_inst,
            chat_ctx,
            is_sathi=IS_SATHI,
            room=ctx.room,
            room_history=room_history,
            orchestrator=global_orchestrator,
        )

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
        # If order 2 execution task is actively running, cancel it
        if global_orchestrator.order2_task and not global_orchestrator.order2_task.done():
            logger.info(f"[{DISPLAY_NAME}] [Orchestration] Interrupting active order 2 execution task.")
            global_orchestrator.order2_task.cancel()
            global_orchestrator.is_interrupted = True

        # If this bot was actively speaking as order 1 in a multi-bot plan, broadcast cancellation
        if global_orchestrator.current_plan and global_orchestrator.my_order == 1:
            logger.info(f"[{DISPLAY_NAME}] [Orchestration] Order 1 interrupted by user speech. Broadcasting cancellation.")
            global_orchestrator.is_interrupted = True
            plan_turn_id = global_orchestrator.current_plan.turn_id
            cancel_payload = {
                "type": "bot_plan_cancelled",
                "turn_id": plan_turn_id,
                "bot": BOT_KEY,
                "reason": "user_interruption",
            }
            global_orchestrator.cancel_current_plan()
            if ctx.room and getattr(ctx.room, "local_participant", None):
                async def _safe_publish_cancel(payload_bytes: bytes):
                    try:
                        await ctx.room.local_participant.publish_data(payload_bytes, topic="bot_orchestration", reliable=True)
                    except Exception:
                        pass
                try:
                    payload = json.dumps(cancel_payload).encode("utf-8")
                    asyncio.create_task(_safe_publish_cancel(payload))
                except Exception:
                    pass

        agent.interrupt(interrupt_all=True)

    @agent.on("user_stopped_speaking")
    def _on_user_stopped_speaking():
        logger.info(f"[{DISPLAY_NAME} VAD] User stopped speaking.")
        global_orchestrator.is_interrupted = False

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
        speaker = SPEAKER_SATHI if IS_SATHI else SPEAKER_DOST
        clean_content = strip_speaker_labels(msg.content or "")
        logger.info(f"[{AGENT_NAME}] [Turn Lifecycle] LLM generation completed: '{clean_content}'")

        turn_start = getattr(agent, "_current_turn_start_time", None)
        if turn_start:
            total_duration_ms = int((time.time() - turn_start) * 1000)
            logger.info(f"[Latency] bot={bot_name} event=turn_completed total_ms={total_duration_ms}")

        if clean_content:
            bot_turn = room_history.create_bot_turn(speaker, clean_content)
            added = room_history.add_turn(bot_turn)
            if added and ctx.room and getattr(ctx.room, "local_participant", None):
                async def _safe_publish_bot(payload_bytes: bytes):
                    try:
                        await ctx.room.local_participant.publish_data(payload_bytes, topic="room_context", reliable=True)
                    except Exception as pub_err:
                        clean_err = sanitize_log_message(str(pub_err))
                        logger.warning(f"[ROOM] event=context_sync_error error='{clean_err}'")

                try:
                    payload = room_history.serialize_turn_event(bot_turn)
                    asyncio.create_task(_safe_publish_bot(payload))
                except Exception as pub_err:
                    clean_err = sanitize_log_message(str(pub_err))
                    logger.warning(f"[ROOM] event=context_sync_error error='{clean_err}'")

        # Phase 7: If this bot was order 1 in an active multi-bot plan and not interrupted, publish completion!
        if (
            global_orchestrator.current_plan
            and global_orchestrator.my_order == 1
            and not global_orchestrator.is_interrupted
        ):
            plan_turn_id = global_orchestrator.current_plan.turn_id
            logger.info(
                f"[Orchestration] bot={DISPLAY_NAME} event=publish_completion "
                f"turn_id={plan_turn_id} order=1 content='{clean_content[:60]}...'"
            )
            plan_text = global_orchestrator.current_plan.original_text if global_orchestrator.current_plan else ""
            completion_payload = {
                "type": "bot_turn_complete",
                "turn_id": plan_turn_id,
                "bot": BOT_KEY,
                "order": 1,
                "content": clean_content,
                "plan_text": plan_text,
                "is_text_chat": False,
            }
            global_orchestrator.mark_order1_finished()
            if ctx.room and getattr(ctx.room, "local_participant", None):
                async def _safe_publish_orch(payload_bytes: bytes, t_id: str):
                    try:
                        logger.info(f"[Orchestration] bot={DISPLAY_NAME} event=publishing_datachannel topic=bot_orchestration turn_id={t_id}")
                        await ctx.room.local_participant.publish_data(payload_bytes, topic="bot_orchestration", reliable=True)
                        logger.info(f"[Orchestration] bot={DISPLAY_NAME} event=published_datachannel_success topic=bot_orchestration turn_id={t_id}")
                    except Exception as pub_err:
                        clean_err = sanitize_log_message(str(pub_err))
                        logger.error(f"[Orchestration] bot={DISPLAY_NAME} event=publish_completion_failed turn_id={t_id} error='{clean_err}'")

                try:
                    payload = json.dumps(completion_payload).encode("utf-8")
                    asyncio.create_task(_safe_publish_orch(payload, plan_turn_id))
                except Exception as pub_err:
                    clean_err = sanitize_log_message(str(pub_err))
                    logger.error(f"[Orchestration] Failed to schedule completion: {clean_err}")

    # 7. Text Chat Handling & Room Context Synchronization over DataChannel
    async def handle_text_chat_response(
        text: str,
        plan: MultiBotPlan | None = None,
        order: int = 1,
    ):
        try:
            clean_text = strip_speaker_labels(text)
            human_turn = room_history.create_human_turn(clean_text)
            added = room_history.add_turn(human_turn)
            if added and ctx.room and getattr(ctx.room, "local_participant", None):
                try:
                    payload = room_history.serialize_turn_event(human_turn)
                    await ctx.room.local_participant.publish_data(payload, topic="room_context", reliable=True)
                except Exception as pub_err:
                    clean_err = sanitize_log_message(str(pub_err))
                    logger.warning(f"[ROOM] event=context_sync_error error='{clean_err}'")

            sys_prompt = SYSTEM_PROMPT
            if plan and order == 1:
                my_target = next((t for t in plan.targets if t.bot == BOT_KEY), None)
                other_name = "Roxstar AI Sathi" if not IS_SATHI else "Roxstar AI Dost"
                if my_target:
                    enrichment = (
                        f"\n\n[Instruction for this turn]: The user asked you and {other_name} to respond in sequence. "
                        f"You are answering first. Focus on: {my_target.instruction}. "
                        f"Be concise and direct so {other_name} can add their part."
                    )
                    sys_prompt = sys_prompt + enrichment

            c_ctx = room_history.build_chat_context(sys_prompt, is_sathi=IS_SATHI)
            reply_text = ""
            try:
                stream = agent.llm.chat(chat_ctx=c_ctx)
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        reply_text += chunk.choices[0].delta.content
                if not reply_text:
                    reply_text = (
                        "Thoda technical issue aa gaya. Ek baar phir poochho."
                        if not IS_SATHI
                        else "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."
                    )
            except Exception as llm_err:
                clean_err = sanitize_log_message(str(llm_err))
                logger.error(f"[LLM] bot={DISPLAY_NAME} event=provider_error error='{clean_err}'")
                reply_text = (
                    "Thoda technical issue aa gaya. Ek baar phir poochho."
                    if not IS_SATHI
                    else "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."
                )

            if reply_text:
                clean_reply = strip_speaker_labels(reply_text)
                speaker = SPEAKER_SATHI if IS_SATHI else SPEAKER_DOST
                bot_turn = room_history.create_bot_turn(speaker, clean_reply)
                b_added = room_history.add_turn(bot_turn)
                if b_added and ctx.room and getattr(ctx.room, "local_participant", None):
                    try:
                        payload = room_history.serialize_turn_event(bot_turn)
                        await ctx.room.local_participant.publish_data(payload, topic="room_context", reliable=True)
                    except Exception as pub_err:
                        clean_err = sanitize_log_message(str(pub_err))
                        logger.warning(f"[ROOM] event=context_sync_error error='{clean_err}'")

                chat_msg = {
                    "id": str(int(time.time() * 1000)),
                    "sender": DISPLAY_NAME,
                    "text": clean_reply,
                    "timestamp": time.strftime("%I:%M %p"),
                    "isBot": True,
                    "botType": BOT_KEY,
                }
                payload = json.dumps(chat_msg).encode("utf-8")
                await ctx.room.local_participant.publish_data(payload, topic="chat", reliable=True)
                await agent.say(clean_reply, allow_interruptions=True)

                # Signal completion if order 1
                if plan and order == 1 and global_orchestrator.current_plan and not global_orchestrator.is_interrupted:
                    plan_turn_id = plan.turn_id
                    logger.info(
                        f"[Orchestration] bot={DISPLAY_NAME} event=publish_completion text_chat=true "
                        f"turn_id={plan_turn_id} order=1 content='{clean_reply[:60]}...'"
                    )
                    completion_payload = {
                        "type": "bot_turn_complete",
                        "turn_id": plan_turn_id,
                        "bot": BOT_KEY,
                        "order": 1,
                        "content": clean_reply,
                        "plan_text": plan.original_text if plan else "",
                        "is_text_chat": True,
                    }
                    global_orchestrator.mark_order1_finished()
                    if ctx.room and getattr(ctx.room, "local_participant", None):
                        try:
                            logger.info(f"[Orchestration] bot={DISPLAY_NAME} event=publishing_datachannel topic=bot_orchestration turn_id={plan_turn_id}")
                            await ctx.room.local_participant.publish_data(
                                json.dumps(completion_payload).encode("utf-8"),
                                topic="bot_orchestration",
                                reliable=True,
                            )
                            logger.info(f"[Orchestration] bot={DISPLAY_NAME} event=published_datachannel_success topic=bot_orchestration turn_id={plan_turn_id}")
                        except Exception as pub_err:
                            clean_err = sanitize_log_message(str(pub_err))
                            logger.error(f"[Orchestration] bot={DISPLAY_NAME} event=publish_completion_failed turn_id={plan_turn_id} error='{clean_err}'")
        except Exception as err:
            clean_err = sanitize_log_message(str(err))
            logger.error(f"[{DISPLAY_NAME}] Text chat response generation error: {clean_err}")

    @ctx.room.on("data_received")
    def on_data_received(dp: rtc.DataPacket):
        try:
            topic = getattr(dp, "topic", "")
            data_bytes = getattr(dp, "data", b"")
            sender_identity = getattr(getattr(dp, "participant", None), "identity", "unknown")

            logger.info(
                f"[DataChannel] bot={DISPLAY_NAME} event=data_received topic='{topic}' "
                f"bytes={len(data_bytes)} participant='{sender_identity}'"
            )

            # Topic "room_context": Room history turn synchronization across processes
            if topic == "room_context":
                turn = room_history.deserialize_turn_event(data_bytes)
                if turn:
                    room_history.add_turn(turn)
                return

            # Topic "bot_orchestration": Multi-bot ordering and completion coordination
            if topic == "bot_orchestration":
                try:
                    raw_str = data_bytes.decode("utf-8")
                    data = json.loads(raw_str)
                    msg_type = data.get("type", "")
                    plan_turn_id = data.get("turn_id", "")
                    received_bot = data.get("bot", "")
                    received_order = data.get("order")

                    my_pending = global_orchestrator.pending_order2_plan
                    my_pending_turn_id = my_pending.turn_id if my_pending else None

                    logger.info(
                        f"[Orchestration] bot={DISPLAY_NAME} event=orchestration_packet_parsed "
                        f"msg_type='{msg_type}' received_turn_id='{plan_turn_id}' received_bot='{received_bot}' "
                        f"received_order={received_order} my_pending_turn_id='{my_pending_turn_id}' "
                        f"has_pending_plan={my_pending is not None} is_interrupted={global_orchestrator.is_interrupted}"
                    )

                    if msg_type == "bot_plan_announced":
                        targets_data = data.get("targets", [])
                        my_target = next((t for t in targets_data if t.get("bot") == BOT_KEY), None)
                        if my_target and my_target.get("order") == 2:
                            logger.info(
                                f"[Orchestration] bot={DISPLAY_NAME} event=adopting_announced_plan "
                                f"plan_turn_id='{plan_turn_id}' my_order=2 instruction='{my_target.get('instruction')}'"
                            )
                            targets = [
                                MultiBotTarget(
                                    bot=t.get("bot", ""),
                                    order=int(t.get("order", 2)),
                                    instruction=t.get("instruction", "")
                                )
                                for t in targets_data
                            ]
                            announced_plan = MultiBotPlan(
                                turn_id=plan_turn_id,
                                targets=targets,
                                timestamp=float(data.get("timestamp", time.time())),
                                original_text=data.get("text", ""),
                            )
                            global_orchestrator.reset(announced_plan, my_order=2, is_text_chat=False)
                            global_orchestrator.pending_order2_plan = announced_plan
                            agent.interrupt(interrupt_all=True)

                    elif msg_type == "bot_turn_complete":
                        is_text_chat = bool(data.get("is_text_chat", False)) or getattr(global_orchestrator, "is_text_chat", False)
                        turn_id_match = (
                            my_pending is not None
                            and my_pending.turn_id == plan_turn_id
                        )
                        # Soft match fallback: if turn_id hashes diverged across processes due to STT formatting,
                        # release order 2 if we have a pending plan, received order 1 from the other bot, and not interrupted
                        order_match = (
                            my_pending is not None
                            and received_order == 1
                            and received_bot != BOT_KEY
                        )

                        # Secondary plan_text fallback if pending plan was not set prior to completion
                        plan_to_execute = my_pending
                        match_type = "exact_turn_id" if turn_id_match else ("order_fallback" if order_match else None)

                        if not plan_to_execute and received_order == 1 and received_bot != BOT_KEY:
                            plan_text = data.get("plan_text", "")
                            if plan_text:
                                reconstructed = parse_ordered_plan(plan_text)
                                if reconstructed and any(t.bot == BOT_KEY and t.order == 2 for t in reconstructed.targets):
                                    plan_to_execute = reconstructed
                                    match_type = "reconstructed_from_plan_text"

                        if plan_to_execute and not global_orchestrator.is_interrupted:
                            global_orchestrator.pending_order2_plan = None
                            first_content = data.get("content", "")
                            logger.info(
                                f"[Orchestration] bot={DISPLAY_NAME} event=order2_released "
                                f"plan_turn_id='{plan_to_execute.turn_id}' matched_by='{match_type}' "
                                f"first_bot='{received_bot}' first_content_len={len(first_content)} "
                                f"is_text_chat={is_text_chat}"
                            )
                            global_orchestrator.order2_task = asyncio.create_task(
                                _execute_order2_response(
                                    agent,
                                    plan_to_execute,
                                    room_history,
                                    ctx.room,
                                    first_bot_content=first_content,
                                    orchestrator=global_orchestrator,
                                    is_text_chat=is_text_chat,
                                )
                            )
                        else:
                            reasons = []
                            if plan_to_execute is None:
                                reasons.append("no_order2_plan")
                            if global_orchestrator.is_interrupted:
                                reasons.append("orchestrator_is_interrupted")
                            if not turn_id_match and not order_match and not plan_to_execute:
                                reasons.append("turn_and_order_mismatch")
                            logger.info(
                                f"[Orchestration] bot={DISPLAY_NAME} event=completion_ignored "
                                f"reasons='{', '.join(reasons)}' turn_id_match={turn_id_match} "
                                f"order_match={order_match}"
                            )
                    elif msg_type == "bot_plan_cancelled":
                        turn_id_match = (
                            my_pending is not None
                            and my_pending.turn_id == plan_turn_id
                        )
                        order_match = (
                            my_pending is not None
                            and received_bot != BOT_KEY
                        )
                        if turn_id_match or order_match:
                            logger.info(
                                f"[Orchestration] bot={DISPLAY_NAME} event=plan_cancelled_received "
                                f"plan_turn_id='{plan_turn_id}' from_bot='{received_bot}'"
                            )
                            global_orchestrator.cancel_current_plan()
                except Exception as orch_err:
                    clean_err = sanitize_log_message(str(orch_err))
                    logger.warning(f"[Orchestration] bot={DISPLAY_NAME} DataChannel receive error: {clean_err}")
                return

            if topic and topic != "chat":
                return

            # Topic "chat": Text chat messages from user / frontend
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

            if selected_agent == "both":
                plan = parse_ordered_plan(text)
                if plan:
                    my_target = next((t for t in plan.targets if t.bot == BOT_KEY), None)
                    if my_target:
                        if my_target.order == 1:
                            global_orchestrator.reset(plan, my_order=1, is_text_chat=True)
                            asyncio.create_task(handle_text_chat_response(text, plan=plan, order=1))
                        else:
                            global_orchestrator.reset(plan, my_order=2, is_text_chat=True)
                            global_orchestrator.pending_order2_plan = plan
                            logger.info(
                                f"[{DISPLAY_NAME}] [Orchestration] Text chat registered as order 2 for plan {plan.turn_id}. "
                                f"Awaiting order 1 completion."
                            )
            elif not IS_SATHI and selected_agent == "dost":
                asyncio.create_task(handle_text_chat_response(text))
            elif IS_SATHI and selected_agent == "sathi":
                asyncio.create_task(handle_text_chat_response(text))
        except Exception as e:
            clean_err = sanitize_log_message(str(e))
            logger.error(f"[{DISPLAY_NAME} DataChannel Error]: {clean_err}")

    # Participant Connection & Disconnection Event Listeners
    @ctx.room.on("participant_connected")
    def _on_room_participant_connected(p: rtc.RemoteParticipant):
        nonlocal human_participant
        logger.info(f"[CONNECTION] event=participant_connected identity='{p.identity}'")
        if is_human_participant(p):
            logger.info(f"[CONNECTION] Human participant joined: '{p.identity}'")
            human_participant = p
            if not getattr(agent, "_started", False):
                agent.start(ctx.room, participant=human_participant)

    @ctx.room.on("participant_disconnected")
    def _on_room_participant_disconnected(p: rtc.RemoteParticipant):
        nonlocal human_participant
        logger.info(f"[CONNECTION] event=participant_disconnected identity='{p.identity}'")
        if human_participant and human_participant.identity == p.identity:
            logger.info(f"[CONNECTION] Human participant '{p.identity}' left room.")
            human_participant = None
            agent.interrupt(interrupt_all=True)
            global_orchestrator.cancel_current_plan()

    # 8. Start VoicePipelineAgent linked exclusively to human participant
    if human_participant is not None:
        logger.info(f"[{DISPLAY_NAME}] LINKING AUDIO TRACK: target_identity='{human_participant.identity}', name='{human_participant.name}', kind='{human_participant.kind}' (Human Participant)")
        agent.start(ctx.room, participant=human_participant)
    else:
        logger.info(f"[{DISPLAY_NAME}] No human in room yet. Standing by for human participant to connect...")
        connected_event = asyncio.Event()

        def _on_participant_connected_init(p: rtc.RemoteParticipant):
            nonlocal human_participant
            if is_human_participant(p) and not connected_event.is_set():
                logger.info(f"[{DISPLAY_NAME}] LINKING AUDIO TRACK: human participant connected target_identity='{p.identity}', name='{p.name}', kind='{p.kind}'")
                human_participant = p
                connected_event.set()

        ctx.room.on("participant_connected", _on_participant_connected_init)
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
            clean_err = sanitize_log_message(str(say_err))
            logger.error(f"[{DISPLAY_NAME} Error] Failed to speak greeting: {clean_err}")
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

